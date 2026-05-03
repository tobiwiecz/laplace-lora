#!/usr/bin/env bash
# Companion script for calibrate_vp.py — fits KFAC Laplace and calibrates
# VP temperatures (log_s, log_T) on the validation NLL.
#
# Which calibration variants run is controlled by the RUN_* flags at the top
# of calibrate_vp.py.  This script controls which Laplace posterior is used:
#
#   laplace_sub=last_layer  — Laplace fits only on lm_head LoRA;
#                             use when RUN_LAST_LAYER=True only.
#   laplace_sub=all         — Laplace fits on ALL LoRA weights (q_proj, v_proj,
#                             lm_head); required for meaningful backbone VP
#                             (RUN_GLOBAL / RUN_PER_LAYER / RUN_PER_SUB_BLOCK).

source "$(dirname "$0")/.venv/bin/activate"
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit here
# ══════════════════════════════════════════════════════════════════════════════
#tasks=(winogrande_s ARC-Challenge ARC-Easy winogrande_m openbookqa boolq)
#seeds=(21 42 87 13 100)
tasks=(winogrande_s)
seeds=(21)
load_steps=(4000)
split_val=false   # true → split val 50/50 (calib / eval); false → use full val for both
# ══════════════════════════════════════════════════════════════════════════════

# ── VP calibration hypers ─────────────────────────────────────────────────────
rms_norm_method=${RMS_NORM_METHOD:-mvp}        # "streamlined" or "mvp"
n_epochs=${N_EPOCHS:-20}                       # calibration epochs
calib_batch_size=${CALIB_BATCH_SIZE:-16}       # micro-batch size per gradient step
grad_accum=${GRAD_ACCUM:-16}                   # gradient accumulation steps
lr=${LR:-1e-1}                                 # LR for all calibration parameters
finetune_s=${FINETUNE_S:-true}               # true → optimize log_s at lr_s_finetune in backbone phase; false → freeze
lr_s_finetune=${LR_S_FINETUNE:-1e-2}          # LR for log_s in backbone phase (only if finetune_s=true)
n_mc_calib=${N_MC_CALIB:-100}                  # MC samples during calibration
n_mc_eval=${N_MC_EVAL:-1000}                  # MC samples for final evaluation

# ── Fixed settings ────────────────────────────────────────────────────────────
declare -A seed_to_label=([21]=seed1 [42]=seed2 [87]=seed3 [13]=seed4 [100]=seed5)

model=meta-llama/Llama-2-7b-chat-hf
#model=TinyLlama/TinyLlama-1.1B-Chat-v1.0

train_bs=4
eval_bs=8
max_len=300

# ── Sweep ─────────────────────────────────────────────────────────────────────
for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        seed_label="${seed_to_label[$seed]}"

        # last_layer → Laplace on lm_head LoRA only (cheap, ~half the runtime)
        # all        → Laplace on all LoRA weights (needed for backbone VP variants)
        for laplace_sub in all; do            # last_layer (lm_head only) | all (all LoRA)
            for laplace_hessian in kron; do    # kron (Kronecker) | diag (diagonal)
                for laplace_prior in homo; do  # homo (shared scalar) | hetero (per-layer)
                    for laplace_optim_step in 100; do
                        for load_step in "${load_steps[@]}"; do

                            # boolq has long passages; reduce eval batch size to avoid OOM
                            if [ "$task" = "boolq" ]; then
                                task_eval_bs=4
                            else
                                task_eval_bs=$eval_bs
                            fi

                            model_tag="${model//\//__}"
                            log_dir="logs_vp/${model_tag}/${task}"
                            mkdir -p "$log_dir"
                            log_file="${log_dir}/${seed_label}_bs${train_bs}_maxlen${max_len}_sub${laplace_sub}_hess${laplace_hessian}_prior${laplace_prior}_step${laplace_optim_step}_loadstep${load_step}_rms${rms_norm_method}_epochs${n_epochs}.log"

                            echo "Running VP calibration: $model | $task | $seed_label | sub=$laplace_sub | rms=$rms_norm_method | n_epochs=$n_epochs"
                            accelerate launch --num_processes 1 calibrate_vp.py \
                                --model_name_or_path  $model \
                                --task_name           $task \
                                --seed                $seed \
                                --seed_label          $seed_label \
                                --laplace_sub         $laplace_sub \
                                --laplace_hessian     $laplace_hessian \
                                --laplace_prior       $laplace_prior \
                                --laplace_optim_step  $laplace_optim_step \
                                --load_step           $load_step \
                                --per_device_train_batch_size $train_bs \
                                --per_device_eval_batch_size  $task_eval_bs \
                                --max_length          $max_len \
                                --testing_set         $( [ "$split_val" = true ] && echo test || echo val ) \
                                --lm_head \
                                --rms_norm_method     $rms_norm_method \
                                --n_epochs            $n_epochs \
                                --calib_batch_size    $calib_batch_size \
                                --grad_accum          $grad_accum \
                                --lr                  $lr \
                                $( [ "$finetune_s" = true ] && echo "--finetune_s" ) \
                                --lr_s_finetune       $lr_s_finetune \
                                --n_mc_calib          $n_mc_calib \
                                --n_mc_eval           $n_mc_eval \
                                2>&1 | tee "$log_file"

                        done
                    done
                done
            done
        done
    done
done
