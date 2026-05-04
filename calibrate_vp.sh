#!/usr/bin/env bash
# Companion script for calibrate_vp.py — fits KFAC Laplace and calibrates
# VP temperatures (log_s, log_T) on the validation NLL.
#
# Which calibration variants run is controlled by the RUN_* flags at the top
# of calibrate_vp.py.  This script controls which Laplace posterior is used:
#
#   laplace_sub=last_layer  — Laplace fits only on lm_head LoRA; backbone VP
#                             has zero variance (phase-1 log_s useless for backbone).
#   laplace_sub=all         — Laplace fits on ALL LoRA weights (q_proj, v_proj,
#                             lm_head); required for all variants including phase-1.

source "$(dirname "$0")/.venv/bin/activate"
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit here
# ══════════════════════════════════════════════════════════════════════════════
#tasks=(winogrande_s ARC-Challenge ARC-Easy winogrande_m openbookqa boolq)
#seeds=(21 42 87 13 100)
IFS=' ' read -r -a tasks      <<< "${TASKS:-ARC-Easy}"
IFS=' ' read -r -a seeds      <<< "${SEEDS:-21}"
IFS=' ' read -r -a load_steps <<< "${LOAD_STEPS:-4000}"
split_val=false   # true → split val 50/50 (calib / eval); false → use full val for both
# ══════════════════════════════════════════════════════════════════════════════

# ── VP calibration hypers  (all overridable via env vars of the same name) ────
rms_norm_method=${RMS_NORM_METHOD:-mvp}        # "streamlined" or "mvp"
swiglu_method=${SWIGLU_METHOD:-exact}          # "delta" or "exact"
total_steps=${TOTAL_STEPS:-200}                 # total optimizer steps; n_epochs computed from loader size
calib_batch_size=${CALIB_BATCH_SIZE:-16}       # micro-batch size per gradient step
grad_accum=${GRAD_ACCUM:-16}                   # gradient accumulation steps
lr=${LR:-1e-1}                                 # LR for all calibration parameters
finetune_s=${FINETUNE_S:-true}              # true → optimize log_s at lr_s_finetune in backbone phase; false → freeze
lr_s_finetune=${LR_S_FINETUNE:-1e-1}          # LR for log_s in backbone phase (only if finetune_s=true)
init_log_s=${INIT_LOG_S:-}               # sanity check: -100 → var≈0 → should recover MAP NLL
n_mc_calib=${N_MC_CALIB:-1}                    # sanity check: var=0 so 1 sample is identical to mean
n_mc_eval=${N_MC_EVAL:-1}                     # sanity check: same

# ── Fixed settings ────────────────────────────────────────────────────────────
declare -A seed_to_label=([21]=seed1 [42]=seed2 [87]=seed3 [13]=seed4 [100]=seed5)

model=meta-llama/Llama-2-7b-chat-hf
#model=TinyLlama/TinyLlama-1.1B-Chat-v1.0

train_bs=4
eval_bs=8
max_len=300

# ── Script-level log ─────────────────────────────────────────────────────────
# Captures all output (echo + python) for the primary task/seed combination.
# Named: logs_vp/<model>/<task>_<seed>_<rms>_<swiglu>.log
_model_tag="${model//\//__}"
_log_dir="logs_vp/${_model_tag}/${tasks[0]}"
mkdir -p "$_log_dir"
_script_log="${_log_dir}/${tasks[0]}_${seed_to_label[${seeds[0]}]}_${rms_norm_method}_${swiglu_method}.log"
exec > >(tee "$_script_log") 2>&1
echo "Logging to: $_script_log"

# ── Sweep ─────────────────────────────────────────────────────────────────────
for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        seed_label="${seed_to_label[$seed]}"

        # last_layer → Laplace on lm_head LoRA only (cheap, ~half the runtime)
        # all        → Laplace on all LoRA weights (needed for backbone VP variants)
        for laplace_sub in all; do            # last_layer (lm_head only) | all (all LoRA)
            for laplace_hessian in diag kron; do    # kron (Kronecker) | diag (diagonal)
                for laplace_prior in homo; do  # homo (shared scalar) | hetero (per-layer)
                    for laplace_optim_step in 100; do
                        for load_step in "${load_steps[@]}"; do

                            # boolq has long passages; reduce eval batch size to avoid OOM
                            if [ "$task" = "boolq" ]; then
                                task_eval_bs=4
                            else
                                task_eval_bs=$eval_bs
                            fi

                            log_dir="logs_vp/${_model_tag}/${task}"
                            mkdir -p "$log_dir"
                            log_file="${log_dir}/${seed_label}_bs${train_bs}_maxlen${max_len}_sub${laplace_sub}_hess${laplace_hessian}_prior${laplace_prior}_step${laplace_optim_step}_loadstep${load_step}_rms${rms_norm_method}_swiglu${swiglu_method}_steps${total_steps}.log"

                            echo "Running VP calibration: $model | $task | $seed_label | sub=$laplace_sub | rms=$rms_norm_method | swiglu=$swiglu_method | total_steps=$total_steps"
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
                                --swiglu_method       $swiglu_method \
                                --total_steps         $total_steps \
                                --calib_batch_size    $calib_batch_size \
                                --grad_accum          $grad_accum \
                                --lr                  $lr \
                                $( [ "$finetune_s" = true ] && echo "--finetune_s" ) \
                                --lr_s_finetune       $lr_s_finetune \
                                --n_mc_calib          $n_mc_calib \
                                --n_mc_eval           $n_mc_eval \
                                $( [ -n "$init_log_s" ] && echo "--init_log_s $init_log_s" ) \
                                2>&1 | tee "$log_file"

                        done
                    done
                done
            done
        done
    done
done
