#!/usr/bin/env bash
# Evaluate VP (per_layer_logit) with fixed parameters — no calibration training.
#
# Three modes:
#
#   (a) zero    — log_s=-inf, log_T=0: posterior variance collapses → recovers MAP mean.
#                 Use as sanity check that VP ≈ mean prediction.
#
#   (b) uncalib — log_s=0, log_T=0: full Laplace posterior, no temperature correction.
#                 Uncalibrated VP baseline.
#
#   (c) file    — log_s=0, log_T loaded from checkpoint: calibrated VP.
#                 Checkpoint auto-derived from outputs_laplace layout, or set CALIB_PARAMS.
#
# Env vars:
#   MODE          — "zero", "uncalib", or "file" (default: zero)
#   CALIB_PARAMS  — explicit checkpoint path (mode=file only; auto-derived if unset)
#   TASKS, SEEDS, LOAD_STEPS, SUFFIX — same semantics as calibrate_vp.sh

source "$(dirname "$0")/.venv/bin/activate"
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit here
# ══════════════════════════════════════════════════════════════════════════════
IFS=' ' read -r -a tasks      <<< "${TASKS:-ARC-Easy}"
IFS=' ' read -r -a seeds      <<< "${SEEDS:-21}"
IFS=' ' read -r -a load_steps <<< "${LOAD_STEPS:-4000}"
split_val=false
mode=${MODE:-zero}            # "zero" (sanity check), "uncalib", or "file" (load checkpoint)
results_dir=${RESULTS_DIR:-./results}

# MODE=zero: log_s=-inf → exp(-inf)=0 → LoRA variances collapse → recovers MAP mean
#            log_T=0, log_T_logit=0 → no temperature correction
# MODE=file: log_s=0.0 → exp(0)=1 → no posterior scaling; log_T loaded from checkpoint
init_log_T=0.0
init_log_T_logit=0.0

suffix=${SUFFIX:-}            # appended to log filenames and JSON base tag
# ══════════════════════════════════════════════════════════════════════════════

# ── Fixed settings ────────────────────────────────────────────────────────────
rms_norm_method=${RMS_NORM_METHOD:-mvp}
swiglu_method=${SWIGLU_METHOD:-exact}
calib_batch_size=${CALIB_BATCH_SIZE:-16}
grad_accum=${GRAD_ACCUM:-16}
n_mc_calib=${N_MC_CALIB:-1000}
n_mc_eval=${N_MC_EVAL:-1000}

declare -A seed_to_label=([21]=seed1 [42]=seed2 [87]=seed3 [13]=seed4 [100]=seed5)

model=meta-llama/Llama-2-7b-chat-hf
train_bs=4
eval_bs=8
max_len=300

# Matches laplace_output_dir formula in calibrate_vp.py
_peft_tag="lora_lmhead_16_0.1_5e-05"

# ── Script-level log ──────────────────────────────────────────────────────────
_model_tag="${model//\//__}"
_log_dir="logs_vp/${_model_tag}/${tasks[0]}"
mkdir -p "$_log_dir"
_script_log="${_log_dir}/${tasks[0]}_${seed_to_label[${seeds[0]}]}_${rms_norm_method}_${swiglu_method}_${mode}${suffix:+_${suffix}}.log"
exec > >(tee "$_script_log") 2>&1
echo "Logging to: $_script_log  [mode=${mode}]"

# ── Sweep ─────────────────────────────────────────────────────────────────────
for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        seed_label="${seed_to_label[$seed]}"

        for laplace_sub in all; do
            for laplace_hessian in kron; do
                for laplace_prior in homo; do
                    for laplace_optim_step in 100; do
                        for load_step in "${load_steps[@]}"; do

                            if [ "$task" = "boolq" ]; then
                                task_eval_bs=4
                            else
                                task_eval_bs=$eval_bs
                            fi

                            log_dir="logs_vp/${_model_tag}/${task}"
                            mkdir -p "$log_dir"
                            log_file="${log_dir}/${seed_label}_bs${train_bs}_maxlen${max_len}_sub${laplace_sub}_hess${laplace_hessian}_prior${laplace_prior}_step${laplace_optim_step}_loadstep${load_step}_rms${rms_norm_method}_swiglu${swiglu_method}_steps0_${mode}${suffix:+_${suffix}}.log"

                            # ── Mode-specific parameter args ──────────────────
                            _param_args=""

                            if [ "$mode" = "zero" ]; then
                                # (a) Sanity check: log_s=-100 ≈ -inf → exp≈0 → MAP mean
                                _fixed_log_s=-100.0
                                _param_args="--init_log_T ${init_log_T} --init_log_T_logit ${init_log_T_logit}"
                                echo "Running VP eval [ZERO / sanity]: $model | $task | $seed_label"

                            elif [ "$mode" = "uncalib" ]; then
                                # (b) Uncalibrated VP: log_s=0, log_T=0 → full posterior, no temperature
                                _fixed_log_s=0.0
                                _param_args="--init_log_T ${init_log_T} --init_log_T_logit ${init_log_T_logit}"
                                echo "Running VP eval [UNCALIB]: $model | $task | $seed_label"

                            else
                                # (c) Load pre-learned params from checkpoint.
                                # Auto-derive: first try outputs/, then results/ (for converted clean JSONs).
                                _fixed_log_s=0.0
                                _outputs_base="outputs/${task}/${model}_${_peft_tag}_${seed_label}/step_${load_step}"
                                _results_base="${results_dir}/${task}/${seed_label}"
                                _suffix_tag=${suffix:+_${suffix}}
                                _calib_params="${CALIB_PARAMS:-}"
                                if [ -z "$_calib_params" ]; then
                                    if [ -f "${_outputs_base}/calib_params_per_layer_logit.pt" ]; then
                                        _calib_params="${_outputs_base}/calib_params_per_layer_logit.pt"
                                    elif [ -f "${_results_base}/calib_params_${swiglu_method}_${rms_norm_method}${_suffix_tag}.pt" ]; then
                                        _calib_params="${_results_base}/calib_params_${swiglu_method}_${rms_norm_method}${_suffix_tag}.pt"
                                    fi
                                fi
                                if [ ! -f "$_calib_params" ]; then
                                    echo "ERROR: checkpoint not found (tried outputs/ and results/) — skipping"
                                    continue
                                fi
                                _param_args="--load_calib_params $_calib_params"
                                echo "Running VP eval [FILE]: $model | $task | $seed_label | calib=$_calib_params"
                            fi

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
                                --total_steps         0 \
                                --calib_batch_size    $calib_batch_size \
                                --grad_accum          $grad_accum \
                                --lr_min_factor       1.0 \
                                --lr_logit_factor     1.0 \
                                --n_mc_calib          $n_mc_calib \
                                --n_mc_eval           $n_mc_eval \
                                --fixed_log_s         $_fixed_log_s \
                                $_param_args \
                                ${suffix:+--suffix "$suffix"} \
                                --results_dir         $results_dir \
                                ${MERGE_RESULTS_INTO:+--merge_results_into "$MERGE_RESULTS_INTO"} \
                                ${VAL_CALIB_KEY:+--val_calib_key "$VAL_CALIB_KEY"} \
                                ${TEST_CALIB_KEY:+--test_calib_key "$TEST_CALIB_KEY"} \
                                $( [ "${EVAL_ON_TEST:-0}" = "1" ] && echo "--eval_on_test" ) \
                                2>&1 | tee "$log_file"

                        done
                    done
                done
            done
        done
    done
done
