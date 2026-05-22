#!/usr/bin/env bash
# Calibrate GLM predictive variance scale (log_s_pred) on val NLL.
# No model loading — uses precomputed f_mu / f_var tensors from outputs_laplace.
# Mean logits are never modified; only exp(log_s_pred) * JΣJᵀ is scaled.

source "$(dirname "$0")/.venv/bin/activate"

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit here
# ══════════════════════════════════════════════════════════════════════════════
IFS=' ' read -r -a tasks      <<< "${TASKS:-ARC-Easy openbookqa}"
IFS=' ' read -r -a seeds      <<< "${SEEDS:-21 42 87 13 100}"
IFS=' ' read -r -a load_steps <<< "${LOAD_STEPS:-4000}"
suffix=${SUFFIX:-}
results_dir=${RESULTS_DIR:-./results}
# ══════════════════════════════════════════════════════════════════════════════

total_steps=${TOTAL_STEPS:-200}
lr=${LR:-1e-1}
n_mc_calib=${N_MC_CALIB:-1000}
n_mc_eval=${N_MC_EVAL:-100000}

declare -A seed_to_label=([21]=seed1 [42]=seed2 [87]=seed3 [13]=seed4 [100]=seed5)

model=meta-llama/Llama-2-7b-chat-hf
laplace_hessian=kron
laplace_sub=all
laplace_prior=homo
laplace_optim_step=100

# ── Script-level log ──────────────────────────────────────────────────────────
_model_tag="${model//\//__}"
_log_dir="logs_vp/${_model_tag}/${tasks[0]}"
mkdir -p "$_log_dir"
_script_log="${_log_dir}/${tasks[0]}_${seed_to_label[${seeds[0]}]}_glm${suffix:+_${suffix}}.log"
exec > >(tee "$_script_log") 2>&1
echo "Logging to: $_script_log"

# ── Sweep ─────────────────────────────────────────────────────────────────────
for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        seed_label="${seed_to_label[$seed]}"

        for load_step in "${load_steps[@]}"; do

            log_dir="logs_vp/${_model_tag}/${task}"
            mkdir -p "$log_dir"
            log_file="${log_dir}/${seed_label}_glm_loadstep${load_step}_steps${total_steps}${suffix:+_${suffix}}.log"

            echo "Running GLM calibration: $model | $task | $seed_label | load_step=$load_step"
            python calibrate_glm.py \
                --model_name_or_path  $model \
                --task_name           $task \
                --seed                $seed \
                --seed_label          $seed_label \
                --load_step           $load_step \
                --laplace_hessian     $laplace_hessian \
                --laplace_sub         $laplace_sub \
                --laplace_prior       $laplace_prior \
                --laplace_optim_step  $laplace_optim_step \
                --lm_head \
                --total_steps         $total_steps \
                --lr                  $lr \
                --n_mc_calib          $n_mc_calib \
                --n_mc_eval           $n_mc_eval \
                --also_eval_test \
                --results_dir         $results_dir \
                ${suffix:+--suffix "$suffix"} \
                2>&1 | tee "$log_file"

        done
    done
done
