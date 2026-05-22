source "$(dirname "$0")/.venv/bin/activate"
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#tasks=(winogrande_s ARC-Challenge ARC-Easy winogrande_m openbookqa boolq)
#seeds=(21 42 87 13 100)

tasks=(${TASKS:-ARC-Easy openbookqa})
seeds=(${SEEDS:-21 42 87 13 100})
load_steps=(${LOAD_STEPS:-4000})

# Optional flags — set to 1 to enable
load_laplace=${LOAD_LAPLACE:-0}      # 1 → load saved H + prior_precision from disk instead of refitting
skip_val_eval=${SKIP_VAL_EVAL:-0}   # 1 → skip val inference loop (f_mu/f_var already computed)
also_eval_test=${ALSO_EVAL_TEST:-0} # 1 → run GLM inference on HF test split and save tensors

declare -A seed_to_label=([21]=seed1 [42]=seed2 [87]=seed3 [13]=seed4 [100]=seed5)

model=meta-llama/Llama-2-7b-chat-hf
#model=TinyLlama/TinyLlama-1.1B-Chat-v1.0

train_bs=4
eval_bs=8
max_len=300

for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        seed_label="${seed_to_label[$seed]}"
        for laplace_sub in all; do
            for laplace_hessian in diag kron; do
                for laplace_prior in homo; do
                    for laplace_optim_step in 100; do
                        for load_step in "${load_steps[@]}"; do
                        # boolq has long passages; reduce GLM predictive eval batch size to avoid OOM
                        if [ "$task" = "boolq" ]; then
                            task_eval_bs=4
                        else
                            task_eval_bs=$eval_bs
                        fi

                        model_tag="${model//\//__}"
                        log_dir="logs/${model_tag}/${task}"
                        mkdir -p "$log_dir"
                        log_file="${log_dir}/${seed_label}_bs${train_bs}_maxlen${max_len}_sub${laplace_sub}_hess${laplace_hessian}_prior${laplace_prior}_step${laplace_optim_step}_loadstep${load_step}.log"

                        echo "Running $model on task $task with $seed_label (seed=$seed, sub=$laplace_sub, hessian=$laplace_hessian, prior=$laplace_prior, optim_step=$laplace_optim_step, load_step=$load_step)"
                        accelerate launch --num_processes 1 run_gpt_laplace.py \
                            --model_name_or_path $model \
                            --task_name $task \
                            --seed $seed \
                            --seed_label $seed_label \
                            --laplace_sub $laplace_sub \
                            --laplace_hessian $laplace_hessian \
                            --laplace_prior $laplace_prior \
                            --laplace_optim_step $laplace_optim_step \
                            --load_step $load_step \
                            --per_device_train_batch_size $train_bs \
                            --per_device_eval_batch_size $task_eval_bs \
                            --max_length $max_len \
                            --testing_set val \
                            --lm_head \
                            $( [ "$load_laplace"   = "1" ] && echo "--load_laplace" ) \
                            $( [ "$skip_val_eval"  = "1" ] && echo "--skip_val_eval" ) \
                            $( [ "$also_eval_test" = "1" ] && echo "--also_eval_test" ) \
                            2>&1 | tee "$log_file"
                        done
                    done
                done
            done
        done
    done
done
