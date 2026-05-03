source "$(dirname "$0")/.venv/bin/activate"
export TORCHDYNAMO_DISABLE=1

tasks=(${TASKS:-winogrande_s})
seeds=(${SEEDS:-21})
load_steps=(${LOAD_STEPS:-4000})
n_samples=(${N_SAMPLES:-1 2 4 8 16})
sampling_methods=(${SAMPLING_METHODS:-kron diag})

declare -A seed_to_label=([21]=seed1 [42]=seed2 [87]=seed3 [13]=seed4 [100]=seed5)

model=meta-llama/Llama-2-7b-chat-hf
#model=TinyLlama/TinyLlama-1.1B-Chat-v1.0

eval_bs=8
max_len=300

for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        seed_label="${seed_to_label[$seed]}"
        for laplace_sub in all last_layer; do
            for laplace_hessian in kron; do
                for laplace_prior in homo; do
                    for laplace_optim_step in 100; do
                        for load_step in "${load_steps[@]}"; do
                        for sampling_method in "${sampling_methods[@]}"; do
                        model_tag="${model//\//__}"
                        log_dir="logs/${model_tag}/${task}"
                        mkdir -p "$log_dir"
                        max_n=${n_samples[-1]}
                        log_file="${log_dir}/${seed_label}_maxlen${max_len}_sub${laplace_sub}_hess${laplace_hessian}_prior${laplace_prior}_step${laplace_optim_step}_loadstep${load_step}_nn_sampling_${sampling_method}_max${max_n}.log"

                        echo "Running NN sampling/${sampling_method} (${n_samples[*]} samples) on $model, task=$task, $seed_label, sub=$laplace_sub, hessian=$laplace_hessian, load_step=$load_step"
                        accelerate launch --num_processes 1 run_gpt_laplace_nn_sampling.py \
                            --model_name_or_path $model \
                            --task_name $task \
                            --seed $seed \
                            --seed_label $seed_label \
                            --laplace_sub $laplace_sub \
                            --laplace_hessian $laplace_hessian \
                            --laplace_prior $laplace_prior \
                            --laplace_optim_step $laplace_optim_step \
                            --load_step $load_step \
                            --per_device_eval_batch_size $eval_bs \
                            --max_length $max_len \
                            --testing_set val \
                            --lm_head \
                            --sampling_method $sampling_method \
                            --n_samples "${n_samples[@]}" \
                            2>&1 | tee "$log_file"
                        done
                        done
                    done
                done
            done
        done
    done
done
