source "$(dirname "$0")/.venv/bin/activate"
export TORCHDYNAMO_DISABLE=1

#tasks=(winogrande_s ARC-Challenge ARC-Easy winogrande_m openbookqa boolq)
#seeds=(21 42 87)

tasks=(${TASKS:-winogrande_s})
seeds=(${SEEDS:-21 42 87 13 100})
load_steps=(${LOAD_STEPS:-1000})

model=meta-llama/Llama-2-7b-chat-hf
#model=TinyLlama/TinyLlama-1.1B-Chat-v1.0

train_bs=4
eval_bs=4
max_len=300

for task in "${tasks[@]}"; do
    for i in "${!seeds[@]}"; do
        seed=${seeds[$i]}
        seed_label="seed$((i+1))"
        for laplace_sub in all last_layer; do
            for laplace_hessian in kron; do
                for laplace_prior in homo; do
                    for laplace_optim_step in 100; do
                        for load_step in "${load_steps[@]}"; do
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
                            --per_device_eval_batch_size $eval_bs \
                            --max_length $max_len \
                            --testing_set val \
                            --lm_head \
                            2>&1 | tee "$log_file"
                        done
                    done
                done
            done
        done
    done
done
