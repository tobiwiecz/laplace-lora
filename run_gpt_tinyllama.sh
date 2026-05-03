source "$(dirname "$0")/.venv/bin/activate"
export TORCHDYNAMO_DISABLE=1

#tasks=(winogrande_s ARC-Challenge ARC-Easy winogrande_m openbookqa boolq)

tasks=(${TASKS:-winogrande_s})
seeds=(${SEEDS:-21 42 87 13 100})

declare -A seed_to_label=([21]=seed1 [42]=seed2 [87]=seed3 [13]=seed4 [100]=seed5)

model=TinyLlama/TinyLlama-1.1B-Chat-v1.0

train_bs=8
eval_bs=8
max_len=300

for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        seed_label="${seed_to_label[$seed]}"
        model_tag="${model//\//__}"
        log_dir="logs/${model_tag}/${task}"
        mkdir -p "$log_dir"
        log_file="${log_dir}/${seed_label}_bs${train_bs}_maxlen${max_len}.log"

        echo "Running $model on task $task with $seed_label (seed=$seed)"
        accelerate launch --num_processes 1 run_gpt.py \
            --model_name_or_path $model \
            --task_name $task \
            --seed $seed \
            --seed_label $seed_label \
            --per_device_train_batch_size $train_bs \
            --per_device_eval_batch_size $eval_bs \
            --max_length $max_len \
            --max_train_steps 4000 \
            --checkpointing_steps 500 \
            --testing_set val \
            --lm_head \
            2>&1 | tee "$log_file"
    done
done
