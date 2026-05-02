source "$(dirname "$0")/.venv/bin/activate"
export TORCHDYNAMO_DISABLE=1

#tasks=(winogrande_s ARC-Challenge ARC-Easy winogrande_m openbookqa boolq)
#seeds=(21 42 87)
#steps=(0 999 1999 2999 3999 4999 5999 6999 7999 8999 9999)
#splits=(train val test)

tasks=(${TASKS:-winogrande_s})
seeds=(${SEEDS:-21})
steps=(${STEPS:-999})
splits=(${SPLITS:-val})

tasks=(winogrande_s)
seeds=(21)
steps=(0 999 1999 2999 3999 4999 5999 6999 7999 8999 9999)
splits=(train val test)

model=meta-llama/Llama-2-7b-chat-hf
#model=TinyLlama/TinyLlama-1.1B-Chat-v1.0

eval_bs=4
max_len=300

# LoRA hyperparams — must match the training run used to produce the checkpoints
lora_alpha=16
lora_dropout=0.1
lr=5e-05
peft_method=lora_lmhead

splits_tag="${splits[*]// /_}"

for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        for step in "${steps[@]}"; do
            checkpoint_dir="outputs/${task}/${model}_${peft_method}_${lora_alpha}_${lora_dropout}_${lr}_${seed}/step_${step}"

            model_tag="${model//\//__}"
            log_dir="logs/${model_tag}/${task}"
            mkdir -p "$log_dir"
            log_file="${log_dir}/eval_seed${seed}_step${step}_splits${splits_tag}_maxlen${max_len}.log"

            echo "Evaluating $model on task $task | seed=$seed step=$step splits=${splits[*]}"
            accelerate launch --num_processes 1 eval_gpt.py \
                --model_name_or_path $model \
                --task_name $task \
                --checkpoint_dir "$checkpoint_dir" \
                --splits ${splits[@]} \
                --seed $seed \
                --per_device_eval_batch_size $eval_bs \
                --max_length $max_len \
                2>&1 | tee "$log_file"
        done
    done
done
