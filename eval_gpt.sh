source "$(dirname "$0")/.venv/bin/activate"
export TORCHDYNAMO_DISABLE=1

#tasks=(winogrande_s ARC-Challenge ARC-Easy winogrande_m openbookqa boolq)
#seeds=(21 42 87)
#steps=(4000)
#splits=(train val test)

IFS=' ' read -r -a tasks  <<< "${TASKS:-ARC-Easy openbookqa ARC-Challenge}"
IFS=' ' read -r -a seeds  <<< "${SEEDS:-21 42 87 13 100}"
IFS=' ' read -r -a steps  <<< "${STEPS:-4000}"
IFS=' ' read -r -a splits <<< "${SPLITS:-val test}"
results_dir=${RESULTS_DIR:-./results}

declare -A seed_to_label=([21]=seed1 [42]=seed2 [87]=seed3 [13]=seed4 [100]=seed5)

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
        seed_label="${seed_to_label[$seed]}"
        for step in "${steps[@]}"; do
            checkpoint_dir="outputs/${task}/${model}_${peft_method}_${lora_alpha}_${lora_dropout}_${lr}_${seed_label}/step_${step}"

            model_tag="${model//\//__}"
            log_dir="logs/${model_tag}/${task}"
            mkdir -p "$log_dir"
            log_file="${log_dir}/eval_${seed_label}_step${step}_splits${splits_tag}_maxlen${max_len}.log"

            echo "Evaluating $model on task $task | seed=$seed step=$step splits=${splits[*]}"
            accelerate launch --num_processes 1 eval_gpt.py \
                --model_name_or_path $model \
                --task_name $task \
                --checkpoint_dir "$checkpoint_dir" \
                --splits ${splits[@]} \
                --seed $seed \
                --seed_label $seed_label \
                --per_device_eval_batch_size $eval_bs \
                --max_length $max_len \
                --results_dir $results_dir \
                2>&1 | tee "$log_file"
        done
    done
done
