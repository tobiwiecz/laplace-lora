"""
Consolidate per-method result files into two summary JSONs per experiment:
  all_results_full.json       — full LoRA Laplace  (laplace_sub=all)
  all_results_lastlayer.json  — last-layer Laplace (laplace_sub=last_layer)

Each file contains one block per method:
  "mean"                — MAP / deterministic LoRA (from all_results.json)
  "linearized_laplace"  — GLM predictive           (from all_results_la_kron_{sub}_*.json)
  "1_sample"            — NN weight-space sampling  (from all_results_la_nn_sampling_*.json)
  "2_samples"           ...
  ...

Run after both run_gpt_laplace.py and run_gpt_laplace_nn_sampling.py have finished.
NN sampling results are optional — the file is written with whatever is available.

Usage (mirrors the arg style of run_gpt_laplace.py):
  python consolidate_results.py \
      --task_name ARC-Challenge \
      --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
      --lora_alpha 16 --lora_dropout 0.1 --learning_rate 5e-05 \
      --seed_label seed1 \
      --load_step 4000 \
      --laplace_optim_step 100 \
      --n_samples 1 2 4 8
"""
import argparse
import json
import os
import glob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_name",           type=str, required=True)
    p.add_argument("--model_name_or_path",  type=str, default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--output_dir",          type=str, default="./outputs")
    p.add_argument("--lora_alpha",          type=int, default=16)
    p.add_argument("--lora_dropout",        type=float, default=0.1)
    p.add_argument("--learning_rate",       type=float, default=5e-5)
    p.add_argument("--seed",                type=int, default=21)
    p.add_argument("--seed_label",          type=str, default=None)
    p.add_argument("--load_step",           type=int, default=4000)
    p.add_argument("--lm_head",             action="store_true", default=True)
    p.add_argument("--testing_set",         type=str, default="val")
    p.add_argument("--laplace_hessian",     type=str, default="kron")
    p.add_argument("--laplace_prior",       type=str, default="homo")
    p.add_argument("--laplace_optim_step",  type=int, default=100)
    p.add_argument("--laplace_predict",     type=str, default="mc_corr")
    p.add_argument("--n_samples",           type=int, nargs="+", default=[1, 2, 4, 8],
                   help="Sample counts used when running run_gpt_laplace_nn_sampling.py")
    return p.parse_args()


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def consolidate(step_dir, sub, args):
    h    = args.laplace_hessian
    pr   = args.laplace_prior
    opt  = args.laplace_optim_step
    pred = args.laplace_predict
    max_n = max(args.n_samples)

    result = {}

    map_data = load_json(os.path.join(step_dir, "all_results.json"))
    if map_data is not None:
        result["mean"] = map_data

    glm_data = load_json(os.path.join(step_dir, f"all_results_la_{h}_{sub}_{pr}_{pred}_{opt}.json"))
    if glm_data is not None:
        result["linearized_laplace"] = glm_data

    for method in ['kron', 'diag']:
        sampling_data = load_json(os.path.join(step_dir, f"all_results_la_nn_sampling_{method}_max{max_n}_{h}_{sub}_{pr}_{opt}.json"))
        if sampling_data is not None:
            for k in sorted(args.n_samples):
                key = f"{k}_sample" if k == 1 else f"{k}_samples"
                if key in sampling_data:
                    result[f"{method}_{key}"] = sampling_data[key]

    return result


def run(args):
    """Consolidate one experiment; returns a short status string."""
    peft_method = "lora_lmhead" if args.lm_head else "lora"
    if args.testing_set != "val":
        peft_method += args.testing_set

    seed_label = args.seed_label if args.seed_label is not None else str(args.seed)
    step_dir = os.path.join(
        args.output_dir,
        args.task_name,
        f"{args.model_name_or_path}_{peft_method}_{args.lora_alpha}_{args.lora_dropout}_{args.learning_rate}_{seed_label}",
        f"step_{args.load_step}",
    )

    if not os.path.isdir(step_dir):
        raise FileNotFoundError(f"Run directory not found: {step_dir}")

    sub_to_outfile = {
        "all":        "all_results_full.json",
        "last_layer": "all_results_lastlayer.json",
    }

    written = []
    for sub, outfile in sub_to_outfile.items():
        result = consolidate(step_dir, sub, args)
        if not result:
            continue
        with open(os.path.join(step_dir, outfile), "w") as f:
            json.dump(result, f, indent=2)
        written.append(f"{outfile}({'+'.join(result.keys())})")

    return "  ".join(written)


def main():
    args = parse_args()
    status = run(args)
    print(status)


if __name__ == "__main__":
    main()
