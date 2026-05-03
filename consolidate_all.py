"""
Auto-consolidate results for every (task, seed, step) that has any Laplace result file,
for a given model.

Usage:
  python consolidate_all.py
  python consolidate_all.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  python consolidate_all.py --model meta-llama/Llama-2-7b-chat-hf --n_samples 1 2 4 8 16
"""
import argparse
import glob
import os
import sys

from consolidate_results import run as consolidate_run


# Ordered from longest to shortest so we match the most specific prefix first
PEFT_OPTIONS = [
    ("lora_lmheadtrain_val", True,  "train_val"),
    ("lora_lmheadtest",      True,  "test"),
    ("lora_lmhead",          True,  "val"),
    ("loratrain_val",        False, "train_val"),
    ("loratest",             False, "test"),
    ("lora",                 False, "val"),
]


def parse_run_dir_name(run_dir_name, model_name):
    """
    Parse e.g. 'Llama-2-7b-chat-hf_lora_lmhead_16_0.1_5e-05_seed1'
    into (lora_alpha, lora_dropout, learning_rate, seed_label, lm_head, testing_set).
    Returns None on failure.
    """
    prefix = model_name + "_"
    if not run_dir_name.startswith(prefix):
        return None
    suffix = run_dir_name[len(prefix):]  # e.g. lora_lmhead_16_0.1_5e-05_seed1

    lm_head = True
    testing_set = "val"
    params_str = None
    for peft_method, lm, ts in PEFT_OPTIONS:
        if suffix.startswith(peft_method + "_"):
            lm_head = lm
            testing_set = ts
            params_str = suffix[len(peft_method) + 1:]
            break

    if params_str is None:
        return None

    # params_str = '16_0.1_5e-05_seed1'
    parts = params_str.split("_")
    if len(parts) < 4:
        return None

    alpha      = parts[0]           # e.g. '16'
    dropout    = parts[1]           # e.g. '0.1'
    lr         = parts[2]           # e.g. '5e-05'
    seed_label = "_".join(parts[3:])  # e.g. 'seed1'

    return alpha, dropout, lr, seed_label, lm_head, testing_set


def discover_step_dirs(output_dir, model):
    """
    Find all step dirs that have at least one Laplace result file for the given model.
    Returns a sorted list of step_dir paths.

    The model string (e.g. 'meta-llama/Llama-2-7b-chat-hf') maps to two directory
    levels in the outputs tree, so we split on '/' and glob accordingly.
    """
    model_org  = model.split("/")[0]           # e.g. 'meta-llama'
    model_name = os.path.basename(model)       # e.g. 'Llama-2-7b-chat-hf'
    pattern = os.path.join(
        output_dir, "*", model_org, model_name + "_*", "step_*", "all_results_la_*.json"
    )
    matches = glob.glob(pattern)
    step_dirs = {os.path.dirname(m) for m in matches}
    return sorted(step_dirs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf",
                   help="Model path as used in the run dirs, e.g. meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--output_dir",         default="./outputs")
    p.add_argument("--laplace_optim_step", type=int, default=100)
    p.add_argument("--n_samples",          type=int, nargs="+", default=[1, 2, 4, 8])
    args = p.parse_args()

    model_name = os.path.basename(args.model)  # e.g. Llama-2-7b-chat-hf

    step_dirs = discover_step_dirs(args.output_dir, args.model)
    if not step_dirs:
        print(f"No Laplace result files found for model '{args.model}' under '{args.output_dir}'.")
        sys.exit(1)

    print(f"Found {len(step_dirs)} run(s) for model: {args.model}\n")

    ok, skipped = 0, 0
    for step_dir in step_dirs:
        # Parse path components
        rel = os.path.relpath(step_dir, args.output_dir)
        parts = rel.split(os.sep)
        # parts: [task, model_org, run_dir_name, step_dir_name]
        task          = parts[0]
        run_dir_name  = parts[2]
        load_step     = int(parts[3].removeprefix("step_"))

        parsed = parse_run_dir_name(run_dir_name, model_name)
        if parsed is None:
            print(f"[skip] could not parse run dir: {run_dir_name}")
            skipped += 1
            continue

        alpha, dropout, lr, seed_label, lm_head, testing_set = parsed

        run_args = argparse.Namespace(
            task_name=task,
            model_name_or_path=args.model,
            output_dir=args.output_dir,
            lora_alpha=alpha,
            lora_dropout=float(dropout),
            learning_rate=float(lr),
            seed_label=seed_label,
            seed=None,
            load_step=load_step,
            lm_head=lm_head,
            testing_set=testing_set,
            laplace_hessian="kron",
            laplace_prior="homo",
            laplace_predict="mc_corr",
            laplace_optim_step=args.laplace_optim_step,
            n_samples=args.n_samples,
        )
        status = consolidate_run(run_args)
        print(f"  [{task}] {seed_label}  step={load_step}  {status}")
        ok += 1

    print(f"\nDone: {ok} consolidated, {skipped} skipped.")


if __name__ == "__main__":
    main()
