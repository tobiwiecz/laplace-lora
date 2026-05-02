"""
Generate accuracy-vs-steps plots for all (model, task) combinations found in outputs/.
Reads eval_res.json at each step_N checkpoint and computes accuracy as pred==true.
Saves plots to plots/{model}/{task}/accuracy_vs_steps.png.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUTS_DIR = Path("outputs")
PLOTS_DIR = Path("plots")


def compute_accuracy(eval_res_path):
    correct = total = 0
    with open(eval_res_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d['pred'] == d['true']:
                correct += 1
            total += 1
    return correct / total if total > 0 else None


def parse_run_dir(run_dir: Path, task_dir: Path):
    """Return (model, seed) parsed from a run directory, or (None, None) on failure.

    Run dir name format: {model_name}_{peft_method}_{lora_alpha}_{dropout}_{lr}_{seed}
    The org prefix lives in the directory structure between task_dir and run_dir.parent.
    """
    run_name = run_dir.name
    # Strip the last 4 tokens: alpha, dropout, lr, seed
    parts = run_name.rsplit('_', 4)
    if len(parts) < 5:
        return None, None
    try:
        seed = int(parts[-1])
    except ValueError:
        return None, None

    prefix = parts[0]  # e.g. "Llama-2-7b-chat-hf_lora_lmhead"

    # Split off the peft method: everything from the last "_lora" onward
    lora_idx = prefix.rfind('_lora')
    model_name_part = prefix[:lora_idx] if lora_idx != -1 else prefix

    # Org is the relative path between task_dir and run_dir.parent (e.g. "meta-llama")
    try:
        org = str(run_dir.parent.relative_to(task_dir))
    except ValueError:
        return None, None

    model = f"{org}/{model_name_part}" if org != '.' else model_name_part
    return model, seed


def collect_data():
    """Return {model: {task: {seed: {step: accuracy}}}}"""
    data = {}
    for eval_res_path in sorted(OUTPUTS_DIR.rglob("eval_res.json")):
        step_dir = eval_res_path.parent
        m = re.match(r"step_(\d+)$", step_dir.name)
        if not m:
            continue
        step = int(m.group(1))

        run_dir = step_dir.parent
        try:
            task = run_dir.relative_to(OUTPUTS_DIR).parts[0]
        except (ValueError, IndexError):
            continue
        task_dir = OUTPUTS_DIR / task

        model, seed = parse_run_dir(run_dir, task_dir)
        if model is None:
            continue

        try:
            acc = compute_accuracy(eval_res_path)
        except Exception:
            continue
        if acc is None:
            continue

        data.setdefault(model, {}).setdefault(task, {}).setdefault(seed, {})[step] = acc

    return data


def make_plots(data):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for model, tasks in sorted(data.items()):
        for task, seeds_data in sorted(tasks.items()):
            model_tag = model.replace('/', '__')
            plot_dir = PLOTS_DIR / model_tag / task
            plot_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(8, 5))

            for i, (seed, step_accs) in enumerate(sorted(seeds_data.items())):
                steps = sorted(step_accs.keys())
                accs = [step_accs[s] for s in steps]
                ax.plot(steps, accs, marker='o', label=f"seed {seed}",
                        color=colors[i % len(colors)])

            ax.set_xlabel("Step")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{model}\n{task}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            out_path = plot_dir / "accuracy_vs_steps.png"
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    data = collect_data()
    if not data:
        print("No eval_res.json files found under outputs/")
    else:
        make_plots(data)
        print(f"\nAll plots written to {PLOTS_DIR}/")
