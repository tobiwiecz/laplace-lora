"""
Generate per-metric plots for all (model, task) combinations found in outputs/.
Reads all_results.json at each step_N checkpoint.
Saves one plot per metric to plots/{model}/{task}/{metric}_vs_steps.png.
"""
import json
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for multiprocessing
import matplotlib.pyplot as plt

OUTPUTS_DIR = Path("outputs")
PLOTS_DIR = Path("plots")

SKIP_KEYS = {"nll", "auc_rc", "n_seeds"}

RANDOM_BASELINE = {
    "winogrande_s":  0.50,
    "winogrande_m":  0.50,
    "boolq":         0.50,
    "ARC-Challenge": 0.25,
    "ARC-Easy":      0.25,
    "openbookqa":    0.25,
}


def parse_run_dir(run_dir: Path, task_dir: Path):
    """Return (model, seed_label) from a run directory, or (None, None) on failure."""
    run_name = run_dir.name
    parts = run_name.rsplit('_', 4)
    if len(parts) < 5:
        return None, None

    seed_str = parts[-1]
    seed_label = seed_str if seed_str.startswith("seed") else f"seed={seed_str}"

    prefix = parts[0]
    lora_idx = prefix.rfind('_lora')
    model_name_part = prefix[:lora_idx] if lora_idx != -1 else prefix

    try:
        org = str(run_dir.parent.relative_to(task_dir))
    except ValueError:
        return None, None

    model = f"{org}/{model_name_part}" if org != '.' else model_name_part
    return model, seed_label


def collect_data():
    """Return {model: {task: {seed_label: {step: {metric: value}}}}}"""
    data = {}
    for path in sorted(OUTPUTS_DIR.rglob("all_results.json")):
        step_dir = path.parent
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

        model, seed_label = parse_run_dir(run_dir, task_dir)
        if model is None:
            continue

        try:
            with open(path) as f:
                results = json.load(f)
        except Exception:
            continue

        metrics = {k: v for k, v in results.items()
                   if isinstance(v, (int, float)) and k not in SKIP_KEYS}
        if not metrics:
            continue

        data.setdefault(model, {}).setdefault(task, {}).setdefault(seed_label, {})[step] = metrics

    return data


def _plot_one(args):
    model, task, metric, seeds_data, plot_dir = args
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (seed_label, step_metrics) in enumerate(sorted(seeds_data.items())):
        steps = sorted(s for s in step_metrics if metric in step_metrics[s])
        if not steps:
            continue
        values = [step_metrics[s][metric] for s in steps]
        ax.plot(steps, values, marker='o', label=seed_label,
                color=colors[i % len(colors)])

    if metric == "accuracy" and task in RANDOM_BASELINE:
        ax.axhline(RANDOM_BASELINE[task], color='black', linestyle='--',
                   linewidth=1.2, label='random')

    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(f"{model}\n{task} — {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = Path(plot_dir) / f"{metric}_vs_steps.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(out_path)


def make_plots(data):
    jobs = []
    for model, tasks in sorted(data.items()):
        for task, seeds_data in sorted(tasks.items()):
            all_metrics = set()
            for step_metrics in seeds_data.values():
                for metrics in step_metrics.values():
                    all_metrics.update(metrics.keys())

            model_tag = model.replace('/', '__')
            plot_dir = PLOTS_DIR / model_tag / task
            plot_dir.mkdir(parents=True, exist_ok=True)

            for metric in sorted(all_metrics):
                jobs.append((model, task, metric, seeds_data, str(plot_dir)))

    n_workers = min(cpu_count(), len(jobs))
    print(f"Generating {len(jobs)} plots using {n_workers} workers...")

    counts = {}
    with Pool(processes=n_workers) as pool:
        for out_path in pool.imap_unordered(_plot_one, jobs, chunksize=4):
            # out_path is e.g. plots/model/task/metric_vs_steps.png
            key = "/".join(Path(out_path).parts[-3:-1])  # model_tag/task
            counts[key] = counts.get(key, 0) + 1

    for key, n in sorted(counts.items()):
        print(f"  {key}: {n} plots saved")


if __name__ == "__main__":
    data = collect_data()
    if not data:
        print("No all_results.json files found under outputs/")
    else:
        make_plots(data)
        print(f"\nAll plots written to {PLOTS_DIR}/")
