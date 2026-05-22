#!/usr/bin/env python3
"""Live monitor: plots mean NLL ± SEM across seeds from log files.
Run with:  python plot_training.py [--task ARC-Easy] [--interval 60]
Refreshes every --interval seconds.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SEEDS = [1, 2, 3, 4, 5]
MODEL_LOG  = "meta-llama__Llama-2-7b-chat-hf"
PEFT_TAG   = "lora_lmhead_16_0.1_5e-05"
# outputs/<task>/meta-llama/Llama-2-7b-chat-hf_<peft>_seed<N>/step_4000/
BASELINE_ROOT = Path("outputs")
EPOCH_RE   = re.compile(r"epoch\s+(\d+)/\d+\s+nll=([\d.]+)")

BASELINES = [
    ("mean",      "all_results.json",                         "black",    "--"),
    ("hess-diag", "all_results_la_diag_all_homo_mc_corr_100.json", "darkorange", "-."),
    ("hess-kron", "all_results_la_kron_all_homo_mc_corr_100.json", "darkgreen",  ":"),
]


def parse_log(path: Path) -> dict[int, float]:
    result = {}
    try:
        for line in path.read_text().splitlines():
            m = EPOCH_RE.search(line)
            if m:
                result[int(m.group(1))] = float(m.group(2))
    except FileNotFoundError:
        pass
    return result


def load_all(task: str, variant: str) -> tuple[list[int], np.ndarray]:
    log_dir = Path(f"logs_vp/{MODEL_LOG}/{task}")
    data = {}
    for s in SEEDS:
        d = parse_log(log_dir / f"{task}_seed{s}_{variant}.log")
        if d:
            data[s] = d
    if not data:
        return [], np.empty((0, 0))
    common = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
    if not common:
        return [], np.empty((0, 0))
    arr = np.array([[data[s][e] for e in common] for s in data])
    return common, arr


def load_baselines(task: str) -> list[tuple[str, float, float]]:
    """Return list of (label, mean_nll, sem_nll) across seeds for each baseline."""
    results = []
    for label, filename, color, ls in BASELINES:
        nlls = []
        for s in SEEDS:
            p = (BASELINE_ROOT / task / "meta-llama"
                 / f"Llama-2-7b-chat-hf_{PEFT_TAG}_seed{s}"
                 / "step_4000" / filename)
            try:
                nlls.append(json.loads(p.read_text())["loss"])
            except (FileNotFoundError, KeyError):
                pass
        if nlls:
            a = np.array(nlls)
            results.append((label, a.mean(), a.std(ddof=1) / len(a) ** 0.5, color, ls))
    return results


def plot(task: str, variant: str, ax: plt.Axes):
    epochs, arr = load_all(task, variant)
    baselines   = load_baselines(task)
    ax.cla()

    if len(epochs) == 0:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=13)
        ax.set_title(f"{task}  |  {variant}")
        return

    mean = arr.mean(axis=0)
    ddof = 1 if arr.shape[0] > 1 else 0
    sem  = arr.std(axis=0, ddof=ddof) / np.sqrt(arr.shape[0])

    # Per-seed thin lines
    for i in range(arr.shape[0]):
        ax.plot(epochs, arr[i], linewidth=0.6, alpha=0.4, color="grey")

    ax.plot(epochs, mean, color="steelblue", linewidth=1.8,
            label=f"VP calib train NLL — in-sample (n={arr.shape[0]})")
    ax.fill_between(epochs, mean - sem, mean + sem, alpha=0.25, color="steelblue")

    # Baseline horizontal lines (out-of-sample: evaluated on val, not optimised on it)
    for label, bl_mean, bl_sem, color, ls in baselines:
        ax.axhline(bl_mean, color=color, linewidth=1.4, linestyle=ls,
                   label=f"{label} [val OOS] {bl_mean:.4f}±{bl_sem:.4f}")
        ax.axhspan(bl_mean - bl_sem, bl_mean + bl_sem, alpha=0.08, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.set_title(
        f"{task}  |  {variant}  — epoch {epochs[-1]}  nll={mean[-1]:.4f} ± {sem[-1]:.4f}",
        fontsize=10,
    )
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task",     default=["ARC-Challenge"], nargs="+",
                   help="One or more task names (each gets its own subplot)")
    p.add_argument("--variant",  default="streamlined_delta_const_posterior",
                   help="Suffix after seed<N>_ in the log filename")
    p.add_argument("--interval", type=int, default=60,
                   help="Refresh interval in seconds")
    args = p.parse_args()

    tasks = args.task
    n     = len(tasks)
    out   = Path("training_progress.png")

    while True:
        try:
            fig, axes = plt.subplots(1, n, figsize=(9 * n, 5), squeeze=False)
            fig.tight_layout(pad=2)
            for task, ax in zip(tasks, axes[0]):
                plot(task, args.variant, ax)
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[{time.strftime('%H:%M:%S')}] saved → {out.resolve()}  (refreshing in {args.interval}s)", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] error (will retry): {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
