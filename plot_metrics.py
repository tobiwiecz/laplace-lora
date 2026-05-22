"""Per-(task, metric) metric plots across methods with SEM error bands.

Results structure:
    results/{task}/seed{N}/{method}.json

Output:
    plots/{task}/{metric}.png   (two subplots: val left, test right)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ══ CONFIGURATION — edit here ════════════════════════════════════════════════

# Methods to plot: key = JSON basename (without .json), value = display label.
# "mean" is special: val → mean.json (flat), test → mean_test.json (flat).
METHODS: dict[str, str] = {
    "mean":                     "MAP mean",
    "glm_baseline":             "GLM baseline",
    "glm":                      "GLM calib",
    "delta_mvp":                "VP δ-MVP",
    "delta_streamlined":        "VP δ-stream",
    "delta_streamlined_calib":  "VP δ-stream calib",
    "exact_mvp":                "VP exact-MVP",
    "exact_mvp_calib":          "VP exact-MVP calib",
    "exact_streamlined":        "VP exact-stream",
}

# For calibrated variants: (source_json_basename, val_key, test_key).
# Methods not listed here use the default (own filename, "val", "test").
METHOD_SOURCE: dict[str, tuple[str, str, str]] = {
    "delta_streamlined_calib": ("delta_streamlined", "val_calibrated", "test_calibrated"),
    "exact_mvp_calib":         ("exact_mvp",         "val_calibrated", "test_calibrated"),
}

# Metrics to plot.  "nll" also matches the key "loss" in JSONs that use it.
METRICS: list[str] = [
    "accuracy", "nll", "ece", "ace", "brier", "auc",
    "C@0.001", "C@0.005", "C@0.01", "C@0.02", "C@0.05", "C@0.1", "C@0.2", "C@0.3",
]

# None → auto-discover all task directories under RESULTS_DIR.
TASKS: list[str] | None = None

RESULTS_DIR = Path("results")
OUT_DIR     = Path("plots")

# Explicit colors (tab10 palette) so inserting new methods doesn't shift others.
_T = dict(zip(
    ["blue","orange","green","red","purple","brown","pink","gray","olive","cyan"],
    ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"],
))
METHOD_COLORS: dict[str, str] = {
    "mean":                    _T["blue"],
    "glm_baseline":            _T["orange"],
    "glm":                     _T["orange"],
    "delta_mvp":               _T["red"],
    "delta_streamlined":       _T["purple"],
    "delta_streamlined_calib": _T["purple"],
    "exact_mvp":               _T["brown"],
    "exact_mvp_calib":         _T["brown"],
    "exact_streamlined":       _T["pink"],
}

# Marker per method: "o" = circle, "D" = diamond (calibrated variants).
METHOD_MARKERS: dict[str, str] = {
    "mean":                    "o",
    "glm_baseline":            "o",
    "glm":                     "D",
    "delta_mvp":               "o",
    "delta_streamlined":       "o",
    "delta_streamlined_calib": "D",
    "exact_mvp":               "o",
    "exact_mvp_calib":         "D",
    "exact_streamlined":       "o",
}

# ── aesthetics ────────────────────────────────────────────────────────────────
FIG_HEIGHT  = 4.0   # inches
BAR_WIDTH   = 0.55
BAR_ALPHA   = 0.75
BAND_ALPHA  = 0.25  # fill_between shading around mean±SEM
DPI         = 150
# ════════════════════════════════════════════════════════════════════════════


SEED_PREFIX = "seed"


# ── data loading ─────────────────────────────────────────────────────────────

def _get_value(d: dict, split: str, metric: str) -> float | None:
    """Extract scalar from a flat or split-nested JSON dict."""
    if split in d and isinstance(d[split], dict):
        src = d[split]
    elif not any(k in d for k in ("val", "test")):
        src = d  # flat (mean.json / mean_test.json)
    else:
        return None

    v = src.get(metric)
    if v is None and metric == "nll":
        v = src.get("loss")
    if v is None and metric == "loss":
        v = src.get("nll")
    return float(v) if v is not None else None


def _load_seed(seed_dir: Path, method: str, split: str, metric: str) -> float | None:
    if method == "mean":
        fname = "mean.json" if split == "val" else "mean_test.json"
        path = seed_dir / fname
        if not path.exists():
            return None
        return _get_value(json.loads(path.read_text()), split, metric)

    if method in METHOD_SOURCE:
        src_base, val_key, test_key = METHOD_SOURCE[method]
        split_key = val_key if split == "val" else test_key
        path = seed_dir / f"{src_base}.json"
        if not path.exists():
            return None
        return _get_value(json.loads(path.read_text()), split_key, metric)

    if method == "glm_baseline":
        path = seed_dir / "glm.json"
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        src = d.get(f"{split}_baseline", {})
        v = src.get(metric)
        if v is None and metric == "nll":
            v = src.get("loss")
        if v is None and metric == "loss":
            v = src.get("nll")
        return float(v) if v is not None else None

    path = seed_dir / f"{method}.json"
    if not path.exists():
        return None
    return _get_value(json.loads(path.read_text()), split, metric)


def collect(task_dir: Path, method: str, split: str, metric: str) -> list[float]:
    seed_dirs = sorted(
        d for d in task_dir.iterdir()
        if d.is_dir() and d.name.startswith(SEED_PREFIX)
    )
    return [
        v for sd in seed_dirs
        if (v := _load_seed(sd, method, split, metric)) is not None
    ]


# ── statistics ────────────────────────────────────────────────────────────────

def mean_sem(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    m   = float(arr.mean())
    s   = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return m, s


# ── plotting ──────────────────────────────────────────────────────────────────

def _draw_split(ax: plt.Axes, task_dir: Path, split: str, metric: str) -> tuple[float, float] | None:
    """Forest-plot: methods on y-axis, metric value on x-axis.
    Returns (x_min, x_max) over all seed values, or None if no data."""
    rows, means, sems, all_vals = [], [], [], []

    for i, (method, _) in enumerate(METHODS.items()):
        vals = collect(task_dir, method, split, metric)
        if not vals:
            continue
        m, s = mean_sem(vals)
        rows.append(i)
        means.append(m)
        sems.append(s)
        all_vals.extend(vals)

    if not rows:
        return None

    method_keys = list(METHODS.keys())
    labels  = [list(METHODS.values())[r] for r in rows]
    y       = np.arange(len(rows))
    col     = [METHOD_COLORS[method_keys[r]] for r in rows]
    markers = [METHOD_MARKERS.get(method_keys[r], "o") for r in rows]

    for yi, m, s, c, mk in zip(y, means, sems, col, markers):
        ax.axhspan(yi - 0.4, yi + 0.4, xmin=0, xmax=1,
                   color="white", zorder=0)
        ax.fill_betweenx([yi - 0.4, yi + 0.4], m - s, m + s,
                         color=c, alpha=BAND_ALPHA, zorder=1)
        ax.hlines(yi, m - s, m + s, colors=c, linewidth=2.0, zorder=2)
        ax.plot(m, yi, mk, color=c, markersize=6, zorder=3)

    ax.set_title(split, fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(metric, fontsize=9)
    ax.xaxis.grid(True, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.invert_yaxis()
    return float(min(all_vals)), float(max(all_vals))


def plot_task_metric(task_dir: Path, metric: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(6, FIG_HEIGHT * 2), max(2.5, len(METHODS) * 0.45 + 1.0)),
        sharey=True,
    )
    fig.suptitle(f"{task_dir.name} — {metric}", fontsize=12, fontweight="bold")

    x_bounds = []
    for ax, split in zip(axes, ["val", "test"]):
        result = _draw_split(ax, task_dir, split, metric)
        if result is None:
            ax.set_title(split, fontsize=11)
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=10)
        else:
            x_bounds.append(result)

    # shared x-axis range with 20 % padding
    if x_bounds:
        lo = min(b[0] for b in x_bounds)
        hi = max(b[1] for b in x_bounds)
        pad = max((hi - lo) * 0.20, 1e-4)
        for ax in axes:
            ax.set_xlim(lo - pad, hi + pad)

    # with sharey the right panel shares the tick objects; hide its labels manually
    plt.setp(axes[1].get_yticklabels(), visible=False)
    fig.tight_layout()

    out_path = out_dir / task_dir.name / f"{metric}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not RESULTS_DIR.exists():
        raise SystemExit(f"results/ not found: {RESULTS_DIR.resolve()}")

    tasks = TASKS or sorted(
        d.name for d in RESULTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(SEED_PREFIX)
    )

    print(f"Tasks:   {tasks}")
    print(f"Methods: {list(METHODS.keys())}")
    print(f"Metrics: {METRICS}\n")

    for task in tasks:
        task_dir = RESULTS_DIR / task
        if not task_dir.exists():
            print(f"  WARN: {task_dir} not found — skipping")
            continue
        print(f"[{task}]")
        for metric in METRICS:
            plot_task_metric(task_dir, metric, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
