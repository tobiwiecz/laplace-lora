"""
Compares RMSNorm variance propagation methods against MC sampling ground truth.

RMSNorm (used by Llama-2) normalises without mean-subtraction:
    y = x / sqrt(mean(x²) + eps) * weight

Two analytical VP methods are evaluated:
  - Streamlined: uses only mean statistics for the normaliser denominator
                 scale = 1/sqrt(E[x]² + eps)
  - MVP:         accounts for input variance in the normaliser
                 scale = 1/sqrt(E[x²] + E[var] + eps)
                       = 1/sqrt(mean(m²) + mean(v) + eps)

MC sampling draws x ~ N(mean, diag(var)) and passes through RMSNorm exactly,
giving empirical mean and variance as the ground truth.

Four sweeps:
  1. Input variance              (weight fixed at ones, deterministic)
  2. Hidden dimension D          (input_var=1.0 fixed)
  3. Weight scale                (non-unit but deterministic weight)
  4. Joint input variance × hidden dim grid

Outputs (plots/RMSNorm_Approximation_Testing/):
  - Sweep line plots (mean error and var error vs swept parameter)
  - Scatter plots (predicted vs MC mean/variance)
  - Console table per configuration
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ── match Llama-2 RMSNorm exactly ────────────────────────────────────────────
from transformers.models.llama.modeling_llama import LlamaRMSNorm

PLOT_DIR   = "plots/RMSNorm_Approximation_Testing"
N_MC       = 50_000
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(PLOT_DIR, exist_ok=True)

FS = 14
plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
    "legend.fontsize": FS - 2, "lines.linewidth": 2.5, "lines.markersize": 8,
})


# ---------------------------------------------------------------------------
# ParamPair (lightweight, no MVP dependency needed here)
# ---------------------------------------------------------------------------

class ParamPair:
    def __init__(self, mean: torch.Tensor, var: torch.Tensor):
        self.mean = mean
        self.var  = var


# ---------------------------------------------------------------------------
# VP implementations (mirrors calibrate_vp.py exactly)
# ---------------------------------------------------------------------------

def vp_rms_norm_streamlined(x: ParamPair, norm: LlamaRMSNorm) -> ParamPair:
    """Uses only mean statistics for the normaliser — ignores input variance."""
    m, v  = x.mean, x.var
    ms    = m.float().pow(2).mean(dim=-1, keepdim=True)
    scale = torch.rsqrt(ms + norm.variance_epsilon)
    w     = norm.weight.float()
    return ParamPair(
        (m.float() * scale) * w,
        scale * scale * v.float() * w ** 2,
    )


def vp_rms_norm_mvp(x: ParamPair, norm: LlamaRMSNorm) -> ParamPair:
    """Accounts for input variance in the normaliser denominator."""
    m, v    = x.mean, x.var
    ms      = m.float().pow(2).mean(dim=-1, keepdim=True)
    v_mean  = v.float().mean(dim=-1, keepdim=True)
    scale   = torch.rsqrt((ms + v_mean).clamp(min=0) + norm.variance_epsilon)
    w       = norm.weight.float()
    return ParamPair(
        (m.float() * scale) * w,
        scale * scale * v.float() * w ** 2,
    )


# ---------------------------------------------------------------------------
# MC ground truth
# ---------------------------------------------------------------------------

def mc_rms_norm(norm: LlamaRMSNorm, inp: ParamPair, n_samples: int = N_MC):
    """Draw x ~ N(mean, diag(var)), pass through RMSNorm, return empirical stats."""
    mean = inp.mean.float()
    std  = inp.var.float().clamp(min=0).sqrt()
    with torch.no_grad():
        # Draw all samples at once: [N, *shape]
        eps      = torch.randn(n_samples, *mean.shape, device=mean.device)
        x_samp   = mean.unsqueeze(0) + std.unsqueeze(0) * eps          # [N, *shape]
        # Vectorised RMSNorm over last dim — no weight applied yet (handle separately)
        rms      = x_samp.pow(2).mean(dim=-1, keepdim=True).add(norm.variance_epsilon).rsqrt()
        y_samp   = x_samp * rms * norm.weight.float()                  # [N, *shape]
        mc_mean  = y_samp.mean(0)
        mc_var   = y_samp.var(0, unbiased=True)
    return mc_mean, mc_var


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(pred: ParamPair, mc_mean: torch.Tensor, mc_var: torch.Tensor) -> dict:
    """Mean absolute error of predicted mean and variance vs MC ground truth."""
    err_mean = (pred.mean - mc_mean).abs().mean().item()
    err_var  = (pred.var  - mc_var ).abs().mean().item()
    rel_mean = err_mean / (mc_mean.abs().mean().item() + 1e-8)
    rel_var  = err_var  / (mc_var.abs().mean().item()  + 1e-8)
    return {"mae_mean": err_mean, "mae_var": err_var,
            "rel_mean": rel_mean, "rel_var": rel_var}


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run(hidden_dim: int, input_var: float, weight_scale: float = 1.0,
        batch: int = 4, seq: int = 16, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    norm = LlamaRMSNorm(hidden_dim).to(DEVICE)
    norm.weight.data.fill_(weight_scale)

    inp_mean = torch.randn(batch, seq, hidden_dim, device=DEVICE)
    inp_var  = torch.rand(batch, seq, hidden_dim, device=DEVICE) * input_var
    inp      = ParamPair(inp_mean, inp_var)

    mc_mean, mc_var = mc_rms_norm(norm, inp)

    with torch.no_grad():
        sl  = vp_rms_norm_streamlined(inp, norm)
        mvp = vp_rms_norm_mvp(inp, norm)

    return {
        "mc_mean": mc_mean, "mc_var": mc_var,
        "sl": sl, "mvp": mvp,
        "sl_metrics":  metrics(sl,  mc_mean, mc_var),
        "mvp_metrics": metrics(mvp, mc_mean, mc_var),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

METRIC_LABELS = [
    ("MAE mean",      "mae_mean"),
    ("MAE var",       "mae_var"),
    ("Rel. err mean", "rel_mean"),
    ("Rel. err var",  "rel_var"),
]

def print_results(label: str, res: dict):
    sl, mvp = res["sl_metrics"], res["mvp_metrics"]
    print(f"\n  {label}")
    print(f"    {'Metric':<20} {'Streamlined':>14} {'MVP':>14}  Winner")
    print(f"    {'─'*55}")
    for name, key in METRIC_LABELS:
        sv, mv = sl[key], mvp[key]
        winner = "MVP" if mv < sv else ("Streamlined" if sv < mv else "tie")
        print(f"    {name:<20} {sv:>14.6f} {mv:>14.6f}  {winner}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _sweep_plot(sweep_vals, sl_results, mvp_results, xlabel, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (metric_name, key) in zip(axes, [("MAE mean", "mae_mean"), ("MAE var", "mae_var")]):
        ax.plot(sweep_vals, [r["sl_metrics"][key]  for r in sl_results],
                "s-", label="Streamlined", color="#dc2626")
        ax.plot(sweep_vals, [r["mvp_metrics"][key] for r in mvp_results],
                "o-", label="MVP",         color="#2563eb")
        ax.set_xlabel(xlabel); ax.set_ylabel(metric_name)
        ax.set_title(metric_name); ax.legend(); ax.grid(True, alpha=0.3)
        if all(v > 0 for v in sweep_vals):
            ax.set_xscale("log")
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def _scatter_plot(res, label, path):
    mc_m = res["mc_mean"].flatten().cpu().numpy()
    mc_v = res["mc_var"].flatten().cpu().numpy()
    idx  = np.random.choice(len(mc_m), size=min(800, len(mc_m)), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for col, (name, pred, color) in enumerate([
        ("Streamlined", res["sl"],  "#dc2626"),
        ("MVP",         res["mvp"], "#2563eb"),
    ]):
        pm = pred.mean.flatten().cpu().numpy()[idx]
        pv = pred.var.flatten().cpu().numpy()[idx]

        ax = axes[0, col]
        lo = min(mc_m[idx].min(), pm.min()); hi = max(mc_m[idx].max(), pm.max())
        ax.scatter(mc_m[idx], pm, alpha=0.25, s=6, color=color)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("MC mean"); ax.set_ylabel(f"{name} mean")
        ax.set_title(f"{name} — mean"); ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        vmax = max(mc_v[idx].max(), pv.max()) * 1.05
        ax.scatter(mc_v[idx], pv, alpha=0.25, s=6, color=color)
        ax.plot([0, vmax], [0, vmax], "k--", lw=1)
        ax.set_xlim(0, vmax); ax.set_ylim(0, vmax); ax.set_aspect("equal")
        ax.set_xlabel("MC variance"); ax.set_ylabel(f"{name} variance")
        ax.set_title(f"{name} — variance"); ax.grid(True, alpha=0.3)

    fig.suptitle(f"RMSNorm VP scatter: {label}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}  |  N_MC={N_MC:,}")

    # ── Sweep 1: input variance ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("SWEEP 1: Input variance  (D=128, weight=1.0)")
    print("="*70)
    input_vars = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    s1_sl, s1_mvp = [], []
    for iv in input_vars:
        res = run(hidden_dim=128, input_var=iv)
        print_results(f"input_var={iv}", res)
        s1_sl.append(res); s1_mvp.append(res)
    _sweep_plot(input_vars, s1_sl, s1_mvp,
                "Input variance", "RMSNorm VP vs input variance (D=128)",
                os.path.join(PLOT_DIR, "sweep1_input_variance.png"))
    _scatter_plot(s1_sl[3], "input_var=1.0, D=128",
                  os.path.join(PLOT_DIR, "scatter_sweep1_iv1.0.png"))

    # ── Sweep 2: hidden dimension ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("SWEEP 2: Hidden dimension  (input_var=1.0, weight=1.0)")
    print("="*70)
    hidden_dims = [16, 64, 128, 256, 512, 1024, 4096]
    s2_sl, s2_mvp = [], []
    for D in hidden_dims:
        res = run(hidden_dim=D, input_var=1.0)
        print_results(f"D={D}", res)
        s2_sl.append(res); s2_mvp.append(res)
    _sweep_plot(hidden_dims, s2_sl, s2_mvp,
                "Hidden dim D", "RMSNorm VP vs hidden dimension (input_var=1.0)",
                os.path.join(PLOT_DIR, "sweep2_hidden_dim.png"))
    _scatter_plot(s2_sl[-1], "D=4096 (Llama-2 size), input_var=1.0",
                  os.path.join(PLOT_DIR, "scatter_sweep2_D4096.png"))

    # ── Sweep 3: weight scale ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SWEEP 3: Weight scale  (D=128, input_var=1.0)")
    print("="*70)
    weight_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    s3_sl, s3_mvp = [], []
    for ws in weight_scales:
        res = run(hidden_dim=128, input_var=1.0, weight_scale=ws)
        print_results(f"weight_scale={ws}", res)
        s3_sl.append(res); s3_mvp.append(res)
    _sweep_plot(weight_scales, s3_sl, s3_mvp,
                "Weight scale", "RMSNorm VP vs weight scale (D=128, input_var=1.0)",
                os.path.join(PLOT_DIR, "sweep3_weight_scale.png"))

    # ── Sweep 4: joint input variance × hidden dim grid ───────────────────────
    print("\n" + "="*70)
    print("SWEEP 4: Joint input variance × hidden dim grid")
    print("="*70)
    iv_grid = [0.1, 1.0, 5.0]
    D_grid  = [128, 1024, 4096]
    print(f"\n  {'':>12}", end="")
    for D in D_grid:
        print(f"  D={D:>4} (SL mae_var / MVP mae_var)", end="")
    print()
    for iv in iv_grid:
        print(f"  iv={iv:<8}", end="")
        for D in D_grid:
            res = run(hidden_dim=D, input_var=iv)
            sl_v  = res["sl_metrics"]["mae_var"]
            mvp_v = res["mvp_metrics"]["mae_var"]
            winner = "MVP" if mvp_v < sl_v else "SL "
            print(f"  {sl_v:.4f} / {mvp_v:.4f} ({winner})", end="")
        print()

    print(f"\nAll plots written to {PLOT_DIR}/")
