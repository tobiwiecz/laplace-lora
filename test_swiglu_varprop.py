"""
Toy script: variance propagation through SwiGLU = SiLU(gate) * up.

For each (input mean, input std) grid point, samples x ~ N(mu, sigma^2 * I)
are projected through fixed W_gate and W_up, then passed through SwiGLU.
The MC empirical mean/std are compared to two analytical approximations:

  delta     : first-order delta method, no cross-covariance (current impl)
  delta+cov : delta method + cross-covariance term Cov[gate_j, up_j]

Since both gate and up are linear functions of the same uncertain input x,
they are correlated: Cov[gate_j, up_j] = sigma^2 * (W_gate[j] · W_up[j]).
The delta method ignores this, delta+cov adds the correction:

  Var[y_j] += 2 * f'(mu_g_j) * mu_u_j * f(mu_g_j) * Cov[gate_j, up_j]

Outputs:
  - Console table of mean/std errors per method
  - Heatmaps of errors over the (mean, std) grid
  - Improvement map: where does cross-covariance help most
"""
import math
import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
N_SAMPLES   = 50_000
D_IN        = 16
D_OUT       = 32
STEPS_MEAN  = 30
STEPS_STD   = 30
SEED        = 0
PLOT_DIR    = "plots/swiglu_varprop"
os.makedirs(PLOT_DIR, exist_ok=True)

torch.manual_seed(SEED)

mean_values = torch.linspace(-50.0, 50.0, steps=STEPS_MEAN)
std_values  = torch.linspace(0.05, 25.0, steps=STEPS_STD)

W_gate = torch.randn(D_OUT, D_IN) / math.sqrt(D_IN)
W_up   = torch.randn(D_OUT, D_IN) / math.sqrt(D_IN)

# Weight statistics used to derive gate/up moments from x moments [D_OUT]
W_gate_mean_fac = W_gate.sum(dim=1)           # mu_g_j  = mu_x  * this
W_up_mean_fac   = W_up.sum(dim=1)
W_gate_var_fac  = (W_gate ** 2).sum(dim=1)    # var_g_j = var_x * this
W_up_var_fac    = (W_up ** 2).sum(dim=1)
W_cov_fac       = (W_gate * W_up).sum(dim=1)  # cov_gu_j = var_x * this


# ── SiLU derivatives ──────────────────────────────────────────────────────────
def silu_deriv(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def silu_deriv2(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 - s) * (2.0 + x * (1.0 - 2.0 * s))


# ── Analytical approximations ─────────────────────────────────────────────────
def swiglu_delta(mu_g, var_g, mu_u, var_u):
    """Delta method — no cross-covariance (current calibrate_vp.py implementation)."""
    f  = F.silu(mu_g)
    df = silu_deriv(mu_g)
    return f * mu_u, (df * mu_u) ** 2 * var_g + f ** 2 * var_u


def swiglu_delta_cov(mu_g, var_g, mu_u, var_u, cov_gu):
    """Delta method + cross-covariance: 2 * f'(mu_g) * mu_u * f(mu_g) * Cov[g, u]."""
    f  = F.silu(mu_g)
    df = silu_deriv(mu_g)
    return f * mu_u, (df * mu_u) ** 2 * var_g + f ** 2 * var_u + 2 * df * mu_u * f * cov_gu


def swiglu_2nd_mean(mu_g, var_g, mu_u, var_u, cov_gu):
    """Second-order mean correction + first-order var + cross-cov.

    E[y] ≈ f(μ_g)·μ_u  +  f'(μ_g)·cov_gu  +  f''(μ_g)/2·var_g·μ_u
    """
    f   = F.silu(mu_g)
    df  = silu_deriv(mu_g)
    d2f = silu_deriv2(mu_g)
    mean = f * mu_u + df * cov_gu + (d2f / 2.0) * var_g * mu_u
    var  = (df * mu_u) ** 2 * var_g + f ** 2 * var_u + 2 * df * mu_u * f * cov_gu
    return mean, var


def swiglu_exact_prod_var(mu_g, var_g, mu_u, var_u, cov_gu):
    """Exact product-variance formula (Var[ab] = Var[a]Var[b]+Var[a]μ_b²+μ_a²Var[b]+2Cov·μ_a·μ_b)
    with Var[f(g)] ≈ (f'(μ_g))²·var_g (first-order) and the cross-cov correction."""
    f      = F.silu(mu_g)
    df     = silu_deriv(mu_g)
    var_fg = df ** 2 * var_g                              # Var[SiLU(g)] first-order approx
    var    = (var_fg * var_u                              # Var[a]·Var[b]
              + var_fg * mu_u ** 2                        # Var[a]·μ_b²  (= delta term)
              + f ** 2 * var_u                            # μ_a²·Var[b]
              + 2 * df * cov_gu * f * mu_u)              # cross-cov correction
    return f * mu_u, var


def swiglu_full(mu_g, var_g, mu_u, var_u, cov_gu):
    """Second-order mean + exact product variance."""
    f      = F.silu(mu_g)
    df     = silu_deriv(mu_g)
    d2f    = silu_deriv2(mu_g)
    mean   = f * mu_u + df * cov_gu + (d2f / 2.0) * var_g * mu_u
    var_fg = df ** 2 * var_g
    var    = (var_fg * var_u
              + var_fg * mu_u ** 2
              + f ** 2 * var_u
              + 2 * df * cov_gu * f * mu_u)
    return mean, var


# ── SiLU ≈ GELU Hermite expansion ─────────────────────────────────────────────
# σ(x) ≈ Φ(c·x)  where c = √(π/8), so SiLU(x) = x·σ(x) ≈ GELU(c·x)/c.
# For gate ~ N(μ_g, var_g): c·gate ~ N(c·μ_g, c²·var_g).
# The GELU Hermite moments of c·gate, divided by c, give SiLU moments.

_C          = math.sqrt(math.pi / 8.0)      # σ(x) ≈ Φ(_C·x)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)

def _npdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) * _INV_SQRT2PI


def _gelu_moments(m: torch.Tensor, v: torch.Tensor, k: int):
    """(E[GELU(y)], Var[GELU(y)]) for y ~ N(m, v), Hermite order k."""
    zeta  = torch.rsqrt(1.0 + v)
    m_out = m * torch.special.ndtr(m * zeta) + v * zeta * _npdf(m * zeta)
    if k <= 0:
        return m_out, torch.zeros_like(m_out)
    sigma = v.clamp(min=0.0).sqrt()
    gamma = m * zeta                   # μ / √(1+σ²)
    alpha = sigma * zeta               # σ / √(1+σ²)
    phig  = _npdf(gamma)
    v_out = (sigma * torch.special.ndtr(gamma) + alpha * (1.0 - alpha**2) * m * phig) ** 2
    if k == 1:
        return m_out, v_out
    gamma = gamma.clamp(-20.0, 20.0)
    herm  = [torch.special.hermite_polynomial_he(gamma, i) for i in range(k + 1)]
    pre   = (sigma * phig) ** 2
    for i in range(2, k + 1):
        v_out += pre * (alpha**(i-1) * (herm[i-2] - (1.0 - alpha**2) * herm[i])) ** 2 / math.factorial(i)
    return m_out, v_out


def swiglu_hermite(mu_g, var_g, mu_u, var_u, cov_gu, k: int):
    """SwiGLU VP: SiLU moments via GELU Hermite (order k) + exact product var.

    SiLU(gate) ≈ GELU(_C·gate) / _C, so:
      E[SiLU(gate)]   ≈ E_GELU(_C·μ_g, _C²·var_g) / _C
      Var[SiLU(gate)] ≈ Var_GELU(_C·μ_g, _C²·var_g) / _C²
    Mean correction uses first-order Stein: Cov[SiLU(gate), up] ≈ f'(μ_g)·cov_gu.
    """
    gm, gv = _gelu_moments(_C * mu_g, _C**2 * var_g, k)
    silu_m = gm / _C        # E[SiLU(gate)]
    silu_v = gv / _C**2     # Var[SiLU(gate)], higher-order approx
    df     = silu_deriv(mu_g)
    mean   = silu_m * mu_u + df * cov_gu
    var    = (silu_v * var_u
              + silu_v * mu_u**2
              + silu_m**2 * var_u
              + 2.0 * df * cov_gu * silu_m * mu_u)
    return mean, var


HERMITE_ORDERS = [1, 2, 3, 4, 5]


# ── Scan ──────────────────────────────────────────────────────────────────────
# Error matrices: [STEPS_MEAN, STEPS_STD]
err_delta_mean    = torch.zeros(STEPS_MEAN, STEPS_STD)
err_delta_std     = torch.zeros(STEPS_MEAN, STEPS_STD)
err_cov_mean      = torch.zeros(STEPS_MEAN, STEPS_STD)
err_cov_std       = torch.zeros(STEPS_MEAN, STEPS_STD)
err_2nd_mean_mean = torch.zeros(STEPS_MEAN, STEPS_STD)
err_2nd_mean_std  = torch.zeros(STEPS_MEAN, STEPS_STD)
err_epv_mean      = torch.zeros(STEPS_MEAN, STEPS_STD)
err_epv_std       = torch.zeros(STEPS_MEAN, STEPS_STD)
err_full_mean     = torch.zeros(STEPS_MEAN, STEPS_STD)
err_full_std      = torch.zeros(STEPS_MEAN, STEPS_STD)
err_herm_mean     = {k: torch.zeros(STEPS_MEAN, STEPS_STD) for k in HERMITE_ORDERS}
err_herm_std      = {k: torch.zeros(STEPS_MEAN, STEPS_STD) for k in HERMITE_ORDERS}

# Vectorise over mean dimension, loop over std to keep peak memory manageable.
# x_batch: [STEPS_MEAN, N_SAMPLES, D_IN] — reused each std step.
mu_col = mean_values.view(STEPS_MEAN, 1, 1)   # for broadcasting

for j, sig in enumerate(std_values):
    var_x = sig.item() ** 2

    # ── MC ground truth ──────────────────────────────────────────────────────
    x = torch.randn(STEPS_MEAN, N_SAMPLES, D_IN) * sig + mu_col  # [M, N, D_in]
    gate_s = x @ W_gate.T       # [M, N, D_out]
    up_s   = x @ W_up.T
    y_s    = F.silu(gate_s) * up_s
    mc_mean = y_s.mean(1)       # [M, D_out]
    mc_std  = y_s.std(1)

    # ── Analytical moments from weight statistics ─────────────────────────────
    mu_x = mean_values.view(STEPS_MEAN, 1)       # [M, 1]
    mu_g  = mu_x  * W_gate_mean_fac              # [M, D_out]
    mu_u  = mu_x  * W_up_mean_fac
    var_g = var_x * W_gate_var_fac               # [M, D_out]  (broadcast scalar)
    var_u = var_x * W_up_var_fac
    cov   = var_x * W_cov_fac

    # delta
    d_mean, d_var = swiglu_delta(mu_g, var_g, mu_u, var_u)
    d_std = d_var.clamp(min=0).sqrt()
    err_delta_mean[:, j] = (d_mean - mc_mean).abs().mean(dim=1)
    err_delta_std [:, j] = (d_std  - mc_std ).abs().mean(dim=1)

    # delta + cov
    c_mean, c_var = swiglu_delta_cov(mu_g, var_g, mu_u, var_u, cov)
    c_std = c_var.clamp(min=0).sqrt()
    err_cov_mean[:, j] = (c_mean - mc_mean).abs().mean(dim=1)
    err_cov_std [:, j] = (c_std  - mc_std ).abs().mean(dim=1)

    # second-order mean
    m2_mean, m2_var = swiglu_2nd_mean(mu_g, var_g, mu_u, var_u, cov)
    m2_std = m2_var.clamp(min=0).sqrt()
    err_2nd_mean_mean[:, j] = (m2_mean - mc_mean).abs().mean(dim=1)
    err_2nd_mean_std [:, j] = (m2_std  - mc_std ).abs().mean(dim=1)

    # exact product variance
    ep_mean, ep_var = swiglu_exact_prod_var(mu_g, var_g, mu_u, var_u, cov)
    ep_std = ep_var.clamp(min=0).sqrt()
    err_epv_mean[:, j] = (ep_mean - mc_mean).abs().mean(dim=1)
    err_epv_std [:, j] = (ep_std  - mc_std ).abs().mean(dim=1)

    # full (2nd-order mean + exact product var)
    fl_mean, fl_var = swiglu_full(mu_g, var_g, mu_u, var_u, cov)
    fl_std = fl_var.clamp(min=0).sqrt()
    err_full_mean[:, j] = (fl_mean - mc_mean).abs().mean(dim=1)
    err_full_std [:, j] = (fl_std  - mc_std ).abs().mean(dim=1)

    # hermite approximation (multiple orders)
    for k in HERMITE_ORDERS:
        hm, hv = swiglu_hermite(mu_g, var_g, mu_u, var_u, cov, k)
        hs = hv.clamp(min=0).sqrt()
        err_herm_mean[k][:, j] = (hm - mc_mean).abs().mean(dim=1)
        err_herm_std [k][:, j] = (hs - mc_std ).abs().mean(dim=1)

    best_k = min(HERMITE_ORDERS, key=lambda k: err_herm_std[k][:, j].mean())
    print(f"  std={sig:.2f}  delta_std_err={err_delta_std[:,j].mean():.4f}"
          f"  full_std_err={err_full_std[:,j].mean():.4f}"
          f"  hermite_best(k={best_k})={err_herm_std[best_k][:,j].mean():.4f}")


# ── Console table ─────────────────────────────────────────────────────────────
rows = [
    ("Delta (no cov)",            err_delta_mean.mean().item(),    err_delta_std.mean().item()),
    ("Delta + cov",               err_cov_mean.mean().item(),      err_cov_std.mean().item()),
    ("Delta + cov + 2nd mean",    err_2nd_mean_mean.mean().item(), err_2nd_mean_std.mean().item()),
    ("Exact prod var + cov",      err_epv_mean.mean().item(),      err_epv_std.mean().item()),
    ("Full (2nd mean + prod var)", err_full_mean.mean().item(),    err_full_std.mean().item()),
    None,
] + [(f"Hermite k={k} (SiLU≈GELU)", err_herm_mean[k].mean().item(), err_herm_std[k].mean().item())
     for k in HERMITE_ORDERS]
col_w = max(len(r[0]) for r in rows if r is not None)
sep   = f"  {'─' * col_w}  {'─' * 12}  {'─' * 12}"
print(f"\n  {'Method':<{col_w}}  {'Mean Error':>12}  {'Std Error':>12}")
print(sep)
for row in rows:
    if row is None:
        print(sep)
    else:
        name, me, se = row
        print(f"  {name:<{col_w}}  {me:>12.6f}  {se:>12.6f}")
print()


# ── Heatmaps ──────────────────────────────────────────────────────────────────
def save_heatmap(data: np.ndarray, title: str, path: str, symmetric: bool = False):
    fig, ax = plt.subplots(figsize=(7, 5))
    ext = [std_values.min().item(), std_values.max().item(),
           mean_values.min().item(), mean_values.max().item()]
    vmax = float(np.abs(data).max())
    vmin = -vmax if symmetric else 0.0
    im = ax.imshow(data, aspect="auto", origin="lower", extent=ext,
                   cmap="RdBu_r" if symmetric else "viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Input std σ", fontsize=12)
    ax.set_ylabel("Input mean μ", fontsize=12)
    ax.set_title(title, fontsize=13)
    cb = plt.colorbar(im, ax=ax)
    ax.text(0.99, 0.02, f"max |err|={vmax:.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            color="white", bbox=dict(facecolor="black", alpha=0.4, pad=2))
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


mn = mean_values.numpy()
sn = std_values.numpy()

save_heatmap(err_delta_mean.numpy(), "Delta (no cov) — mean error", f"{PLOT_DIR}/delta_mean_err.png")
save_heatmap(err_delta_std .numpy(), "Delta (no cov) — std error",  f"{PLOT_DIR}/delta_std_err.png")
save_heatmap(err_cov_mean  .numpy(), "Delta + cov — mean error",    f"{PLOT_DIR}/cov_mean_err.png")
save_heatmap(err_cov_std   .numpy(), "Delta + cov — std error",     f"{PLOT_DIR}/cov_std_err.png")

save_heatmap(err_2nd_mean_mean.numpy(), "2nd-order mean + cov — mean error",    f"{PLOT_DIR}/2nd_mean_mean_err.png")
save_heatmap(err_2nd_mean_std .numpy(), "2nd-order mean + cov — std error",     f"{PLOT_DIR}/2nd_mean_std_err.png")
save_heatmap(err_epv_mean     .numpy(), "Exact prod var + cov — mean error",    f"{PLOT_DIR}/epv_mean_err.png")
save_heatmap(err_epv_std      .numpy(), "Exact prod var + cov — std error",     f"{PLOT_DIR}/epv_std_err.png")
save_heatmap(err_full_mean    .numpy(), "Full (2nd mean + prod var) — mean error", f"{PLOT_DIR}/full_mean_err.png")
save_heatmap(err_full_std     .numpy(), "Full (2nd mean + prod var) — std error",  f"{PLOT_DIR}/full_std_err.png")

improvement_cov  = err_delta_std   - err_cov_std
improvement_full = err_delta_std   - err_full_std
improvement_mean = err_delta_mean  - err_full_mean
save_heatmap(improvement_cov .numpy(),
             "Std-error improvement: delta → delta+cov\n(positive = cross-cov helps)",
             f"{PLOT_DIR}/improvement_cov_std.png", symmetric=True)
save_heatmap(improvement_full.numpy(),
             "Std-error improvement: delta → full\n(positive = full helps)",
             f"{PLOT_DIR}/improvement_full_std.png", symmetric=True)
save_heatmap(improvement_mean.numpy(),
             "Mean-error improvement: delta → full\n(positive = full helps)",
             f"{PLOT_DIR}/improvement_full_mean.png", symmetric=True)

# Hermite heatmaps (std error only — mean error unchanged vs full)
for k in HERMITE_ORDERS:
    save_heatmap(err_herm_std[k].numpy(),
                 f"Hermite k={k} (SiLU≈GELU) — std error",
                 f"{PLOT_DIR}/hermite_k{k}_std_err.png")
    improvement_hk = err_delta_std - err_herm_std[k]
    save_heatmap(improvement_hk.numpy(),
                 f"Std-error improvement: delta → Hermite k={k}\n(positive = Hermite helps)",
                 f"{PLOT_DIR}/improvement_hermite_k{k}_std.png", symmetric=True)
