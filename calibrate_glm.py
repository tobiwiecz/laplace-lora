"""
GLM variance-scale calibration.

Loads precomputed f_mu [N, C] and f_var [N, C, C] from disk, then learns a
single scalar log_s_pred such that the scaled predictive covariance

    exp(log_s_pred) * JΣJᵀ

minimises val NLL.  Mean logits f_mu are never modified.

Outputs per run:
  outputs_laplace/.../step_<k>/calib_params_glm[_<suffix>].pt
  outputs/<task>/.../step_<k>/all_results_glm[_<suffix>].json
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--task_name",          type=str, required=True)
    p.add_argument("--seed",               type=int, required=True)
    p.add_argument("--seed_label",         type=str, default=None)
    p.add_argument("--load_step",          type=int, default=4000)
    p.add_argument("--laplace_hessian",    type=str, default="kron")
    p.add_argument("--laplace_sub",        type=str, default="all")
    p.add_argument("--laplace_prior",      type=str, default="homo")
    p.add_argument("--laplace_optim_step", type=int, default=100)
    p.add_argument("--lm_head",            action="store_true", default=True)
    p.add_argument("--lora_alpha",         type=int,   default=16)
    p.add_argument("--lora_dropout",       type=float, default=0.1)
    p.add_argument("--total_steps",        type=int,   default=200)
    p.add_argument("--lr",                 type=float, default=1e-1)
    p.add_argument("--n_mc_calib",         type=int,   default=1000)
    p.add_argument("--n_mc_eval",          type=int,   default=100000)
    p.add_argument("--also_eval_test",     action="store_true", default=False,
                   help="Load precomputed f_mu_test/f_var_test and evaluate on the HF test split")
    p.add_argument("--suffix",             type=str,   default="")
    p.add_argument("--results_dir",        type=str,   default="./results",
                   help="Root of the shared results folder; saves to results_dir/task/seed_label/")
    p.add_argument("--output_dir",         type=str,   default=None,
                   help="Override for outputs dir (auto-derived if unset)")
    p.add_argument("--laplace_output_dir", type=str,   default=None,
                   help="Override for outputs_laplace dir (auto-derived if unset)")

    args = p.parse_args()

    lr = 5e-5
    seed_label = args.seed_label or str(args.seed)
    peft_method = "lora_lmhead" if args.lm_head else "lora"

    tag = (f"{args.model_name_or_path}_{peft_method}_{args.lora_alpha}"
           f"_{args.lora_dropout}_{lr}_{seed_label}")

    if args.output_dir is None:
        args.output_dir = f"outputs/{args.task_name}/{tag}/step_{args.load_step}"
    if args.laplace_output_dir is None:
        args.laplace_output_dir = (
            f"outputs_laplace/{args.task_name}/{tag}/step_{args.load_step}"
        )

    args._la_tag = (f"{args.laplace_hessian}_{args.laplace_sub}"
                    f"_{args.laplace_prior}_{args.laplace_optim_step}")
    args._suffix_str = f"_{args.suffix}" if args.suffix else ""
    return args


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _cholesky(f_var: torch.Tensor) -> torch.Tensor:
    """Cholesky of f_var + 1e-6 I, shape [N, C, C]."""
    C = f_var.shape[-1]
    eye = 1e-6 * torch.eye(C, device=f_var.device, dtype=f_var.dtype)
    return torch.linalg.cholesky(f_var + eye)


def mc_nll(f_mu: torch.Tensor, L: torch.Tensor, labels: torch.Tensor,
           log_s: torch.Tensor, n_mc: int) -> torch.Tensor:
    """
    NLL = -log E_z[p(y|z)],  z ~ N(f_mu, exp(log_s) * LLᵀ).

    L is the precomputed Cholesky of f_var (no grad needed).
    Gradient flows through log_s only.
    """
    N, C = f_mu.shape
    eps = torch.randn(n_mc, N, C, device=f_mu.device, dtype=f_mu.dtype)
    # z = f_mu + exp(log_s/2) * L @ eps
    noise = (L.unsqueeze(0) @ eps.unsqueeze(-1)).squeeze(-1)   # [n_mc, N, C]
    z = f_mu.unsqueeze(0) + (log_s / 2).exp() * noise          # [n_mc, N, C]
    log_p = F.log_softmax(z, dim=-1)                            # [n_mc, N, C]
    log_p_y = log_p[:, torch.arange(N, device=f_mu.device), labels]  # [n_mc, N]
    # log E[p(y|z)] via log-mean-exp
    log_prob = torch.logsumexp(log_p_y, dim=0) - math.log(n_mc)  # [N]
    return -log_prob.mean()


@torch.no_grad()
def mc_probs(f_mu: torch.Tensor, f_var: torch.Tensor, log_s: float,
             n_mc: int, batch_size: int = 128) -> torch.Tensor:
    """E[softmax(z)], z ~ N(f_mu, exp(log_s)*f_var). Processed in batches."""
    N, C = f_mu.shape
    s_sqrt = math.exp(log_s / 2)
    all_probs = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        mu_b  = f_mu[start:end]
        var_b = f_var[start:end]
        B = mu_b.shape[0]
        L_b = _cholesky(var_b)
        eps = torch.randn(n_mc, B, C, device=f_mu.device, dtype=f_mu.dtype)
        z = mu_b.unsqueeze(0) + s_sqrt * (L_b.unsqueeze(0) @ eps.unsqueeze(-1)).squeeze(-1)
        all_probs.append(F.softmax(z, dim=-1).mean(0).cpu())
    return torch.cat(all_probs, dim=0)   # [N, C]


def _fmt(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load precomputed tensors ──────────────────────────────────────────────
    f_mu_path  = os.path.join(args.laplace_output_dir, f"f_mu_{args._la_tag}.pt")
    f_var_path = os.path.join(args.laplace_output_dir, f"f_var_{args._la_tag}.pt")
    # Naming: eval_res_la_{hessian}_{sub}_{prior}_mc_corr_{optim_step}.json
    _la_tag_no_step = (f"{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}")
    eval_res_path = os.path.join(
        args.output_dir,
        f"eval_res_la_{_la_tag_no_step}_mc_corr_{args.laplace_optim_step}.json"
    )

    print(f"Loading f_mu  from {f_mu_path}")
    print(f"Loading f_var from {f_var_path}")
    print(f"Loading labels from {eval_res_path}")

    f_mu  = torch.load(f_mu_path,  map_location=device, weights_only=False).float()
    f_var = torch.load(f_var_path, map_location=device, weights_only=False).float()

    with open(eval_res_path) as fh:
        labels = torch.tensor(
            [json.loads(line)["true"] for line in fh if line.strip()],
            dtype=torch.long, device=device
        )

    N, C = f_mu.shape
    assert f_var.shape == (N, C, C), f"f_var shape mismatch: {f_var.shape}"
    assert labels.shape == (N,), f"labels shape mismatch: {labels.shape}"
    print(f"  Val: N={N}, C={C}, device={device}")

    # ── Optionally load test split tensors ───────────────────────────────────
    f_mu_test = f_var_test = labels_test = None
    if args.also_eval_test:
        f_mu_test_path  = os.path.join(args.laplace_output_dir, f"f_mu_test_{args._la_tag}.pt")
        f_var_test_path = os.path.join(args.laplace_output_dir, f"f_var_test_{args._la_tag}.pt")
        _la_tag_no_step = f"{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}"
        test_res_path = os.path.join(
            args.output_dir,
            f"eval_res_la_{_la_tag_no_step}_mc_corr_{args.laplace_optim_step}_test.json"
        )
        if os.path.exists(f_mu_test_path):
            f_mu_test  = torch.load(f_mu_test_path,  map_location=device, weights_only=False).float()
            f_var_test = torch.load(f_var_test_path, map_location=device, weights_only=False).float()
            with open(test_res_path) as fh:
                labels_test = torch.tensor(
                    [json.loads(line)["true"] for line in fh if line.strip()],
                    dtype=torch.long, device=device
                )
            print(f"  Test: N={f_mu_test.shape[0]}")
        else:
            print(f"  WARNING: test tensors not found at {f_mu_test_path} — skipping test eval")

    # Calibrate on full val set; test split is separate if loaded
    f_mu_c, f_var_c, lab_c = f_mu, f_var, labels

    # Precompute Cholesky on calib split (fixed; only log_s has grad)
    L_c = _cholesky(f_var_c)

    # ── Baseline (log_s = 0, uncalibrated) ───────────────────────────────────
    print(f"\n--- Baseline (uncalibrated, log_s=0.0, s=1.0) ---")
    base_val_probs   = mc_probs(f_mu, f_var, 0.0, args.n_mc_eval)
    base_val_metrics = compute_all_metrics(base_val_probs, labels.cpu())
    print(f"  Val:  {base_val_metrics}")
    base_test_metrics = None
    if f_mu_test is not None:
        base_test_probs   = mc_probs(f_mu_test, f_var_test, 0.0, args.n_mc_eval)
        base_test_metrics = compute_all_metrics(base_test_probs, labels_test.cpu())
        print(f"  Test: {base_test_metrics}")

    log_s = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([log_s], lr=args.lr)

    best_nll = float("inf")
    t0 = time.perf_counter()

    print(f"\n--- GLM variance-scale calibration ({args.total_steps} steps) ---")
    for step in range(args.total_steps):
        optimizer.zero_grad()
        nll = mc_nll(f_mu_c, L_c, lab_c, log_s, args.n_mc_calib)
        nll.backward()
        optimizer.step()

        nll_val = nll.item()
        ls_val  = log_s.item()
        if nll_val < best_nll:
            best_nll = nll_val

        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (step + 1) * (args.total_steps - step - 1)
            print(f"  step {step+1:4d}/{args.total_steps}  "
                  f"nll={nll_val:.4f}  log_s={ls_val:.4f}  "
                  f"s={math.exp(ls_val):.4f}  elapsed={_fmt(elapsed)}  eta={_fmt(eta)}")

    final_log_s = log_s.item()
    print(f"\n  Final log_s={final_log_s:.4f}  s={math.exp(final_log_s):.4f}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n  Evaluating (n_mc={args.n_mc_eval}) …")

    val_probs   = mc_probs(f_mu, f_var, final_log_s, args.n_mc_eval)
    val_metrics = compute_all_metrics(val_probs, labels.cpu())
    print(f"  Val:  {val_metrics}")

    test_metrics = None
    if f_mu_test is not None:
        test_probs   = mc_probs(f_mu_test, f_var_test, final_log_s, args.n_mc_eval)
        test_metrics = compute_all_metrics(test_probs, labels_test.cpu())
        print(f"  Test: {test_metrics}")

    print(f"\n--- Before vs. after calibration (val) ---")
    print(f"  {'metric':<12}  {'before':>10}  {'after':>10}  {'delta':>10}")
    print(f"  {'-'*46}")
    for k in base_val_metrics:
        if k not in val_metrics:
            continue
        before = base_val_metrics[k]
        after  = val_metrics[k]
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            print(f"  {k:<12}  {before:>10.4f}  {after:>10.4f}  {after - before:>+10.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.laplace_output_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.laplace_output_dir, f"calib_params_glm{args._suffix_str}.pt"
    )
    torch.save({"log_s_pred": final_log_s}, ckpt_path)
    print(f"\n  Saved calib params → {ckpt_path}")

    def _ser(m):
        return {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in m.items()}

    results = {
        "log_s_pred":     final_log_s,
        "s_pred":         math.exp(final_log_s),
        "best_calib_nll": best_nll,
        "val":            _ser(val_metrics),
        "val_baseline":   _ser(base_val_metrics),
        **({"test": _ser(test_metrics), "test_baseline": _ser(base_test_metrics)}
           if test_metrics is not None and base_test_metrics is not None else {}),
    }

    # Legacy path (outputs/)
    os.makedirs(args.output_dir, exist_ok=True)
    legacy_path = os.path.join(args.output_dir, f"all_results_glm_{args._la_tag}{args._suffix_str}.json")
    with open(legacy_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Clean results/ path: results/{task}/{seed_label}/
    seed_label = args.seed_label or str(args.seed)
    clean_dir  = os.path.join(args.results_dir, args.task_name, seed_label)
    os.makedirs(clean_dir, exist_ok=True)
    clean_path = os.path.join(clean_dir, f"glm{args._suffix_str}.json")
    with open(clean_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Results saved → {clean_path}")


if __name__ == "__main__":
    main()
