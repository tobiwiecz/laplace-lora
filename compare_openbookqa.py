#!/usr/bin/env python3
"""Compare VP calibration vs baselines on openbookqa — val and test.

Reads:
  - Baseline val metrics from all_results*.json  (per seed, already exist)
  - VP val+test metrics from all_results_vp_*.json (produced by calibrate_vp.py --eval_on_test)

Usage:
  python compare_openbookqa.py [--load_step 4000] [--variant per_layer_logit]
"""
import argparse
import json
import numpy as np
from pathlib import Path

SEEDS      = [1, 2, 3, 4, 5]
MODEL      = "meta-llama/Llama-2-7b-chat-hf"
PEFT_TAG   = "lora_lmhead_16_0.1_5e-05"
TASK       = "openbookqa"


def seed_dir(seed: int, load_step: int) -> Path:
    return (Path("outputs") / TASK / "meta-llama"
            / f"Llama-2-7b-chat-hf_{PEFT_TAG}_seed{seed}"
            / f"step_{load_step}")


def load_baseline(filename: str, load_step: int) -> dict[str, tuple[float, float]]:
    """Load a baseline across seeds → {metric: (mean, sem)}."""
    rows = []
    for s in SEEDS:
        p = seed_dir(s, load_step) / filename
        try:
            rows.append(json.loads(p.read_text()))
        except FileNotFoundError:
            pass
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if isinstance(r[k], (int, float))])
        if len(vals):
            out[k] = (vals.mean(), vals.std(ddof=1) / len(vals) ** 0.5)
    return out


def load_vp(load_step: int, hessian: str, variant: str) -> tuple[dict, dict]:
    """Load VP val and test metrics across seeds from all_results_vp_*.json."""
    val_rows, test_rows = [], []
    for s in SEEDS:
        # glob for any matching vp result file
        d = seed_dir(s, load_step)
        matches = list(d.glob(f"all_results_vp_{hessian}_*.json"))
        if not matches:
            continue
        try:
            results = json.loads(matches[0].read_text())
        except Exception:
            continue
        # results is a list of variant dicts
        for r in results:
            if r.get("variant") == variant:
                val_rows.append(r.get("val") or r.get("eval") or {})
                if r.get("test"):
                    test_rows.append(r["test"])
                break

    def aggregate(rows):
        if not rows:
            return {}
        keys = rows[0].keys()
        out = {}
        for k in keys:
            vals = np.array([r[k] for r in rows if isinstance(r.get(k), (int, float))])
            if len(vals):
                out[k] = (vals.mean(), vals.std(ddof=1) / len(vals) ** 0.5)
        return out

    return aggregate(val_rows), aggregate(test_rows)


def fmt(mean_sem, key="loss"):
    if not mean_sem or key not in mean_sem:
        return "   —   "
    m, s = mean_sem[key]
    return f"{m:.4f}±{s:.4f}"


def print_table(load_step: int, hessian: str, variant: str):
    baselines = {
        "MAP mean":   load_baseline("all_results.json", load_step),
        "Laplace diag": load_baseline(f"all_results_la_diag_all_homo_mc_corr_100.json", load_step),
        "Laplace kron": load_baseline(f"all_results_la_kron_all_homo_mc_corr_100.json", load_step),
    }
    vp_val, vp_test = load_vp(load_step, hessian, variant)

    metrics = ["loss", "accuracy", "ece"]
    header  = f"{'Method':<20}  " + "  ".join(f"{'val '+m:<18}  {'test '+m:<18}" for m in metrics)
    print(f"\nopenbookqa  |  hessian={hessian}  |  VP variant={variant}\n")
    print(header)
    print("-" * len(header))

    for label, val_bl in baselines.items():
        row = f"{label:<20}  "
        for m in metrics:
            row += f"{fmt(val_bl, m):<20}  {'—':^20}  "
        print(row)

    row = f"{'VP calib':<20}  "
    for m in metrics:
        row += f"{fmt(vp_val, m):<20}  {fmt(vp_test, m):<20}  "
    print(row)
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load_step", type=int, default=4000)
    p.add_argument("--hessian",   default="diag", choices=["diag", "kron"])
    p.add_argument("--variant",   default="per_layer_logit")
    args = p.parse_args()
    print_table(args.load_step, args.hessian, args.variant)


if __name__ == "__main__":
    main()
