"""
Scan outputs/ (and outputs_laplace/) for all_results*.json files, load the
paired eval_res*.json, compute the full metric suite, and update the
all_results*.json in-place.

Usage:
    python compute_metrics.py                        # scans ./outputs
    python compute_metrics.py --dirs outputs outputs_laplace
    python compute_metrics.py --force               # recompute even if metrics exist
"""
import argparse
import json
from pathlib import Path

import torch

from metrics import compute_all_metrics


def load_eval_res(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load probs and labels from a line-delimited eval_res JSON file."""
    probs_list, labels_list = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            probs_list.append(d["probs"])
            labels_list.append(d["true"])
    probs = torch.tensor(probs_list, dtype=torch.float32)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return probs, labels


def needs_softmax(probs: torch.Tensor) -> bool:
    """Return True if probs look like raw logits rather than probabilities."""
    return bool((probs > 1).any() or (probs < 0).any())


def process_file(all_results_path: Path, force: bool) -> bool:
    """Compute metrics for one all_results*.json. Returns True if updated."""
    suffix = all_results_path.name[len("all_results"):]   # e.g. "" or "_val" or "_la_..."
    eval_res_path = all_results_path.parent / f"eval_res{suffix}"

    if not eval_res_path.exists():
        print(f"  SKIP  {all_results_path} (no paired {eval_res_path.name})")
        return False

    with open(all_results_path) as f:
        all_results = json.load(f)

    if not force and "eval_nll" in all_results:
        print(f"  SKIP  {all_results_path} (already has full metrics)")
        return False

    probs, labels = load_eval_res(eval_res_path)

    if needs_softmax(probs):
        probs = torch.softmax(probs, dim=-1)

    extra = compute_all_metrics(probs, labels)
    all_results.update({f"eval_{k}": v for k, v in extra.items()})

    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"  OK    {all_results_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", default=["outputs", "outputs_laplace"])
    parser.add_argument("--force", action="store_true", help="Recompute even if metrics already present")
    args = parser.parse_args()

    root = Path(__file__).parent
    updated = skipped = 0

    for dir_name in args.dirs:
        scan_dir = root / dir_name
        if not scan_dir.exists():
            print(f"Directory not found, skipping: {scan_dir}")
            continue
        files = sorted(scan_dir.rglob("all_results*.json"))
        print(f"\nScanning {scan_dir} — {len(files)} file(s) found")
        for path in files:
            if process_file(path, args.force):
                updated += 1
            else:
                skipped += 1

    print(f"\nDone: {updated} updated, {skipped} skipped.")


if __name__ == "__main__":
    main()
