"""
Scan outputs/ (and outputs_laplace/) for all_results*.json files, compute the
full metric suite from the paired eval_res*.json, and optionally average across seeds.

Usage:
    python compute_metrics.py                  # update all_results*.json in-place
    python compute_metrics.py --force          # recompute even if metrics exist
    python compute_metrics.py --avg            # also write seed-averaged files one level up
    python compute_metrics.py --avg --avg-only # only write seed-averaged files, skip per-seed update
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from metrics import compute_all_metrics


def load_eval_res(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
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
    return bool((probs > 1).any() or (probs < 0).any())


def process_file(all_results_path: Path, force: bool) -> bool:
    """Compute and write full metrics for one all_results*.json. Returns True if updated."""
    suffix = all_results_path.name[len("all_results"):]
    eval_res_path = all_results_path.parent / f"eval_res{suffix}"

    if not eval_res_path.exists():
        return False

    with open(all_results_path) as f:
        all_results = json.load(f)

    if not force and "loss" in all_results:
        return False

    probs, labels = load_eval_res(eval_res_path)
    if needs_softmax(probs):
        probs = torch.softmax(probs, dim=-1)

    clean = {k.removeprefix("eval_"): v for k, v in all_results.items()}
    clean.update(compute_all_metrics(probs, labels))

    with open(all_results_path, "w") as f:
        json.dump(clean, f, indent=2)

    return True


def avg_seed_folder(run_folder: str) -> str:
    """Strip trailing seed suffix (e.g. '_21', '_42') from a run folder name."""
    return re.sub(r"_\d+$", "", run_folder)


def avg_across_seeds(scan_dir: Path, force: bool) -> int:
    """
    For every group of all_results*.json files that share the same
    (task, model_family, run_base, step, filename) but differ only in seed,
    write a seed-averaged JSON one level above the seed folder.
    Returns number of files written.
    """
    # Group by: path with seed-folder replaced by its base
    groups: dict[Path, list[Path]] = defaultdict(list)
    for path in sorted(scan_dir.rglob("all_results*.json")):
        # path structure: scan_dir / task / model_family / run_seed / step_N / filename
        parts = path.relative_to(scan_dir).parts
        if len(parts) < 4:
            continue
        # parts[-3] is the seed run folder (e.g. Llama-2-7b-chat-hf_lora_lmhead_16_0.1_5e-05_21)
        # parts[-2] is step_N, parts[-1] is filename
        run_base = avg_seed_folder(parts[-3])
        avg_path = scan_dir / Path(*parts[:-3]) / run_base / parts[-2] / parts[-1]
        groups[avg_path].append(path)

    written = 0
    for avg_path, seed_paths in sorted(groups.items()):
        if not force and avg_path.exists():
            continue

        seed_dicts = []
        for p in seed_paths:
            with open(p) as f:
                d = json.load(f)
            if "loss" not in d:
                break  # skip groups with incomplete metrics
            seed_dicts.append(d)
        else:
            if not seed_dicts:
                continue
            keys = seed_dicts[0].keys()
            averaged = {
                k: sum(d[k] for d in seed_dicts) / len(seed_dicts)
                for k in keys
                if isinstance(seed_dicts[0][k], (int, float))
            }
            averaged["n_seeds"] = len(seed_dicts)
            avg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(avg_path, "w") as f:
                json.dump(averaged, f, indent=2)
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", default=["outputs", "outputs_laplace"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--avg", action="store_true", help="Write seed-averaged files one level up")
    parser.add_argument("--avg-only", action="store_true", help="Skip per-seed update, only write averages")
    args = parser.parse_args()

    root = Path(__file__).parent
    updated = skipped = avg_written = 0

    for dir_name in args.dirs:
        scan_dir = root / dir_name
        if not scan_dir.exists():
            continue

        if not args.avg_only:
            for path in sorted(scan_dir.rglob("all_results*.json")):
                if process_file(path, args.force):
                    updated += 1
                else:
                    skipped += 1

        if args.avg or args.avg_only:
            avg_written += avg_across_seeds(scan_dir, args.force)

    parts = [f"{updated} updated, {skipped} skipped"]
    if args.avg or args.avg_only:
        parts.append(f"{avg_written} averages written")
    print("Done: " + ", ".join(parts) + ".")


if __name__ == "__main__":
    main()
