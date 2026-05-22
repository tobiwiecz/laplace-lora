"""Average per-seed result JSONs across seeds for each task.

Structure expected:
    results/{task}/seed{N}/{method}.json   (e.g. mean.json, delta_streamlined.json)

Output written to:
    results/{task}/{method}.json           (averaged over all present seeds)

A file is skipped (with a warning) if:
  - it is missing in one or more seeds, OR
  - a top-level section (e.g. "test") is present in some seeds but not others.

Numeric leaf values (scalars) are averaged; list leaves are averaged element-wise.
"""

import json
import sys
import warnings
from pathlib import Path


RESULTS_DIR = Path("results")
SEED_PREFIX = "seed"


# ── averaging helpers ────────────────────────────────────────────────────────

def _avg_leaf(values: list):
    """Average a list of scalars or lists element-wise (recursive)."""
    if isinstance(values[0], list):
        if len({len(v) for v in values}) > 1:
            raise ValueError(f"Lists have inconsistent lengths: {[len(v) for v in values]}")
        return [_avg_leaf([v[i] for v in values]) for i in range(len(values[0]))]
    return sum(values) / len(values)


def _avg_flat_dict(dicts: list[dict]) -> tuple[dict, list[str]]:
    """Average a list of flat metric dicts. Returns (averaged, warnings)."""
    all_keys = set().union(*(d.keys() for d in dicts))
    common   = [k for k in all_keys if all(k in d for d in dicts)]
    missing  = sorted(all_keys - set(common))
    averaged = {k: _avg_leaf([d[k] for d in dicts]) for k in sorted(common)}
    return averaged, missing


def _avg_nested(dicts: list[dict]) -> tuple[dict, list[str]]:
    """Average a list of possibly-nested JSONs (flat or {val, test, scalars}).

    Top-level sections that are missing in any seed are dropped with a warning.
    Sections present in all seeds are averaged recursively via _avg_flat_dict.
    """
    all_sections = set().union(*(d.keys() for d in dicts))
    common       = [k for k in all_sections if all(k in d for d in dicts)]
    dropped      = sorted(all_sections - set(common))

    _section_order = ["val", "test", "scalars"]
    ordered = [s for s in _section_order if s in common]
    ordered += sorted(s for s in common if s not in _section_order)

    averaged = {}
    for section in ordered:
        vals = [d[section] for d in dicts]
        if isinstance(vals[0], dict):
            avg_section, missing_keys = _avg_flat_dict(vals)
            averaged[section] = avg_section
            if missing_keys:
                dropped.extend(f"{section}.{k}" for k in missing_keys)
        else:
            averaged[section] = _avg_leaf(vals)

    return averaged, dropped


# ── per-task aggregation ─────────────────────────────────────────────────────

def aggregate_task(task_dir: Path) -> None:
    seed_dirs = sorted(d for d in task_dir.iterdir()
                       if d.is_dir() and d.name.startswith(SEED_PREFIX))
    if not seed_dirs:
        return

    n = len(seed_dirs)
    print(f"\n[{task_dir.name}]  {n} seeds: {[d.name for d in seed_dirs]}")

    # Collect all JSON filenames across seeds
    all_files: set[str] = set().union(
        *(set(p.name for p in sd.glob("*.json")) for sd in seed_dirs)
    )

    for fname in sorted(all_files):
        have     = [sd for sd in seed_dirs if (sd / fname).exists()]
        missing  = [sd.name for sd in seed_dirs if sd not in have]

        if missing:
            warnings.warn(
                f"  SKIP  {task_dir.name}/{fname}: missing in {missing} ({len(have)}/{n} seeds)"
            )
            continue

        dicts = [json.loads((sd / fname).read_text()) for sd in seed_dirs]

        try:
            averaged, dropped = _avg_nested(dicts)
        except Exception as e:
            warnings.warn(f"  ERROR {task_dir.name}/{fname}: {e}")
            continue

        out_path = task_dir / fname
        out_path.write_text(json.dumps(averaged, indent=2))

        sections = ", ".join(averaged.keys())
        print(f"  OK    {fname}  [{sections}]  ({n} seeds)")
        for key in dropped:
            warnings.warn(f"  NOTE  {task_dir.name}/{fname}: dropped '{key}' (not in all seeds)")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    results_dir = RESULTS_DIR
    if not results_dir.exists():
        sys.exit(f"results/ directory not found: {results_dir.resolve()}")

    task_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and not d.name.startswith(SEED_PREFIX)
    )
    if not task_dirs:
        sys.exit("No task directories found under results/")

    print(f"Tasks: {[d.name for d in task_dirs]}")
    warnings.simplefilter("always")

    for task_dir in task_dirs:
        aggregate_task(task_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
