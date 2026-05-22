"""Convert a VP JSON checkpoint to the .pt format expected by
calibrate_vp.py --load_calib_params.

Handles two JSON formats:
  (a) all_results_vp_*.json  — list of {variant, scalars, ...} dicts
  (b) clean results JSON     — {val, scalars, ...} dict (our results/ files)

Usage:
    python convert_vp_json_to_pt.py <json_path> [<out_pt_path>]

If <out_pt_path> is omitted the .pt is written next to the JSON as
calib_params_per_layer_logit.pt.
"""

import json
import sys
from pathlib import Path

import torch


def _scalars_from_json(json_path: Path) -> dict:
    raw = json.loads(json_path.read_text())
    if isinstance(raw, list):
        # format (a): list of variant dicts
        record = next(
            (r for r in raw if r.get("variant") == "per_layer_logit"),
            raw[0],
        )
        return record["scalars"]
    else:
        # format (b): clean {val, scalars, ...} dict
        return raw["scalars"]


def convert(json_path: Path, out_path: Path) -> None:
    scalars = _scalars_from_json(json_path)

    log_T       = scalars["log_T"]         # list of N floats
    log_T_logit = scalars["log_T_logit"]   # scalar float

    ckpt = {f"log_T_list.{i}": torch.tensor([v]) for i, v in enumerate(log_T)}
    ckpt["log_T_logit"] = torch.tensor([log_T_logit])

    torch.save(ckpt, out_path)
    print(f"Saved {len(log_T)} log_T layers + log_T_logit={log_T_logit:.4f} → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    json_path = Path(sys.argv[1])
    out_path  = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        json_path.parent / "calib_params_per_layer_logit.pt"
    )
    convert(json_path, out_path)
