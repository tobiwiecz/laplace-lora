#!/usr/bin/env python3
"""benchmark_inference.py — per-batch GPU latency: MAP vs VP vs GLM.

Edit the CONFIGURATION block below, then run:
  python benchmark_inference.py
"""

import gc
import os
import sys
import warnings

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from peft import PeftConfig, PeftModel
from accelerate.utils import set_seed
from laplace import KronLaplace

# Import VP infrastructure from calibrate_vp (triggers MVP module loading)
import calibrate_vp as cvp

# Disable grad checkpointing for inference — it saves memory during training
# but recomputes activations and slows down a pure forward pass.
cvp.USE_GRAD_CHECKPOINT = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit here
# ══════════════════════════════════════════════════════════════════════════════
TASK               = "openbookqa"   # openbookqa | ARC-Easy | ARC-Challenge | boolq | winogrande_*
SEED               = 21             # 21 42 87 13 100
LOAD_STEP          = 4000

BATCH_SIZE         = 16             # examples per batch
N_BATCHES          = 1            # batches to average timing over
N_WARMUP           = 5             # discarded warm-up iterations per batch
N_RUNS             = 10            # timed iterations per batch

N_MC_VP            = 100           # logit-MC samples for VP
N_MC_GLM           = 100_000       # Cholesky-MC samples for GLM full

COMPILE_MAP        = False        # torch.compile MAP model — broken with current PEFT+transformers (graph-break in _enable_peft_forward_hooks → NameError in output_capturing.py)
COMPILE_VP         = True         # torch.compile VP model (dynamic=True)
EXPLAIN_COMPILE    = False        # run graph-break analysis (needs ~200 MiB extra GPU headroom)
RUN_GLM            = True          # False → skip GLM (faster, MAP+VP only)

# ── Fixed settings ────────────────────────────────────────────────────────────
MAX_LENGTH         = 300
LORA_R, LORA_ALPHA = 8, 16
LORA_DROPOUT       = 0.1
HESSIAN            = "kron"
LAPLACE_SUB        = "all"
LAPLACE_PRIOR      = "homo"
LAPLACE_OPTIM_STEP = 100
# ══════════════════════════════════════════════════════════════════════════════

SEED_TO_LABEL = {21: "seed1", 42: "seed2", 87: "seed3", 13: "seed4", 100: "seed5"}
MODEL_ID      = "meta-llama/Llama-2-7b-chat-hf"
PEFT_TAG      = "lora_lmhead_16_0.1_5e-05"


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def cuda_timer(fn, n_warmup: int, n_runs: int) -> np.ndarray:
    """Returns per-call latency in ms (GPU wall-clock via CUDA events)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn()
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return np.array(times)


def static_mb() -> float:
    """Total GPU memory currently allocated (model weights + buffers), in MiB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def peak_mb(fn) -> float:
    """Total peak GPU memory during one forward pass in MiB (weights + activations)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def report(label: str, ms: np.ndarray, batch_size: int,
           baseline_ms: float = None, mem_mb: float = None):
    ratio = f"  {ms.mean() / baseline_ms:5.2f}× MAP" if baseline_ms else ""
    mem   = f"  {mem_mb:7.0f} MiB" if mem_mb is not None else ""
    print(
        f"  {label:<38s}  "
        f"{ms.mean():7.2f} ± {ms.std():5.2f} ms/batch  "
        f"({ms.mean() / batch_size:.2f} ms/ex)"
        f"{ratio}{mem}"
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_batches(tokenizer, device):
    """Return a list of N_BATCHES GPU-ready batch dicts."""
    if "ARC" in TASK:
        raw = load_dataset("ai2_arc", TASK)
    elif TASK == "openbookqa":
        raw = load_dataset("openbookqa")
    elif TASK == "boolq":
        raw = load_dataset("super_glue", "boolq")
    elif "winogrande" in TASK:
        raw = load_dataset("winogrande", TASK)
    else:
        raise ValueError(f"Unknown task: {TASK}")

    # Filter + normalise (same as calibrate_vp.py).
    # new_fingerprint bypasses dill-based cache-key hashing, avoiding PicklingWarnings
    # from dill inspecting the MVP enum classes embedded in the cvp module closure.
    if "ARC" in TASK or TASK == "openbookqa":
        for split in raw:
            raw[split] = raw[split].filter(
                lambda ex: len(ex["choices"]["label"]) == 4,
                new_fingerprint=f"filter4_{TASK}_{split}",
            )
        def _convert(ex):
            m = {"1": "A", "2": "B", "3": "C", "4": "D"}
            ex["choices"]["label"] = [m.get(l, l) for l in ex["choices"]["label"]]
            ex["answerKey"] = m.get(ex["answerKey"], ex["answerKey"])
            ex["choices"]["text"] = [
                (t if t.endswith(".") else t + ".") for t in ex["choices"]["text"]
            ]
            ex["choices"]["text"] = [
                (t[0].upper() + t[1:] if t else t) for t in ex["choices"]["text"]
            ]
            return ex
        for split in raw:
            raw[split] = raw[split].map(
                _convert, new_fingerprint=f"convert_{TASK}_{split}"
            )

    padding = False
    for split in raw:
        raw[split] = raw[split].map(
            lambda ex: cvp.preprocess_function(ex, TASK, tokenizer, padding, MAX_LENGTH),
            batched=True,
            remove_columns=raw[split].column_names,
            desc=f"Tokenising {split}",
            new_fingerprint=f"preprocess_{TASK}_{MAX_LENGTH}_{split}",
        )
    processed = raw

    # Prefer test split for truly unseen data; fall back to validation
    split_name = "test" if "test" in processed else "validation"
    ds = processed[split_name]
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    batches = []
    for batch in loader:
        batches.append({k: v.to(device) for k, v in batch.items()})
        if len(batches) >= N_BATCHES:
            break
    print(f"  Loaded {len(batches)} batch(es) from '{split_name}' split "
          f"({BATCH_SIZE} ex/batch, seq_len≈{batches[0]['input_ids'].shape[1]})")
    return batches


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

class CustomLMHead(nn.Module):
    """Vocabulary-trimmed LoRA LM head (identical to calibrate_vp.py)."""

    def __init__(self, original_lm_head, id_list, lora_r, lora_alpha, device):
        super().__init__()
        self.id_list = id_list
        w = original_lm_head.weight[id_list, :].clone()
        self.linear = nn.Linear(w.shape[1], len(id_list), bias=False, device=device)
        self.linear.weight.data = w.float()
        self.linear.weight.requires_grad = False

        self.lora_dropout = original_lm_head.lora_dropout["default"]
        Aw = original_lm_head.lora_A["default"].weight.clone()
        self.lora_A = nn.Linear(Aw.shape[1], Aw.shape[0], bias=False, device=device)
        self.lora_A.weight.data = Aw.float()
        self.lora_A.weight.requires_grad = True

        Bw = original_lm_head.lora_B["default"].weight[id_list, :].clone()
        self.lora_B = nn.Linear(Bw.shape[1], len(id_list), bias=False, device=device)
        self.lora_B.weight.data = Bw.float()
        self.lora_B.weight.requires_grad = True

        self.scaling = lora_alpha / lora_r

    def forward(self, x):
        h = x[:, -1, :].float()
        return self.linear(h) + self.lora_B(self.lora_A(self.lora_dropout(h))) * self.scaling


class WrappedModel(nn.Module):
    """Thin wrapper matching run_gpt_laplace.py — output: [B, n_classes] logits."""

    def __init__(self, model, output_size):
        super().__init__()
        self.model = model
        self.output_size = output_size  # required by laplace asdl Jacobians loop

    def forward(self, **kwargs):
        kwargs.pop("labels", None)
        return self.model(**kwargs)["logits"].float()


def build_models(seed_label, device):
    """Load PEFT model and build MAP / VP / GLM variants. Returns all three."""

    output_dir   = (f"outputs/{TASK}/{MODEL_ID}_{PEFT_TAG}_{seed_label}"
                    f"/step_{LOAD_STEP}")
    laplace_dir  = (f"outputs_laplace/{TASK}/{MODEL_ID}_{PEFT_TAG}_{seed_label}"
                    f"/step_{LOAD_STEP}")

    print(f"\n  checkpoint : {output_dir}")
    print(f"  laplace    : {laplace_dir}")

    # ── Base PEFT model ──────────────────────────────────────────────────────
    peft_cfg   = PeftConfig.from_pretrained(output_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_cfg.base_model_name_or_path, dtype=torch.bfloat16, token=True
    )
    model = PeftModel.from_pretrained(base_model, output_dir)

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "lora" in name:
            param.requires_grad = True

    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    print(f"  mem after model load  : {torch.cuda.memory_allocated() / 1024**2:.0f} MiB")

    # ── Answer token ids ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, padding_side="left", token=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.bos_token
    if TASK == "boolq":
        tokenizer.add_eos_token = True

    if TASK == "boolq":
        id_list = [tokenizer.encode("False", add_special_tokens=False)[0],
                   tokenizer.encode("True",  add_special_tokens=False)[0]]
    elif "winogrande" in TASK:
        id_list = [tokenizer.encode(c, add_special_tokens=False)[0] for c in "AB"]
    else:
        id_list = [tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCD"]

    # ── Replace lm_head with trimmed LoRA head (used by all three models) ───
    orig_lm_head = model.base_model.model.lm_head
    custom_head  = CustomLMHead(orig_lm_head, id_list, LORA_R, LORA_ALPHA, device)
    model.base_model.model.lm_head = custom_head
    model.gradient_checkpointing_disable()

    # ── Load KFAC posterior ──────────────────────────────────────────────────
    H_path  = f"{laplace_dir}/laplace_H_{HESSIAN}_{LAPLACE_SUB}.pt"
    pp_path = (f"{laplace_dir}/prior_precision_{HESSIAN}_{LAPLACE_SUB}"
               f"_{LAPLACE_PRIOR}_{LAPLACE_OPTIM_STEP}.pt")
    H_dict         = torch.load(H_path,  map_location=device, weights_only=False)
    prior_precision = torch.load(pp_path, map_location=device, weights_only=False)
    print(f"  prior_precision = {prior_precision.item():.4f}")
    print(f"  mem after KFAC load   : {torch.cuda.memory_allocated() / 1024**2:.0f} MiB")

    trainable_params = [
        (f"model.{n}", p) for n, p in model.named_parameters() if p.requires_grad
    ]
    param_variances = cvp.extract_param_variances(
        H_dict, prior_precision, trainable_params, device
    )

    # ── (a) MAP model ────────────────────────────────────────────────────────
    map_model = model  # already set up above

    # ── (b) VP model ────────────────────────────────────────────────────────
    lora_A_var = next(v for n, v in param_variances.items() if "lm_head.lora_A" in n)
    lora_B_var = next(v for n, v in param_variances.items() if "lm_head.lora_B" in n)

    backbone = model.base_model.model.model
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    vp_backbone = cvp.build_vp_backbone(
        backbone, param_variances, "mvp", "exact"
    ).to(device)
    vp_backbone.eval()

    vp_head = cvp.VPLMHead(custom_head, lora_A_var, lora_B_var).to(device)
    vp_model = cvp.VPCalibPerLayerLogit(vp_backbone, vp_head).to(device)

    # Load calibrated T parameters
    calib_path = f"{laplace_dir}/calib_params_per_layer_logit.pt"
    if os.path.exists(calib_path):
        ckpt = torch.load(calib_path, map_location=device, weights_only=False)
        with torch.no_grad():
            for n, p in vp_model.named_parameters():
                if n in ckpt:
                    p.data.copy_(ckpt[n].to(device))
        print(f"  VP: loaded calibrated params from {calib_path}")
        print(f"      log_T_logit = {vp_model.log_T_logit.item():.4f}  "
              f"log_T mean = {vp_model.log_T.mean().item():.4f}")
    else:
        print(f"  VP: calib checkpoint not found — using T=1 (uncalibrated)")
    vp_model.eval()
    del param_variances
    gc.collect(); torch.cuda.synchronize()
    print(f"  mem after VP build    : {torch.cuda.memory_allocated() / 1024**2:.0f} MiB")

    if COMPILE_VP:
        if EXPLAIN_COMPILE:
            torch._logging.set_logs(graph_breaks=True)
        print("  VP: compiling with torch.compile(dynamic=True) "
              "(graph breaks logged during warmup)" if EXPLAIN_COMPILE
              else "  VP: compiling with torch.compile(dynamic=True) ...")
        vp_model = torch.compile(vp_model, dynamic=True)

    # ── (c) GLM model (KronLaplace) ──────────────────────────────────────────
    # VP setup set backbone.requires_grad=False on the shared backbone, so restore
    # LoRA requires_grad on map_model so GLM Jacobians cover all KFAC layers.
    for name, param in map_model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    wrapped = WrappedModel(map_model, output_size=len(id_list)).to(device)
    wrapped.eval()

    # Import AsdlGGN directly from the local asdl module so the backend is always
    # set explicitly — avoids relying on the default in baselaplace.py which can
    # silently resolve to None when the asdl import path is ambiguous.
    from laplace.curvature.asdl import AsdlGGN as _AsdlGGN

    la = KronLaplace(
        wrapped,
        likelihood="classification",
        prior_precision=float(prior_precision),
        backend=_AsdlGGN,
    )
    la.H         = H_dict["H"]
    la.n_outputs = H_dict["n_outputs"]
    la.prior_precision = prior_precision
    print(f"  GLM: KronLaplace built  (n_outputs={la.n_outputs})")
    print(f"  mem after GLM build   : {torch.cuda.memory_allocated() / 1024**2:.0f} MiB")

    # Compile MAP model after GLM setup so la retains the uncompiled reference.
    if COMPILE_MAP:
        print("  MAP: compiling with torch.compile(dynamic=True) ...")
        map_model = torch.compile(map_model, dynamic=True)

    return tokenizer, map_model, vp_model, la, id_list


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def run_map(model, batch):
    ids  = batch["input_ids"]
    mask = batch["attention_mask"]
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask)["logits"]


def run_vp(vp_model, batch, n_mc: int):
    ids  = batch["input_ids"]
    mask = batch["attention_mask"]
    with torch.no_grad():
        return cvp.mc_probs(vp_model(ids, mask), n_mc)


def run_glm_dist(la, batch):
    """_glm_predictive_distribution only (no MC sampling)."""
    return la._glm_predictive_distribution(batch)


def run_glm_full(la, batch, n_mc: int):
    """_glm_predictive_distribution + Cholesky MC sampling."""
    f_mu, f_var = la._glm_predictive_distribution(batch)
    B, C = f_mu.shape
    eps  = torch.randn(n_mc, B, C, device=f_mu.device)
    eye  = torch.eye(C, device=f_var.device) * 1e-6
    L    = torch.linalg.cholesky(f_var + eye)          # [B, C, C]
    # [n_mc, B, C] = f_mu + L @ eps[..., None]
    logits = f_mu.unsqueeze(0) + (L.unsqueeze(0) @ eps.unsqueeze(-1)).squeeze(-1)
    return F.softmax(logits, dim=-1).mean(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_label = SEED_TO_LABEL.get(SEED, str(SEED))
    set_seed(SEED)

    print("=" * 70)
    print(f"Inference benchmark  |  {TASK}  |  {seed_label}  "
          f"|  step {LOAD_STEP}  |  device {device}")
    print("=" * 70)

    # ── Tokenizer (needed for data loading) ──────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, padding_side="left", token=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.bos_token

    print("\n[1/3] Loading data ...")
    batches = load_batches(tokenizer, device)

    print("\n[2/3] Building models ...")
    _, map_model, vp_model, la, id_list = build_models(seed_label, device)
    print(f"  Static GPU footprint (all models loaded): {static_mb():.0f} MiB")

    print(f"\n[3/3] Benchmarking  "
          f"(warmup={N_WARMUP}, runs={N_RUNS} per batch × {N_BATCHES} batches)\n")

    all_map, all_vp = [], []
    all_glm_dist, all_glm_full = [], []

    for i, batch in enumerate(batches):
        print(f"  --- batch {i+1}/{len(batches)} "
              f"(shape {list(batch['input_ids'].shape)}) ---")

        ms  = cuda_timer(lambda: run_map(map_model, batch), N_WARMUP, N_RUNS)
        mem = peak_mb(lambda: run_map(map_model, batch))
        all_map.append(ms)
        report("MAP", ms, BATCH_SIZE, mem_mb=mem)

        ms  = cuda_timer(lambda: run_vp(vp_model, batch, N_MC_VP), N_WARMUP, N_RUNS)
        mem = peak_mb(lambda: run_vp(vp_model, batch, N_MC_VP))
        all_vp.append(ms)
        report(f"VP  ({N_MC_VP} logit-MC)", ms, BATCH_SIZE, all_map[-1].mean(), mem_mb=mem)

        if RUN_GLM:
            torch.cuda.empty_cache()
            ms  = cuda_timer(lambda: run_glm_dist(la, batch), N_WARMUP, N_RUNS)
            mem = peak_mb(lambda: run_glm_dist(la, batch))
            all_glm_dist.append(ms)
            report("GLM  (_glm_pred_dist only)", ms, BATCH_SIZE, all_map[-1].mean(), mem_mb=mem)

            ms  = cuda_timer(lambda: run_glm_full(la, batch, N_MC_GLM), N_WARMUP, N_RUNS)
            mem = peak_mb(lambda: run_glm_full(la, batch, N_MC_GLM))
            all_glm_full.append(ms)
            report(f"GLM  (+{N_MC_GLM//1000}k Cholesky-MC)", ms, BATCH_SIZE, all_map[-1].mean(), mem_mb=mem)

    # ── Summary ───────────────────────────────────────────────────────────────
    def pool(lst):
        return np.concatenate(lst)

    map_all = pool(all_map)
    vp_all  = pool(all_vp)

    print(f"\n{'='*70}")
    print(f"Summary  (all {len(batches)} batches × {N_RUNS} runs pooled)\n")
    baseline = map_all.mean()
    report("MAP  (baseline)", map_all, BATCH_SIZE)
    report(f"VP   ({N_MC_VP} logit-MC)", vp_all, BATCH_SIZE, baseline)
    if all_glm_dist:
        report("GLM  (dist only)",              pool(all_glm_dist), BATCH_SIZE, baseline)
        report(f"GLM  (+{N_MC_GLM//1000}k MC)", pool(all_glm_full), BATCH_SIZE, baseline)

    print(f"\n  VP overhead vs MAP:        {vp_all.mean()/baseline:.2f}×")
    if all_glm_dist:
        glm_d = pool(all_glm_dist).mean()
        glm_f = pool(all_glm_full).mean()
        print(f"  GLM-dist overhead vs MAP:  {glm_d/baseline:.2f}×")
        print(f"  GLM-full overhead vs MAP:  {glm_f/baseline:.2f}×")
        print(f"  VP vs GLM-dist:            {vp_all.mean()/glm_d:.2f}×  "
              f"({'faster' if vp_all.mean() < glm_d else 'slower'})")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    import json
    out = {
        "task": TASK, "seed": seed_label, "load_step": LOAD_STEP,
        "batch_size": BATCH_SIZE, "n_mc_vp": N_MC_VP, "n_mc_glm": N_MC_GLM,
        "compile_map": COMPILE_MAP, "compile_vp": COMPILE_VP,
        "map_ms_mean":  float(map_all.mean()),  "map_ms_std":  float(map_all.std()),
        "vp_ms_mean":   float(vp_all.mean()),   "vp_ms_std":   float(vp_all.std()),
    }
    if all_glm_dist:
        gd = pool(all_glm_dist); gf = pool(all_glm_full)
        out.update({
            "glm_dist_ms_mean": float(gd.mean()), "glm_dist_ms_std": float(gd.std()),
            "glm_full_ms_mean": float(gf.mean()), "glm_full_ms_std": float(gf.std()),
        })
    tag = f"bench_{TASK}_{seed_label}_bs{BATCH_SIZE}"
    if COMPILE_MAP or COMPILE_VP:
        parts = []
        if COMPILE_MAP: parts.append("map")
        if COMPILE_VP:  parts.append("vp")
        tag += "_compiled_" + "+".join(parts)
    path = f"{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved → {path}")


if __name__ == "__main__":
    main()
