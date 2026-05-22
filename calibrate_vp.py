"""Calibrate Llama-2 LoRA uncertainty via MVP variance propagation.

Fits KFAC Laplace on LoRA parameters, then calibrates two classes of learnable
scalars on the validation NLL:

  log_s  (global, 1 param): scales all LoRA posterior variances entering
         LoRALinearMVP.  Calibrates how much to trust the KFAC posterior.

  log_T  (layer-wise, various shapes): scales the *propagated* activation
         variance at key positions in the backbone.  Calibrates whether the
         VP formulas (RMSNorm, SwiGLU, attention mixing) track the true
         variance accumulation.

Parameterisation variants (flags at top of file):
  log_s_only         — full backbone VP, no temperature scaling; log_s (1 param)
  log_s_per_block    — phase-1: per-layer log_s[N] (N params)
  global             — full backbone VP; log_s + global log_T (2 params)
  per_layer          — full backbone VP; log_s + log_T[N] (N+1 params)
  per_sub_block      — full backbone VP; log_s + log_T[N,2] (2N+1 params)
  per_sub_block_logit— phase-2: log_s frozen + log_T_attn + log_T_mlp + log_T_logit (3 params)

VP methods for backbone components:
  RMSNorm : streamlined or mvp (controlled by RMS_NORM_METHOD)
  SwiGLU  : Hermite k=3 (SiLU≈GELU rescaling) + exact product var + cross-covariance
  Attention: VALUE_ONLY — attention pattern deterministic, V uncertain

Grad-checkpointing (USE_GRAD_CHECKPOINT) halves the memory footprint of
the backbone VP by recomputing sub-block activations during backward.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")
logging.getLogger("accelerate.utils.other").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import datasets
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorWithPadding, default_data_collator,
)
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm, apply_rotary_pos_emb, create_causal_mask,
)
from peft import PeftModel, PeftConfig

import importlib.util
import types

# mvp/models/__init__.py imports Beit3 which is not installed in this env.
# Load only the two modules we need directly from file, bypassing __init__.py
# by pre-registering stub packages in sys.modules.
_MVP_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MVP', 'src'))

def _load_mvp_module(dotted_name: str) -> types.ModuleType:
    parts = dotted_name.split('.')
    # Ensure each parent package stub exists with __path__ so Python can
    # resolve sibling submodule imports without running __init__.py.
    for i in range(1, len(parts)):
        pkg = '.'.join(parts[:i])
        if pkg not in sys.modules:
            stub = types.ModuleType(pkg)
            stub.__path__ = [os.path.join(_MVP_SRC, *parts[:i])]
            stub.__package__ = pkg
            sys.modules[pkg] = stub
    path = os.path.join(_MVP_SRC, *parts[:-1], parts[-1] + '.py')
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load core.py first; its bottom-of-file imports (core_layernorm, core_lora,
# etc.) are triggered from within core.py's exec, so the circular dependency
# resolves naturally — siblings can already see ParamPair in the in-progress
# core module by the time they import it.
_load_mvp_module('mvp.models.mvp.core')

from mvp.models.mvp.core      import ParamPair
from mvp.models.mvp.core_lora import LoRALinearMVP

from metrics import compute_all_metrics

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
FIXED_LOG_S      = -4.0   # fixed posterior scale (exp(-4) ≈ 0.018); not learned
LR               = 1e-2   # learning rate for log_s (max, at last layer for per-block; unused when fixed)
LR_T             = 1e-1   # learning rate for log_T (max at last layer); log_T_logit gets 10×
WEIGHT_DECAY_T   = 1e-2   # weight decay for per-layer log_T (pulls toward 0 = T→1)
LR_MIN_FACTOR    = 0.1    # min LR at layer 0 = LR_MIN_FACTOR * max LR (linear ramp across layers)
INIT_LOG_S       = None   # if set, skip phase-1 and use this value directly (e.g. -0.63)
FINETUNE_S       = False  # True → backbone variants optimize log_s at LR_S_FINETUNE; False → freeze it
LR_S_FINETUNE    = 1e-3   # LR for log_s in backbone variant phase (only used if FINETUNE_S=True)
N_EPOCHS         = 50     # calibration epochs (phase-2: T parameters); overridden by --total_steps
N_EPOCHS_S       = 5      # calibration epochs for phase-1 (log_s); overridden by --total_steps_s
CALIB_BATCH_SIZE = 16     # micro-batch size per gradient step
GRAD_ACCUM       = 16     # accumulate this many micro-batches per optimizer step
N_MC_CALIB       = 100    # MC samples per NLL estimate during calibration
N_MC_EVAL        = 1000   # MC samples for final evaluation
RMS_NORM_METHOD  = "mvp"      # "streamlined" or "mvp"
SWIGLU_METHOD    = "exact"    # "delta" or "exact" (Hermite k=3 + product var + cross-cov)
ATTN_VAR_MODE    = "value_only"  # "value_only" or "full" (delta method through softmax for Q)
USE_GRAD_CHECKPOINT = True   # recompute sub-block activations during backward

# Which variants to run (flip flags to activate)
RUN_LOG_S_ONLY          = False   # full backbone VP, no T scaling; global log_s (phase-1)
RUN_LOG_S_PER_BLOCK     = False   # alternative phase-1: per-layer log_s[N] (mutually exclusive with LOG_S_ONLY)
RUN_GLOBAL              = False   # full backbone VP; log_s + global T
RUN_PER_LAYER           = False   # full backbone VP; log_s + T per layer
RUN_PER_SUB_BLOCK       = False   # full backbone VP; log_s + T per (attn,MLP)
RUN_PER_SUB_BLOCK_LOGIT = False   # shared log_T_attn + log_T_mlp + log_T_logit (3 params)
RUN_PER_LAYER_LOGIT     = True    # per-block T[N] + log_T_logit; log_s fixed at FIXED_LOG_S
RUN_PER_BLOCK_ST        = False   # single phase: per-block log_s[N] + log_T[N] + log_T_logit
# ══════════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_name",           type=str,   default="winogrande_s")
    p.add_argument("--model_name_or_path",  type=str,   default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--max_length",          type=int,   default=300)
    p.add_argument("--pad_to_max_length",   action="store_true")
    p.add_argument("--use_slow_tokenizer",  action="store_true")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size",  type=int, default=8)
    p.add_argument("--output_dir",          type=str,   default="./outputs")
    p.add_argument("--seed",                type=int,   default=21)
    p.add_argument("--seed_label",          type=str,   default=None)
    p.add_argument("--load_step",           type=int,   default=4000)
    p.add_argument("--lora_r",              type=int,   default=8)
    p.add_argument("--lora_alpha",          type=int,   default=16)
    p.add_argument("--lora_dropout",        type=float, default=0.1)
    p.add_argument("--laplace_hessian",     type=str,   default="kron")
    p.add_argument("--laplace_sub",         type=str,   default="log_s_only")
    p.add_argument("--laplace_prior",       type=str,   default="homo")
    p.add_argument("--laplace_optim_step",  type=int,   default=100)
    p.add_argument("--testing_set",         type=str,   default="val")
    p.add_argument("--lm_head",             action="store_true", default=True)
    p.add_argument("--swiglu_method",       type=str,   default=SWIGLU_METHOD,
                        choices=["delta", "exact"],
                        help="SwiGLU VP: 'delta' = first-order linearisation; "
                             "'exact' = Hermite k=3 + product var + cross-cov")
    p.add_argument("--rms_norm_method",     type=str,   default=RMS_NORM_METHOD,
                   choices=["streamlined", "mvp"])
    p.add_argument("--attn_var_mode",       type=str,   default=ATTN_VAR_MODE,
                   choices=["value_only", "full"],
                   help="Attention VP: 'value_only' = only V-path uncertainty; "
                        "'full' = Q uncertainty via delta-method linearisation through softmax")
    p.add_argument("--n_epochs",             type=int,   default=N_EPOCHS)
    p.add_argument("--n_epochs_s",           type=int,   default=N_EPOCHS_S,
                   help="Epochs for phase-1 (log_s only); defaults to ~1/4 of --n_epochs")
    p.add_argument("--total_steps",          type=int,   default=None,
                   help="If set, overrides --n_epochs: n_epochs = ceil(total_steps / steps_per_epoch)")
    p.add_argument("--total_steps_s",       type=int,   default=None,
                   help="If set, overrides --n_epochs_s for phase-1 (log_s) the same way")
    p.add_argument("--calib_batch_size",    type=int,   default=CALIB_BATCH_SIZE)
    p.add_argument("--grad_accum",          type=int,   default=GRAD_ACCUM)
    p.add_argument("--lr",                  type=float, default=LR,
                   help="LR for log_s (max LR at last layer for per-block)")
    p.add_argument("--lr_t",               type=float, default=LR_T,
                   help="LR for log_T / log_T_logit (max LR at last layer)")
    p.add_argument("--lr_min_factor",      type=float, default=LR_MIN_FACTOR,
                   help="Min LR at layer 0 = lr_min_factor * max_lr; linearly ramped to max at last layer; set 1.0 for uniform LR")
    p.add_argument("--lr_logit_factor",   type=float, default=10.0,
                   help="LR multiplier for log_T_logit relative to lr_t; set 1.0 for uniform LR")
    p.add_argument("--weight_decay_t",    type=float, default=WEIGHT_DECAY_T,
                   help="Weight decay for per-layer log_T parameters (0 = off); log_T_logit is never decayed")
    p.add_argument("--init_log_s",          type=float, default=INIT_LOG_S)
    p.add_argument("--fixed_log_s",         type=float, default=0.0,
                   help="Fixed (non-learned) global log_s for per_layer_logit variant")
    p.add_argument("--load_calib_params",   type=str,   default=None,
                   help="Path to a calib_params_*.pt checkpoint; loaded into the model before any training")
    p.add_argument("--init_log_T",          type=float, default=None,
                   help="Scalar initialiser broadcast to all log_T_list[i] (applied after --load_calib_params)")
    p.add_argument("--init_log_T_logit",    type=float, default=None,
                   help="Scalar initialiser for log_T_logit (applied after --load_calib_params)")
    p.add_argument("--freeze_log_T",        action="store_true", default=False,
                   help="Freeze all log_T_list params (requires_grad=False); only log_T_logit is trained")
    p.add_argument("--suffix",              type=str,   default="",
                   help="Optional suffix appended to the output JSON base tag")
    p.add_argument("--results_dir",         type=str,   default=None,
                   help="If set, also writes results/{task}/{seed_label}/{swiglu}_{rms}[_{suffix}].json")
    p.add_argument("--merge_results_into", type=str,   default=None,
                   help="Filename inside results/{task}/{seed_label}/ to merge calibrated results into "
                        "(e.g. exact_mvp.json); adds 'val_calibrated'/'test_calibrated' keys in place of a new file")
    p.add_argument("--val_calib_key",      type=str,   default="val_calibrated",
                   help="Key written under --merge_results_into for the val split")
    p.add_argument("--test_calib_key",     type=str,   default="test_calibrated",
                   help="Key written under --merge_results_into for the test split")
    p.add_argument("--finetune_s",          action="store_true", default=FINETUNE_S)
    p.add_argument("--lr_s_finetune",       type=float, default=LR_S_FINETUNE)
    p.add_argument("--n_mc_calib",          type=int,   default=N_MC_CALIB)
    p.add_argument("--n_mc_eval",           type=int,   default=N_MC_EVAL)
    p.add_argument("--eval_on_test",        action="store_true", default=False,
                   help="Also evaluate on the test split after calibration")
    args = p.parse_args()

    peft_method = "lora_lmhead" if args.lm_head else "lora"
    if args.testing_set != "val":
        peft_method += args.testing_set
    seed_label = args.seed_label or str(args.seed)
    lr = 5e-5
    args.output_dir = (
        f"{args.output_dir}/{args.task_name}/"
        f"{args.model_name_or_path}_{peft_method}_{args.lora_alpha}_{args.lora_dropout}_{lr}_{seed_label}"
    )
    args.laplace_output_dir = (
        f"outputs_laplace/{args.task_name}/"
        f"{args.model_name_or_path}_{peft_method}_{args.lora_alpha}_{args.lora_dropout}_{lr}_{seed_label}/"
    )
    return args


# ---------------------------------------------------------------------------
# KFAC diagonal variance extraction
# ---------------------------------------------------------------------------

def kfac_diagonal_variance(posterior_precision) -> list[torch.Tensor]:
    """Per-element posterior variance from a KronDecomposed precision matrix.

        Var[W_{ij}] = (Q1**2 @ D @ Q2**2.T)_{ij},   D_{k,k'} = 1/(l1_k*l2_k' + delta)
    """
    variances = []
    for Qs, ls, delta in zip(posterior_precision.eigenvectors,
                              posterior_precision.eigenvalues,
                              posterior_precision.deltas):
        if len(ls) == 1:
            Q, l = Qs[0], ls[0]
            variances.append((Q ** 2) @ (1.0 / (l + delta)))
        elif len(ls) == 2:
            Q1, Q2 = Qs
            l1, l2 = ls
            D = 1.0 / (torch.ger(l1, l2) + delta)
            variances.append((Q1 ** 2) @ D @ (Q2 ** 2).T)
        else:
            raise ValueError(f"Unexpected Kronecker factors: {len(ls)}")
    return variances


def extract_kron_lora_factors(
    H_dict: dict,
    trainable_params: list,
    device,
    max_full_dim: int = 64,
) -> dict:
    """Extract small full-matrix Kronecker factor inverses for improved LoRA VP.

    Must be called AFTER extract_param_variances (which sets H.deltas).

    For each 2-factor KFAC layer returns a dict:
      G_inv      — full [n1, n1] output-factor inverse   (None if n1 > max_full_dim)
      G_inv_diag — [n1] diagonal of output-factor inverse
      A_inv      — full [n2, n2] input-factor inverse    (None if n2 > max_full_dim)
      A_inv_diag — [n2] diagonal of input-factor inverse

    For LoRA weights (r=8, d_in=4096):
      lora_A [r, d_in]:  G_inv is [r, r] (small ✓), A_inv is [d_in, d_in] (skipped)
      lora_B [d_out, r]: G_inv is [d_out, d_out] (skipped), A_inv is [r, r] (small ✓)
    """
    H = H_dict["H"]
    if isinstance(H, torch.Tensor):
        return {}

    result = {}
    for (name, _), Qs, ls, delta in zip(
        trainable_params, H.eigenvectors, H.eigenvalues, H.deltas
    ):
        if len(ls) != 2:
            result[name] = None
            continue

        n1, n2 = len(ls[0]), len(ls[1])

        def _diag_inv(which):
            Q, l = Qs[which], ls[which]
            return ((Q ** 2) @ (1.0 / (l + delta))).to(device).float()

        def _full_inv(which):
            Q, l = Qs[which], ls[which]
            inv_l = (1.0 / (l + delta)).to(Q.dtype)
            return ((Q * inv_l.unsqueeze(0)) @ Q.T).to(device).float()

        result[name] = {
            "G_inv":      _full_inv(0) if n1 <= max_full_dim else None,
            "G_inv_diag": _diag_inv(0),
            "A_inv":      _full_inv(1) if n2 <= max_full_dim else None,
            "A_inv_diag": _diag_inv(1),
        }

    return result


def extract_param_variances(
    H_dict: dict,
    prior_precision: torch.Tensor,
    trainable_params: list,
    device,
) -> dict:
    """Dispatch to the right variance extractor based on the saved Hessian type.

    Supports:
      kron — KronDecomposed: marginal variances via kfac_diagonal_variance
      diag — flat (n_params,) tensor: variance = 1 / (H_diag + prior_precision)
    """
    H = H_dict["H"]

    if isinstance(H, torch.Tensor):
        # Diagonal Hessian saved as a flat vector of length n_params
        precision_flat = H.to(device) + prior_precision.to(device)
        variance_flat  = 1.0 / precision_flat.clamp(min=1e-10)
        param_variances = {}
        offset = 0
        for name, param in trainable_params:
            n = param.numel()
            param_variances[name] = (
                variance_flat[offset : offset + n].view(param.shape).float().clamp(min=0)
            )
            offset += n
    else:
        # Kronecker-factored Hessian (KronDecomposed)
        H.deltas = prior_precision.expand(len(H.deltas)).clone()
        kfac_vars = kfac_diagonal_variance(H)
        if len(kfac_vars) != len(trainable_params):
            raise RuntimeError(
                f"KFAC layer count ({len(kfac_vars)}) != trainable param count "
                f"({len(trainable_params)}).  Check requires_grad settings or laplace_sub."
            )
        param_variances = {}
        for (name, _), var in zip(trainable_params, kfac_vars):
            param_variances[name] = var.to(device)

    return param_variances


# ---------------------------------------------------------------------------
# VP utility functions
# ---------------------------------------------------------------------------

def make_eager_causal_mask(
    B: int, S: int, attention_mask, device, dtype=torch.float32
) -> torch.Tensor:
    """Build a [B, 1, S, S] additive causal attention bias for eager attention.

    Returns 0 for positions a query may attend to and -inf for masked
    positions.  Incorporates both the causal constraint (no future tokens)
    and the padding mask (attention_mask=0 tokens are masked as keys).
    """
    # Causal upper-triangular mask: -inf above the diagonal
    mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)           # [S, S]
    mask = mask.unsqueeze(0).unsqueeze(0)         # [1, 1, S, S]
    mask = mask.expand(B, 1, S, S).clone()

    # Padding mask: mask keys (dim -1) where attention_mask = 0
    if attention_mask is not None:
        pad = (attention_mask == 0)               # [B, S], True = padding
        mask.masked_fill_(pad.unsqueeze(1).unsqueeze(2), float("-inf"))

    return mask


def vp_rms_norm_streamlined(x: ParamPair, norm: LlamaRMSNorm) -> ParamPair:
    m, v  = x.mean, x.var
    ms    = m.float().pow(2).mean(dim=-1, keepdim=True)
    scale = torch.rsqrt(ms + norm.variance_epsilon)  # float32
    # Match LlamaRMSNorm exactly: multiply in float32, cast THEN apply weight in input dtype.
    w = norm.weight.to(m.dtype)
    return ParamPair(
        (m.float() * scale).to(m.dtype) * w,
        (scale * scale * v.float() * w.float() ** 2).to(v.dtype),
    )


def vp_rms_norm_mvp(x: ParamPair, norm: LlamaRMSNorm) -> ParamPair:
    m, v   = x.mean, x.var
    ms     = m.float().pow(2).mean(dim=-1, keepdim=True)
    v_mean = v.float().mean(dim=-1, keepdim=True)
    scale  = torch.rsqrt((ms + v_mean).clamp(min=0) + norm.variance_epsilon)
    # Match LlamaRMSNorm exactly: multiply in float32, cast THEN apply weight in input dtype.
    w = norm.weight.to(m.dtype)
    return ParamPair(
        (m.float() * scale).to(m.dtype) * w,
        (scale * scale * v.float() * w.float() ** 2).to(v.dtype),
    )

def vp_linear(x: ParamPair, weight: torch.Tensor, bias=None) -> ParamPair:
    """Standard VP through a deterministic linear layer."""
    return ParamPair(
        F.linear(x.mean, weight, bias),
        x.var @ (weight ** 2).T,
    )


# ── SiLU ≈ GELU rescaling for Hermite variance expansion ──────────────────────
# σ(x) ≈ Φ(c·x), so SiLU(x) = x·σ(x) ≈ GELU(c·x)/c  where c = √(π/8).
# For gate ~ N(μ_g, var_g): c·gate ~ N(c·μ_g, c²·var_g).
# GELU Hermite moments of c·gate, divided by c (or c²), give SiLU moments.

_C_SILU_GELU  = math.sqrt(math.pi / 8.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _npdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) * _INV_SQRT_2PI


def _gelu_moments_k3(m: torch.Tensor, v: torch.Tensor):
    """(E[GELU(y)], Var[GELU(y)]) for y ~ N(m, v) via Hermite order-3 expansion."""
    zeta  = torch.rsqrt(1.0 + v)
    m_out = m * torch.special.ndtr(m * zeta) + v * zeta * _npdf(m * zeta)
    sigma = (v + 1e-6).sqrt()
    gamma = (m * zeta).clamp(-20.0, 20.0)
    alpha = sigma * zeta
    phig  = _npdf(gamma)
    v_out = (sigma * torch.special.ndtr(gamma) + alpha * (1.0 - alpha**2) * m * phig) ** 2
    pre   = (sigma * phig) ** 2
    herm  = [torch.special.hermite_polynomial_he(gamma, i) for i in range(4)]
    for i in range(2, 4):
        v_out += pre * (alpha**(i-1) * (herm[i-2] - (1.0 - alpha**2) * herm[i])) ** 2 / math.factorial(i)
    return m_out, v_out


def _vp_swiglu_delta(gate: ParamPair, up: ParamPair, _cov_gu) -> ParamPair:
    """First-order delta method (original implementation, no cross-covariance)."""
    silu_m  = F.silu(gate.mean)
    sg      = torch.sigmoid(gate.mean)
    df      = sg * (1.0 + gate.mean * (1.0 - sg))
    return ParamPair(
        silu_m * up.mean,
        (df * up.mean) ** 2 * gate.var + silu_m ** 2 * up.var,
    )


def _vp_swiglu_exact(gate: ParamPair, up: ParamPair, cov_gu: torch.Tensor) -> ParamPair:
    """SwiGLU VP: SiLU moments via GELU Hermite k=3 + exact product var + cross-cov.

    SiLU(gate) ≈ GELU(_C·gate)/_C gives better E[SiLU] and Var[SiLU].
    Exact product-variance: Var[SiLU·up] = Var[SiLU]·Var[up] + Var[SiLU]·μ_up²
                                          + E[SiLU]²·Var[up] + 2·Cov[SiLU,up]·E[SiLU]·μ_up.
    Cross-cov (first-order Stein): Cov[SiLU(gate_j), up_j] ≈ f'(μ_g_j)·cov_gu_j.
    """
    c      = _C_SILU_GELU
    gm, gv = _gelu_moments_k3(c * gate.mean, c**2 * gate.var)
    silu_m = gm / c
    silu_v = gv / c**2
    sg     = torch.sigmoid(gate.mean)
    df     = sg * (1.0 + gate.mean * (1.0 - sg))
    mean   = silu_m * up.mean + df * cov_gu
    var    = (silu_v * up.var
              + silu_v * up.mean**2
              + silu_m**2 * up.var
              + 2.0 * df * cov_gu * silu_m * up.mean)
    return ParamPair(mean, var.clamp(min=0))


# ---------------------------------------------------------------------------
# VP modules for Llama-2 transformer layers
# ---------------------------------------------------------------------------

class VPLlamaAttention(nn.Module):
    """VP through LlamaAttention.

    attn_var_mode controls Q-side uncertainty:
    - "value_only": Q/K pattern deterministic; only V uncertainty propagates via
                    Var[out]_t = attn_w_t² ⊗ v_var_t.
    - "full": Q uncertainty propagates via delta-method linearisation through softmax.
              score_var[i,j] = scaling² * q_var[i] · k[j]²  (RoPE ~ variance-preserving)
              Var[p_i] = p_i² * [score_var_i*(1-2*p_i) + Σ_j p_j²*score_var_j]
              out_var  += attn_var @ v_mean²  (additive to V-path term)
    """

    def __init__(
        self,
        k_weight: torch.Tensor,
        o_weight: torch.Tensor,
        q_mvp: LoRALinearMVP,
        v_mvp: LoRALinearMVP,
        n_heads: int,
        head_dim: int,
        scaling: float,
        attn_var_mode: str = "value_only",
    ):
        super().__init__()
        self.register_buffer("k_weight", k_weight)
        self.register_buffer("o_weight", o_weight)
        # q_mvp kept as submodule; mean computed two-step in forward to match PEFT:
        #   q = W_base @ x + scaling * B @ (A @ x)
        # Precomputing W_q_eff = W_base + s*B@A in bfloat16 accumulates ~O(24) error
        # (7 mantissa bits × r=8 outer-product terms over hidden_dim=4096).
        self.q_mvp = q_mvp
        self.v_mvp = v_mvp
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scaling  = scaling
        self.attn_var_mode = attn_var_mode

    def forward(self, x: ParamPair, attn_mask, position_embeddings, var_scale=1.0) -> ParamPair:
        B, S, H = x.mean.shape
        cos, sin = position_embeddings

        with torch.no_grad():
            # Q and K don't depend on var_scale — compute under no_grad to save memory
            # Two-step Q matching PEFT: W_base @ x + scaling * B @ (A @ x)
            q = (F.linear(x.mean, self.q_mvp.base_weight)
                 + self.q_mvp.scaling * F.linear(F.linear(x.mean, self.q_mvp.lora_A_mean), self.q_mvp.lora_B_mean)
                 ).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            k = F.linear(x.mean, self.k_weight).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k_sq = k ** 2
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
            if attn_mask is not None:
                scores = scores + attn_mask
            attn_w = F.softmax(scores.float(), dim=-1).to(x.mean.dtype)  # [B, n, S, S]
            attn_w = torch.nan_to_num(attn_w, nan=0.0)  # pad queries: all-masked → 0 not NaN

        # V: uncertain — gradients flow through var_scale here
        v_pair = self.v_mvp(x, var_scale=var_scale)
        v_mean = v_pair.mean.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v_var  = v_pair.var.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        out_var = torch.matmul(attn_w ** 2, v_var)  # V-path variance [B, n, S, d]

        if self.attn_var_mode == "full":
            # Q variance (gradient flows through var_scale via q_mvp)
            q_var = self.q_mvp(x, var_scale=var_scale).var.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            # score_var[i,j] = scaling² * Σ_d q_var[i,d] * k[j,d]²
            # RoPE is an orthogonal transform; for symmetric LoRA posteriors it is
            # approximately variance-preserving on the diagonal, so we use pre-RoPE q_var.
            score_var = self.scaling ** 2 * torch.matmul(q_var, k_sq.transpose(-2, -1))
            # Delta method: linearise softmax Jacobian J_ij = p_i(δ_ij - p_j)
            # Var[p_i] = p_i² * [score_var_i*(1 - 2*p_i) + Σ_j p_j²*score_var_j]
            p = attn_w
            S_w = (p ** 2 * score_var).sum(-1, keepdim=True)  # [B, n, S_q, 1]
            attn_var = (p ** 2 * (score_var * (1 - 2 * p) + S_w)).clamp(min=0)
            # Attention-weight uncertainty contributes Var[Σ_s p_s*v_s] ≈ Σ_s attn_var_s * v_mean_s²
            out_var = out_var + torch.matmul(attn_var, v_mean ** 2)

        out_mean = torch.matmul(attn_w, v_mean).transpose(1, 2).reshape(B, S, H)
        out_var  = out_var.transpose(1, 2).reshape(B, S, H)
        return vp_linear(ParamPair(out_mean, out_var), self.o_weight)


class VPLlamaMLP(nn.Module):
    """VP through LlamaMLP (SwiGLU) — all projections deterministic (no LoRA).

    Uncertain input propagates through:  gate/up (linear VP) → SwiGLU → down (linear VP).
    swiglu_method: "delta" = first-order linearisation; "exact" = Hermite k=3 + product var + cross-cov.
    """

    def __init__(self, gate_weight: torch.Tensor, up_weight: torch.Tensor,
                 down_weight: torch.Tensor, swiglu_method: str = "exact"):
        super().__init__()
        self.register_buffer("gate_weight", gate_weight)
        self.register_buffer("up_weight",   up_weight)
        self.register_buffer("down_weight", down_weight)
        self._swiglu_fn = _vp_swiglu_exact if swiglu_method == "exact" else _vp_swiglu_delta

    def forward(self, x: ParamPair) -> ParamPair:
        gate   = vp_linear(x, self.gate_weight)
        up     = vp_linear(x, self.up_weight)
        cov_gu = x.var @ (self.gate_weight * self.up_weight).T
        return vp_linear(self._swiglu_fn(gate, up, cov_gu), self.down_weight)


class VPLlamaDecoderLayer(nn.Module):
    """VP through one LlamaDecoderLayer, exposing separate attn and MLP residuals."""

    def __init__(self, input_norm, post_attn_norm, vp_attn, vp_mlp, rms_norm_fn):
        super().__init__()
        self.input_norm     = input_norm
        self.post_attn_norm = post_attn_norm
        self.vp_attn        = vp_attn
        self.vp_mlp         = vp_mlp
        self.rms_norm_fn    = rms_norm_fn

    def forward_attn_residual(self, x: ParamPair, attn_mask, position_embeddings, var_scale) -> ParamPair:
        """input_norm → attention → residual add."""
        h = self.rms_norm_fn(x, self.input_norm)
        h = self.vp_attn(h, attn_mask, position_embeddings, var_scale=var_scale)
        return x + h

    def forward_mlp_residual(self, x: ParamPair) -> ParamPair:
        """post_attn_norm → MLP → residual add."""
        h = self.rms_norm_fn(x, self.post_attn_norm)
        h = self.vp_mlp(h)
        return x + h

    def forward(self, x: ParamPair, attn_mask, position_embeddings, var_scale=1.0) -> ParamPair:
        x = self.forward_attn_residual(x, attn_mask, position_embeddings, var_scale)
        x = self.forward_mlp_residual(x)
        return x


# ---------------------------------------------------------------------------
# Grad-checkpoint wrappers (module-level so they are picklable)
# ---------------------------------------------------------------------------

def _vp_attn_fwd(layer, attn_mask, cos, sin, mean, var, var_scale):
    out = layer.forward_attn_residual(ParamPair(mean, var), attn_mask, (cos, sin), var_scale)
    return out.mean, out.var


def _vp_mlp_fwd(layer, mean, var):
    out = layer.forward_mlp_residual(ParamPair(mean, var))
    return out.mean, out.var


# ---------------------------------------------------------------------------
# VP Backbone
# ---------------------------------------------------------------------------

class VPBackbone(nn.Module):
    """Full variance-propagating Llama backbone.

    Propagates a ParamPair through all transformer layers using KFAC-sourced
    LoRA posteriors for q_proj and v_proj.  Optional per-layer/sub-block
    temperatures T_attn and T_mlp scale the propagated variance after each
    sub-block's residual addition, enabling post-hoc VP calibration.
    """

    def __init__(self, llama_model, vp_decoder_layers: list, rms_norm_fn):
        super().__init__()
        self.llama      = llama_model
        self.vp_layers  = nn.ModuleList(vp_decoder_layers)
        self.final_norm = llama_model.norm
        self.rms_norm_fn = rms_norm_fn

    def forward(
        self,
        input_ids,
        attention_mask,
        var_scale=1.0,
        T_attn: torch.Tensor | None = None,   # [N_layers] or None
        T_mlp:  torch.Tensor | None = None,   # [N_layers] or None
    ) -> ParamPair:
        inputs_embeds = self.llama.embed_tokens(input_ids)
        B, S, _ = inputs_embeds.shape

        position_ids  = torch.arange(S, device=inputs_embeds.device).unsqueeze(0)
        causal_mask   = make_eager_causal_mask(B, S, attention_mask,
                                               device=inputs_embeds.device,
                                               dtype=inputs_embeds.dtype)
        cos, sin = self.llama.rotary_emb(inputs_embeds, position_ids=position_ids)

        x = ParamPair(inputs_embeds, torch.zeros_like(inputs_embeds))

        _vs_per_layer = (isinstance(var_scale, torch.Tensor) and var_scale.numel() > 1)

        for i, vp_layer in enumerate(self.vp_layers):
            vs_i = var_scale[i] if _vs_per_layer else var_scale
            # Attention sub-block
            if USE_GRAD_CHECKPOINT:
                m, v = grad_checkpoint(
                    _vp_attn_fwd, vp_layer, causal_mask, cos, sin,
                    x.mean, x.var, vs_i, use_reentrant=False,
                )
            else:
                out = vp_layer.forward_attn_residual(x, causal_mask, (cos, sin), vs_i)
                m, v = out.mean, out.var
            t_a = T_attn[i] if T_attn is not None else 1.0
            x = ParamPair(m, v * t_a)

            # MLP sub-block
            if USE_GRAD_CHECKPOINT:
                m, v = grad_checkpoint(_vp_mlp_fwd, vp_layer, x.mean, x.var, use_reentrant=False)
            else:
                out = vp_layer.forward_mlp_residual(x)
                m, v = out.mean, out.var
            t_m = T_mlp[i] if T_mlp is not None else 1.0
            x = ParamPair(m, v * t_m)

        return self.rms_norm_fn(x, self.final_norm)


def _get_weight_f32(module) -> torch.Tensor:
    return module.weight.float().detach()


def build_vp_backbone(backbone, param_variances: dict,
                      rms_norm_method: str = "mvp",
                      swiglu_method: str = "exact",
                      attn_var_mode: str = "value_only",
                      kron_factors: dict | None = None) -> VPBackbone:
    """Construct VPBackbone from LlamaModel + KFAC per-element variances.

    Layers whose LoRA variances are absent in param_variances get zero-variance
    buffers (reduces to MAP for that projection).

    If kron_factors is provided (output of extract_kron_lora_factors), each
    LoRALinearMVP uses the improved Kronecker VP instead of the diagonal approx.
    """
    rms_norm_fn = vp_rms_norm_mvp if rms_norm_method == "mvp" else vp_rms_norm_streamlined
    cfg      = backbone.config
    n_heads  = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    scaling  = 1.0 / math.sqrt(head_dim)

    def find_var(pattern: str):
        for name, var in param_variances.items():
            if pattern in name:
                return var.float().clamp(min=0)
        return None

    def find_kron(pattern: str) -> dict | None:
        if kron_factors is None:
            return None
        for name, kf in kron_factors.items():
            if pattern in name and kf is not None:
                return kf
        return None

    def make_mvp(lora_layer, layer_idx, proj_name):
        pfx   = f"layers.{layer_idx}.self_attn.{proj_name}"
        A_mean = lora_layer.lora_A["default"].weight.float().detach()
        B_mean = lora_layer.lora_B["default"].weight.float().detach()
        _av = find_var(f"{pfx}.lora_A"); A_var = _av if _av is not None else torch.zeros(A_mean.shape, dtype=torch.float32, device=A_mean.device)
        _bv = find_var(f"{pfx}.lora_B"); B_var = _bv if _bv is not None else torch.zeros(B_mean.shape, dtype=torch.float32, device=B_mean.device)

        kf_A = find_kron(f"{pfx}.lora_A")
        kf_B = find_kron(f"{pfx}.lora_B")

        return LoRALinearMVP(
            base_weight=_get_weight_f32(lora_layer.base_layer),
            lora_A_mean=A_mean, lora_B_mean=B_mean,
            lora_A_var=A_var,   lora_B_var=B_var,
            scaling=lora_layer.scaling["default"],
            # Kron factors: G_inv [r,r] for A output side, A_inv [r,r] for B input side
            G_A_inv          = kf_A["G_inv"]      if kf_A else None,
            lora_A_A_inv_diag= kf_A["A_inv_diag"] if kf_A else None,
            lora_B_G_inv_diag= kf_B["G_inv_diag"] if kf_B else None,
            A_B_inv          = kf_B["A_inv"]      if kf_B else None,
        )

    vp_decoder_layers = []
    for i, layer in enumerate(backbone.layers):
        vp_attn = VPLlamaAttention(
            k_weight=_get_weight_f32(layer.self_attn.k_proj),
            o_weight=_get_weight_f32(layer.self_attn.o_proj),
            q_mvp=make_mvp(layer.self_attn.q_proj, i, "q_proj"),
            v_mvp=make_mvp(layer.self_attn.v_proj, i, "v_proj"),
            n_heads=n_heads, head_dim=head_dim, scaling=scaling,
            attn_var_mode=attn_var_mode,
        )
        vp_mlp = VPLlamaMLP(
            gate_weight=_get_weight_f32(layer.mlp.gate_proj),
            up_weight=_get_weight_f32(layer.mlp.up_proj),
            down_weight=_get_weight_f32(layer.mlp.down_proj),
            swiglu_method=swiglu_method,
        )
        vp_decoder_layers.append(VPLlamaDecoderLayer(
            input_norm=layer.input_layernorm,
            post_attn_norm=layer.post_attention_layernorm,
            vp_attn=vp_attn, vp_mlp=vp_mlp, rms_norm_fn=rms_norm_fn,
        ))

    return VPBackbone(backbone, vp_decoder_layers, rms_norm_fn)


# ---------------------------------------------------------------------------
# VP LM head (no own learnable params — var_scale passed in)
# ---------------------------------------------------------------------------

class VPLMHead(nn.Module):
    """LoRALinearMVP LM head.  var_scale is owned by the calibration model."""

    def __init__(self, lm_head_lora, lora_A_var: torch.Tensor, lora_B_var: torch.Tensor):
        super().__init__()
        self.mvp = LoRALinearMVP(
            base_weight=lm_head_lora.linear.weight.float().detach().clone(),
            lora_A_mean=lm_head_lora.lora_A.weight.float().detach().clone(),
            lora_B_mean=lm_head_lora.lora_B.weight.float().detach().clone(),
            lora_A_var=lora_A_var.float().clamp(min=0),
            lora_B_var=lora_B_var.float().clamp(min=0),
            scaling=lm_head_lora.scaling,
        )

    def forward(self, hidden, var_scale=1.0) -> ParamPair:
        x = hidden if isinstance(hidden, ParamPair) else ParamPair(hidden, torch.zeros_like(hidden))
        return self.mvp(x, var_scale=var_scale)


# ---------------------------------------------------------------------------
# Calibration model classes
# ---------------------------------------------------------------------------

class VPCalibLogSOnly(nn.Module):
    """Full backbone VP with no temperature scaling.  Only log_s is learned.
    Phase-1 role: identify the global posterior scale across all LoRA layers.
    Learnable: log_s (1 param).
    """

    def __init__(self, vp_backbone, vp_head: VPLMHead):
        super().__init__()
        self.vp_backbone = vp_backbone
        self.vp_head     = vp_head
        self.log_s       = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask) -> ParamPair:
        vs = self.log_s.exp()
        hidden = self.vp_backbone(input_ids, attention_mask, var_scale=vs,
                                   T_attn=None, T_mlp=None)
        h = ParamPair(hidden.mean[:, -1, :], hidden.var[:, -1, :])
        return self.vp_head(h, var_scale=vs)

    @property
    def n_calib_params(self): return 1


class VPCalibLogSPerBlock(nn.Module):
    """Phase-1 variant: per-layer LoRA posterior scale (one log_s per transformer block).
    Learnable: log_s_list[N] — ParameterList so each layer can have its own Adam state / LR.
    """

    def __init__(self, vp_backbone: VPBackbone, vp_head: VPLMHead):
        super().__init__()
        self.vp_backbone  = vp_backbone
        self.vp_head      = vp_head
        N = len(vp_backbone.vp_layers)
        self.log_s_list = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(N)])

    @property
    def log_s(self) -> torch.Tensor:
        return torch.cat([p for p in self.log_s_list])  # [N]

    def forward(self, input_ids, attention_mask) -> ParamPair:
        vs = self.log_s.exp()  # [N]
        hidden = self.vp_backbone(input_ids, attention_mask, var_scale=vs,
                                   T_attn=None, T_mlp=None)
        h = ParamPair(hidden.mean[:, -1, :], hidden.var[:, -1, :])
        return self.vp_head(h, var_scale=vs.mean())

    @property
    def n_calib_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VPCalibGlobal(nn.Module):
    """Full backbone VP + single global activation temperature.
    Learnable: log_s (1) + log_T (1).
    """

    def __init__(self, vp_backbone: VPBackbone, vp_head: VPLMHead):
        super().__init__()
        self.vp_backbone = vp_backbone
        self.vp_head     = vp_head
        self.log_s       = nn.Parameter(torch.zeros(1))
        self.log_T       = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask) -> ParamPair:
        vs = self.log_s.exp()
        T  = self.log_T.exp()
        N  = len(self.vp_backbone.vp_layers)
        T_vec = T.expand(N)
        hidden = self.vp_backbone(input_ids, attention_mask, var_scale=vs,
                                   T_attn=T_vec, T_mlp=T_vec)
        h = ParamPair(hidden.mean[:, -1, :], hidden.var[:, -1, :])
        return self.vp_head(h, var_scale=vs)

    @property
    def n_calib_params(self): return 2


class VPCalibPerLayer(nn.Module):
    """Full backbone VP + one activation temperature per layer.
    Temperature applied to the full layer output (after MLP residual).
    Learnable: log_s (1) + log_T[N] (N params).
    """

    def __init__(self, vp_backbone: VPBackbone, vp_head: VPLMHead):
        super().__init__()
        self.vp_backbone = vp_backbone
        self.vp_head     = vp_head
        N = len(vp_backbone.vp_layers)
        self.log_s = nn.Parameter(torch.zeros(1))
        self.log_T = nn.Parameter(torch.zeros(N))

    def forward(self, input_ids, attention_mask) -> ParamPair:
        vs = self.log_s.exp()
        T  = self.log_T.exp()  # [N]
        # T applied only after MLP (= full layer output); attn runs unscaled
        hidden = self.vp_backbone(input_ids, attention_mask, var_scale=vs,
                                   T_attn=None, T_mlp=T)
        h = ParamPair(hidden.mean[:, -1, :], hidden.var[:, -1, :])
        return self.vp_head(h, var_scale=vs)

    @property
    def n_calib_params(self): return 1 + len(self.vp_backbone.vp_layers)


class VPCalibPerSubBlock(nn.Module):
    """Full backbone VP + per-sub-block temperatures (attn, MLP per layer).
    Learnable: log_s (1) + log_T[N, 2] (2N params).
    """

    def __init__(self, vp_backbone: VPBackbone, vp_head: VPLMHead):
        super().__init__()
        self.vp_backbone = vp_backbone
        self.vp_head     = vp_head
        N = len(vp_backbone.vp_layers)
        self.log_s = nn.Parameter(torch.zeros(1))
        self.log_T = nn.Parameter(torch.zeros(N, 2))   # [:, 0]=attn, [:, 1]=MLP

    def forward(self, input_ids, attention_mask) -> ParamPair:
        vs = self.log_s.exp()
        T  = self.log_T.exp()  # [N, 2]
        hidden = self.vp_backbone(input_ids, attention_mask, var_scale=vs,
                                   T_attn=T[:, 0], T_mlp=T[:, 1])
        h = ParamPair(hidden.mean[:, -1, :], hidden.var[:, -1, :])
        return self.vp_head(h, var_scale=vs)

    @property
    def n_calib_params(self): return 1 + 2 * len(self.vp_backbone.vp_layers)


class VPCalibPerSubBlockLogit(nn.Module):
    """Phase-2: shared attn/MLP temperatures + logit-level scale.

    Shared temperatures (not per-layer) distinguish the attn and MLP
    approximation error structure without per-layer proliferation.

    Learnable in phase-2: log_T_attn (1), log_T_mlp (1), log_T_logit (1).
    log_s frozen from phase-1 (scalar or [N] tensor).
    Total active params: 3.
    """

    def __init__(self, vp_backbone: VPBackbone, vp_head: VPLMHead):
        super().__init__()
        self.vp_backbone = vp_backbone
        self.vp_head     = vp_head
        self.log_s       = nn.Parameter(torch.zeros(1))
        self.log_T_attn  = nn.Parameter(torch.zeros(1))
        self.log_T_mlp   = nn.Parameter(torch.zeros(1))
        self.log_T_logit = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask) -> ParamPair:
        vs = self.log_s.exp()
        N  = len(self.vp_backbone.vp_layers)
        T_attn = self.log_T_attn.exp().expand(N)
        T_mlp  = self.log_T_mlp.exp().expand(N)
        hidden = self.vp_backbone(input_ids, attention_mask, var_scale=vs,
                                   T_attn=T_attn, T_mlp=T_mlp)
        h = ParamPair(hidden.mean[:, -1, :], hidden.var[:, -1, :])
        vs_head = vs.mean() if vs.numel() > 1 else vs
        logits = self.vp_head(h, var_scale=vs_head)
        return ParamPair(logits.mean, logits.var * self.log_T_logit.exp())

    @property
    def n_calib_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VPCalibPerLayerLogit(nn.Module):
    """Per-block T[N] + logit scale. log_s fixed (not learned).

    One temperature per transformer block applied after the full layer output
    (MLP residual). ParameterList gives each block its own Adam state so the
    linear LR ramp works correctly — later blocks (which empirically need higher
    T) get a proportionally larger LR.

    log_s is a fixed buffer (exp(FIXED_LOG_S) ≈ 0.018); no phase-1 needed.
    Learnable: log_T_list[N] + log_T_logit  (N+1 params total).
    """

    def __init__(self, vp_backbone: VPBackbone, vp_head: VPLMHead,
                 fixed_log_s: float = FIXED_LOG_S):
        super().__init__()
        self.vp_backbone = vp_backbone
        self.vp_head     = vp_head
        N = len(vp_backbone.vp_layers)
        self.register_buffer("log_s", torch.tensor(fixed_log_s))
        self.log_T_list  = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(N)])
        self.log_T_logit = nn.Parameter(torch.zeros(1))

    @property
    def log_T(self) -> torch.Tensor:
        return torch.cat([p for p in self.log_T_list])  # [N]

    def forward(self, input_ids, attention_mask) -> ParamPair:
        vs = self.log_s.exp()   # fixed scalar
        T  = self.log_T.exp()   # [N]
        hidden = self.vp_backbone(input_ids, attention_mask, var_scale=vs,
                                   T_attn=None, T_mlp=T)
        h = ParamPair(hidden.mean[:, -1, :], hidden.var[:, -1, :])
        logits = self.vp_head(h, var_scale=vs)
        return ParamPair(logits.mean, logits.var * self.log_T_logit.exp())

    @property
    def n_calib_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VPCalibPerBlockST(nn.Module):
    """Single-phase: per-block LoRA posterior scale + per-block activation temperature + logit scale.

    Both log_s and log_T have one parameter per transformer block, jointly calibrated
    in a single phase.  ParameterLists give each block its own Adam state so the
    linear LR ramp works correctly — later blocks get proportionally larger LRs.

    log_s initialized to FIXED_LOG_S; log_T and log_T_logit initialized to 0.
    Learnable: log_s_list[N] + log_T_list[N] + log_T_logit  (2N+1 params total).
    """

    def __init__(self, vp_backbone: VPBackbone, vp_head: VPLMHead):
        super().__init__()
        self.vp_backbone = vp_backbone
        self.vp_head     = vp_head
        N = len(vp_backbone.vp_layers)
        self.log_s_list  = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(N)])
        self.log_T_list  = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(N)])
        self.log_T_logit = nn.Parameter(torch.zeros(1))

    @property
    def log_s(self) -> torch.Tensor:
        return torch.cat([p for p in self.log_s_list])  # [N]

    @property
    def log_T(self) -> torch.Tensor:
        return torch.cat([p for p in self.log_T_list])  # [N]

    def forward(self, input_ids, attention_mask) -> ParamPair:
        vs = self.log_s.exp()   # [N]
        T  = self.log_T.exp()   # [N]
        hidden = self.vp_backbone(input_ids, attention_mask, var_scale=vs,
                                   T_attn=None, T_mlp=T)
        h = ParamPair(hidden.mean[:, -1, :], hidden.var[:, -1, :])
        logits = self.vp_head(h, var_scale=vs.mean())
        return ParamPair(logits.mean, logits.var * self.log_T_logit.exp())

    @property
    def n_calib_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss and evaluation
# ---------------------------------------------------------------------------

def mc_probs(logit_pair: ParamPair, n_samples: int) -> torch.Tensor:
    """MC estimate of E[softmax(z)], z ~ N(mu, diag(sigma²))."""
    # Use float32 for sigma: bfloat16 overflow → inf, then inf*0=NaN when eps≈0
    sigma = logit_pair.var.float().clamp(min=0).sqrt()
    eps   = torch.randn(n_samples, *logit_pair.mean.shape,
                        device=logit_pair.mean.device, dtype=torch.float32)
    return F.softmax(logit_pair.mean.float().unsqueeze(0) + sigma.unsqueeze(0) * eps, dim=-1).mean(0)


def nll_batch(model: nn.Module, batch: dict, device, n_mc: int) -> torch.Tensor:
    """MC NLL loss for one calibration batch."""
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    labels    = batch["labels"].to(device)
    probs     = mc_probs(model(input_ids, attn_mask), n_mc)
    return F.nll_loss(probs.log().clamp(min=-100), labels)


@torch.no_grad()
def evaluate_vp(model: nn.Module, dataloader, device, n_mc: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Run MC evaluation over a dataloader."""
    all_probs, all_labels = [], []
    for batch in tqdm(dataloader, desc="eval", leave=False):
        probs  = mc_probs(model(batch["input_ids"].to(device),
                                batch["attention_mask"].to(device)), n_mc)
        all_probs.append(probs.cpu())
        all_labels.append(batch["labels"].cpu())

    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds  = probs.argmax(dim=-1)
    acc    = (preds == labels).float().mean().item()
    nll    = -torch.log(probs.gather(1, labels.unsqueeze(1)).clamp(min=1e-12)).mean().item()
    return {"accuracy": acc, "nll": nll}, probs, labels


# ---------------------------------------------------------------------------
# Calibration loop
def run_variant(
    model: nn.Module,
    calib_loader,
    val_loader,
    eval_loader,
    device,
    args,
    variant_name: str,
    accelerator,
    lr_s: float | None = None,
    n_epochs: int | None = None,
    test_loader=None,
) -> dict:
    """Calibrate one variant and return its results.

    lr_s controls the log_s learning rate for this variant.  Pass None to use
    args.lr (the default for log_s_only).  For backbone variants called after
    log_s_only, main sets lr_s=args.lr_s_finetune when FINETUNE_S=True, or
    freezes log_s (requires_grad=False) and passes lr_s=None when False.
    """
    n_epochs = n_epochs if n_epochs is not None else args.n_epochs
    accelerator.print(f"\n--- Variant: {variant_name} ({model.n_calib_params} params, {n_epochs} epochs) ---")

    if variant_name in ("log_s_only", "log_s_per_block", "per_block_st"):
        if hasattr(model, "log_s_list"):
            for p in model.log_s_list:
                p.data.fill_(-4.0)
        else:
            model.log_s.data.fill_(-4.0)
        accelerator.print(f"  initialising log_s = -4.0 (shape={list(model.log_s.shape)})")

    lr_s = lr_s if lr_s is not None else args.lr

    if getattr(args, "freeze_log_T", False) and hasattr(model, "log_T_list"):
        for p in model.log_T_list:
            p.requires_grad_(False)
        accelerator.print("  freeze_log_T: log_T_list frozen at 0.0; only log_T_logit trained")

    # Build param groups with linearly-increasing LR across layers.
    # Layer 0 gets lr * lr_min_factor; last layer gets lr (max).
    def _layer_lr(max_lr: float, layer_idx: int, n_layers: int) -> float:
        if n_layers <= 1:
            return max_lr
        t = layer_idx / (n_layers - 1)
        return max_lr * (args.lr_min_factor + (1.0 - args.lr_min_factor) * t)

    param_groups = []

    # log_s: either a single Parameter or a ParameterList (per-block)
    if hasattr(model, "log_s_list"):
        N = len(model.log_s_list)
        for i, p in enumerate(model.log_s_list):
            if p.requires_grad:
                param_groups.append({"params": [p], "lr": _layer_lr(lr_s, i, N)})
    elif hasattr(model, "log_s") and isinstance(model.log_s, nn.Parameter) and model.log_s.requires_grad:
        param_groups.append({"params": [model.log_s], "lr": lr_s})

    # log_T_list: ParameterList, one per block — linear LR ramp
    if hasattr(model, "log_T_list"):
        N = len(model.log_T_list)
        for i, p in enumerate(model.log_T_list):
            if p.requires_grad:
                param_groups.append({"params": [p], "lr": _layer_lr(args.lr_t, i, N)})

    # log_T_attn / log_T_mlp: shared scalars (simplified variant)
    for attr in ("log_T_attn", "log_T_mlp"):
        p = getattr(model, attr, None)
        if isinstance(p, nn.Parameter) and p.requires_grad:
            param_groups.append({"params": [p], "lr": args.lr_t})

    # log_T: single Parameter (older per-layer / per-sub-block variants)
    if hasattr(model, "log_T") and isinstance(model.log_T, nn.Parameter) and model.log_T.requires_grad:
        param_groups.append({"params": [model.log_T], "lr": args.lr_t})

    # log_T_logit: final scale
    if hasattr(model, "log_T_logit") and model.log_T_logit.requires_grad:
        param_groups.append({"params": [model.log_T_logit], "lr": args.lr_t * args.lr_logit_factor})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay_t)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=0.0,
    )

    if n_epochs > 0:
        model.eval()
        with torch.no_grad():
            _init_nll_sum, _init_nll_steps = 0.0, 0
            for batch in calib_loader:
                _init_nll_sum   += nll_batch(model, batch, device, args.n_mc_calib).item()
                _init_nll_steps += 1
        accelerator.print(f"  init NLL (before training) = {_init_nll_sum / _init_nll_steps:.4f}")

    model.train()
    best_state, best_nll = None, float("inf")
    t_start = time.perf_counter()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        epoch_loss_sum, epoch_steps = 0.0, 0

        n_batches = len(calib_loader)
        log_window_sum, log_window_steps = 0.0, 0
        for batch_idx, batch in enumerate(calib_loader):
            last_batch   = (batch_idx == n_batches - 1)
            window_start = (batch_idx // args.grad_accum) * args.grad_accum
            window_size  = min(args.grad_accum, n_batches - window_start)

            loss = nll_batch(model, batch, device, args.n_mc_calib)
            (loss / window_size).backward()
            epoch_loss_sum += loss.item()
            epoch_steps += 1
            log_window_sum += loss.item()
            log_window_steps += 1

            if (batch_idx + 1) % args.grad_accum == 0 or last_batch:
                all_params = [p for g in optimizer.param_groups for p in g["params"]]
                # Replace NaN/Inf gradients with zero before clipping
                for p in all_params:
                    if p.grad is not None:
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % 10 == 0 or last_batch:
                _ls    = model.log_s.detach()
                _s_str = (f"{_ls.mean().item():.4f}(mean)" if _ls.numel() > 1
                          else f"{_ls.item():.4f}")
                accelerator.print(
                    f"    batch {batch_idx+1:4d}/{n_batches}  "
                    f"nll(last 10)={log_window_sum / log_window_steps:.4f}  "
                    f"log_s={_s_str}"
                )
                if hasattr(model, "log_T_list"):
                    _lt = model.log_T.detach()
                    accelerator.print(
                        f"      log_T mean={_lt.mean():.4f}  min={_lt.min():.4f}  max={_lt.max():.4f}"
                    )
                elif hasattr(model, "log_T_attn"):
                    accelerator.print(
                        f"      log_T_attn={model.log_T_attn.item():.4f}  "
                        f"log_T_mlp={model.log_T_mlp.item():.4f}"
                    )
                elif hasattr(model, "log_T") and model.log_T.numel() > 0:
                    _lt = model.log_T.detach().reshape(-1, 2)
                    accelerator.print(
                        f"      log_T mean=[{_lt[:,0].mean():.3f}, {_lt[:,1].mean():.3f}] (attn, mlp)"
                    )
                if hasattr(model, "log_T_logit"):
                    accelerator.print(
                        f"      log_T_logit={model.log_T_logit.item():.4f}"
                    )
                log_window_sum, log_window_steps = 0.0, 0

        epoch_nll = epoch_loss_sum / epoch_steps
        scheduler.step()
        if epoch_nll < best_nll:
            best_nll   = epoch_nll
            best_state = {n: p.detach().cpu().clone()
                          for n, p in model.named_parameters() if p.requires_grad}

        elapsed  = time.perf_counter() - t_start
        secs_per_epoch = elapsed / (epoch + 1)
        eta      = secs_per_epoch * (n_epochs - epoch - 1)

        def _fmt(secs):
            m, s = divmod(int(secs), 60)
            return f"{m}m{s:02d}s"

        _ls_now = model.log_s.detach()
        s_val   = (_ls_now.mean().item() if _ls_now.numel() > 1 else _ls_now.item())
        lr_s_val = optimizer.param_groups[0]["lr"]
        lr_t_val = optimizer.param_groups[-1]["lr"]  # last group = last layer / log_T_logit (max LR)
        accelerator.print(
            f"  epoch {epoch+1:3d}/{n_epochs}  "
            f"nll={epoch_nll:.4f}  log_s={s_val:.4f}  "
            f"lr_s={lr_s_val:.2e}  lr_t_max={lr_t_val:.2e}  elapsed={_fmt(elapsed)}  eta={_fmt(eta)}"
        )
        if hasattr(model, "log_T_list"):
            _lt = model.log_T.detach()
            accelerator.print(
                f"  log_T mean={_lt.mean():.4f}  min={_lt.min():.4f}  max={_lt.max():.4f}"
            )
            accelerator.print(
                "  log_T=[" + ", ".join(f"{v:.3f}" for v in _lt.tolist()) + "]"
            )
        elif hasattr(model, "log_T_attn"):
            accelerator.print(
                f"  log_T_attn={model.log_T_attn.item():.4f}  "
                f"log_T_mlp={model.log_T_mlp.item():.4f}"
            )
        elif hasattr(model, "log_T") and model.log_T.numel() > 0:
            _lt = model.log_T.detach().reshape(-1, 2)
            accelerator.print(
                f"  log_T mean=[{_lt[:,0].mean():.4f}, {_lt[:,1].mean():.4f}] (attn, mlp)"
            )
        if hasattr(model, "log_T_logit"):
            accelerator.print(f"  log_T_logit={model.log_T_logit.item():.4f}")

    # Restore best checkpoint
    if best_state:
        for n, p in model.named_parameters():
            if n in best_state:
                p.data.copy_(best_state[n])
    accelerator.print(f"  Best calib NLL: {best_nll:.4f}")

    model.eval()
    eval_metrics, eval_probs, eval_labels = evaluate_vp(model, eval_loader, device, args.n_mc_eval)
    eval_metrics.update(compute_all_metrics(eval_probs, eval_labels))
    accelerator.print(f"  Eval: {eval_metrics}")

    test_metrics = None
    if test_loader is not None:
        test_metrics, test_probs, test_labels = evaluate_vp(model, test_loader, device, args.n_mc_eval)
        test_metrics.update(compute_all_metrics(test_probs, test_labels))
        accelerator.print(f"  Test: {test_metrics}")

    # Collect learnable scalars for logging
    # model.log_s may be a property (ParameterList case) — always materialise it
    _log_s_final = model.log_s.detach().cpu() if not isinstance(model.log_s, nn.Parameter) \
                   else model.log_s.detach().cpu()
    if _log_s_final.numel() > 1:
        scalars = {"log_s": _log_s_final,           # [N] tensor — kept as tensor for phase-2 transfer
                   "log_s_list": _log_s_final.tolist(),  # JSON-serializable copy
                   "log_s_mean": _log_s_final.mean().item(),
                   "s_mean": _log_s_final.exp().mean().item()}
    else:
        scalars = {"log_s": _log_s_final.item(), "s": _log_s_final.exp().item()}
    if hasattr(model, "log_T_list"):
        _lt = model.log_T.detach().cpu()
        scalars["log_T"]      = _lt.tolist()
        scalars["log_T_mean"] = _lt.mean().item()
    elif hasattr(model, "log_T_attn"):
        scalars["log_T_attn"] = model.log_T_attn.item()
        scalars["log_T_mlp"]  = model.log_T_mlp.item()
    elif hasattr(model, "log_T"):
        scalars["log_T"]      = model.log_T.detach().cpu().tolist()
        scalars["log_T_mean"] = model.log_T.mean().item()
    if hasattr(model, "log_T_logit"):
        scalars["log_T_logit"] = model.log_T_logit.item()

    # Save calibrated parameters as a .pt checkpoint for later reuse
    calib_ckpt = {n: p.detach().cpu() for n, p in model.named_parameters()}
    calib_ckpt_path = os.path.join(
        args.laplace_output_dir, f"step_{args.load_step}",
        f"calib_params_{variant_name}.pt"
    )
    os.makedirs(os.path.dirname(calib_ckpt_path), exist_ok=True)
    torch.save(calib_ckpt, calib_ckpt_path)
    accelerator.print(f"  Saved calibration params → {calib_ckpt_path}")

    # Build JSON-safe copy of scalars (strip tensors)
    scalars_json = {k: (v.tolist() if isinstance(v, torch.Tensor) else v)
                    for k, v in scalars.items()}

    return {
        "variant":  variant_name,
        "scalars":  scalars,
        "scalars_json": scalars_json,
        "eval":     eval_metrics,
        "test":     test_metrics,
        "best_calib_nll": best_nll,
    }


# ---------------------------------------------------------------------------
# Data preprocessing (identical to run_gpt_laplace.py)
# ---------------------------------------------------------------------------

def preprocess_function(examples, task_name, tokenizer, padding, max_length):
    if task_name == "boolq":
        texts = [f"Answer the question with only True or False: {q} Context: {p}"
                 for p, q in zip(examples["passage"], examples["question"])]
        result = tokenizer(texts, padding=padding, max_length=max_length, truncation=True)
        result["labels"] = examples["label"]
    elif "winogrande" in task_name:
        texts = [
            f"Select one of the choices that answers the following question: "
            f"{s} Choices: A. {o1}. B {o2}. Answer:"
            for s, o1, o2 in zip(examples["sentence"], examples["option1"], examples["option2"])
        ]
        result = tokenizer(texts, padding=padding, max_length=max_length, truncation=True)
        result["labels"] = [{"1": 0, "2": 1, "": None}[a] for a in examples["answer"]]
    elif "openbookqa" in task_name:
        choices_list = [" ".join(f"{l}. {t}" for l, t in zip(c["label"], c["text"]))
                        for c in examples["choices"]]
        texts = [f"Select one of the choices that answers the following question: "
                 f"{q} Choices: {c} Answer:"
                 for q, c in zip(examples["question_stem"], choices_list)]
        result = tokenizer(texts, padding=padding, max_length=max_length, truncation=True)
        result["labels"] = [{"A":0,"B":1,"C":2,"D":3,"1":0,"2":1,"3":2,"4":3}[a] for a in examples["answerKey"]]
    elif "ARC" in task_name:
        choices_list = [" ".join(f"{l}. {t}" for l, t in zip(c["label"], c["text"]))
                        for c in examples["choices"]]
        texts = [f"Select one of the choices that answers the following question: "
                 f"{q} Choices: {c} Answer:"
                 for q, c in zip(examples["question"], choices_list)]
        result = tokenizer(texts, padding=padding, max_length=max_length, truncation=True)
        result["labels"] = [{"A":0,"B":1,"C":2,"D":3,"1":0,"2":1,"3":2,"4":3}[a] for a in examples["answerKey"]]
    else:
        raise ValueError(f"Unsupported task: {task_name}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    accelerator = Accelerator()
    set_seed(args.seed)
    device = accelerator.device

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # ── dataset ──────────────────────────────────────────────────────────────
    if "winogrande" in args.task_name:
        raw_datasets = load_dataset("winogrande", args.task_name)
    elif "ARC" in args.task_name:
        raw_datasets = load_dataset("ai2_arc", args.task_name)
    elif args.task_name in ("boolq", "cb", "wic"):
        raw_datasets = load_dataset("super_glue", args.task_name)
    else:
        raw_datasets = load_dataset(args.task_name)

    # ── tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer,
        padding_side="left", token=True,
    )
    tokenizer.pad_token = tokenizer.bos_token
    if args.task_name == "boolq":
        tokenizer.add_eos_token = True

    # ── tokenise ─────────────────────────────────────────────────────────────
    # ARC and openbookqa: filter to 4-choice questions and normalise choice text
    # to match the training distribution in run_gpt.py exactly.
    if "ARC" in args.task_name or "openbookqa" in args.task_name:
        raw_datasets = raw_datasets.filter(lambda ex: len(ex["choices"]["label"]) == 4)
        # run_gpt.py applies convert_choices_to_alpha: numeric labels → alphabetical,
        # trailing period on choice text, capitalize first letter.  The model was
        # TRAINED on this reformatted text, so we must apply the same transformation
        # at eval time to match the training distribution.
        def _convert_arc_choices(example):
            mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
            example['choices']['label'] = [mapping.get(l, l) for l in example['choices']['label']]
            example['answerKey'] = mapping.get(example['answerKey'], example['answerKey'])
            example['choices']['text'] = [
                (t if t.endswith('.') else t + '.') for t in example['choices']['text']
            ]
            example['choices']['text'] = [
                (t[0].upper() + t[1:] if t else t) for t in example['choices']['text']
            ]
            return example
        with accelerator.main_process_first():
            for split in raw_datasets:
                raw_datasets[split] = raw_datasets[split].map(_convert_arc_choices)

    padding = "max_length" if args.pad_to_max_length else False
    with accelerator.main_process_first():
        processed = raw_datasets.map(
            lambda ex: preprocess_function(ex, args.task_name, tokenizer, padding, args.max_length),
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenising",
        )

    train_dataset = processed["train"]
    val_dataset   = processed["validation_matched" if args.task_name == "mnli" else "validation"]

    if args.testing_set == "test":
        split = val_dataset.train_test_split(test_size=0.5, seed=42, shuffle=False)
        val_dataset, eval_dataset = split["train"], split["test"]
    elif args.testing_set == "train_val":
        split = train_dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
        train_dataset = split["train"]
        eval_dataset  = val_dataset
        val_dataset   = split["test"]
    else:
        eval_dataset = val_dataset

    data_collator = (
        default_data_collator if args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    )
    val_dataset  = val_dataset.shuffle(seed=42)
    calib_loader = DataLoader(val_dataset,   shuffle=False, collate_fn=data_collator,
                              batch_size=args.calib_batch_size)
    val_loader   = DataLoader(val_dataset,   shuffle=False, collate_fn=data_collator,
                              batch_size=args.per_device_eval_batch_size)
    eval_loader  = DataLoader(eval_dataset,  shuffle=False, collate_fn=data_collator,
                              batch_size=args.per_device_eval_batch_size)

    test_loader = None
    if args.eval_on_test and "test" in processed:
        test_loader = DataLoader(processed["test"], shuffle=False, collate_fn=data_collator,
                                 batch_size=args.per_device_eval_batch_size)
        accelerator.print(f"  Test split loaded: {len(processed['test'])} examples")

    # ── model ────────────────────────────────────────────────────────────────
    checkpoint_dir = f"{args.output_dir}/step_{args.load_step}"
    peft_config    = PeftConfig.from_pretrained(checkpoint_dir)
    base_model     = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16, token=True,
    )
    model      = PeftModel.from_pretrained(base_model, checkpoint_dir)

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "lora" in name and "all" in args.laplace_sub:
            param.requires_grad = True

    # ── answer token ids ─────────────────────────────────────────────────────
    if "winogrande" in args.task_name:
        id_list = [tokenizer.encode("A", add_special_tokens=False)[0],
                   tokenizer.encode("B", add_special_tokens=False)[0]]
    elif args.task_name in ("openbookqa",) or "ARC" in args.task_name:
        id_list = [tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCD"]
    elif args.task_name == "boolq":
        id_list = [tokenizer.encode("False", add_special_tokens=False)[0],
                   tokenizer.encode("True",  add_special_tokens=False)[0]]
    else:
        raise ValueError(f"No id_list for task {args.task_name}")

    # ── CustomLMHead_lora — vocabulary trimmed to answer tokens ──────────────
    class CustomLMHead_lora(nn.Module):
        def __init__(self, original_lm_head):
            super().__init__()
            orig_weight = original_lm_head.weight[id_list, :].clone()
            self.linear = nn.Linear(orig_weight.shape[1], len(id_list), bias=False).to(device)
            self.linear.weight.data = orig_weight.float()
            self.linear.weight.requires_grad = False

            self.lora_dropout = original_lm_head.lora_dropout["default"]
            A_w = original_lm_head.lora_A["default"].weight.clone()
            self.lora_A = nn.Linear(A_w.shape[1], A_w.shape[0], bias=False).to(device)
            self.lora_A.weight.data = A_w.float()
            self.lora_A.weight.requires_grad = True

            B_w = original_lm_head.lora_B["default"].weight[id_list, :].clone()
            self.lora_B = nn.Linear(B_w.shape[1], len(id_list), bias=False).to(device)
            self.lora_B.weight.data = B_w.float()
            self.lora_B.weight.requires_grad = True

            self.scaling = args.lora_alpha / args.lora_r

        def forward(self, x):
            h = x[:, -1, :].float()
            return (self.linear(h)
                    + self.lora_B(self.lora_A(self.lora_dropout(h))) * self.scaling)

    original_lm_head = model.base_model.model.lm_head

    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    model.gradient_checkpointing_disable()
    loaders = [calib_loader, val_loader, eval_loader]
    if test_loader is not None:
        loaders.append(test_loader)
    loaders = accelerator.prepare(*loaders)
    calib_loader, val_loader, eval_loader = loaders[:3]
    test_loader = loaders[3] if test_loader is not None else None

    steps_per_epoch = math.ceil(len(calib_loader) / args.grad_accum)
    if args.total_steps is not None:
        args.n_epochs = math.ceil(args.total_steps / steps_per_epoch)
        accelerator.print(f"  total_steps={args.total_steps} → steps_per_epoch={steps_per_epoch} → n_epochs={args.n_epochs}")
    if args.total_steps_s is not None:
        args.n_epochs_s = math.ceil(args.total_steps_s / steps_per_epoch)
        accelerator.print(f"  total_steps_s={args.total_steps_s} → steps_per_epoch={steps_per_epoch} → n_epochs_s={args.n_epochs_s}")

    custom_lm_head   = CustomLMHead_lora(original_lm_head).to(device)
    model.base_model.model.lm_head = custom_lm_head

    # ── Load KFAC posterior variances from prior Laplace run ──────────────────
    laplace_dir = f"{args.laplace_output_dir}/step_{args.load_step}"
    H_path  = os.path.join(laplace_dir,
                           f"laplace_H_{args.laplace_hessian}_{args.laplace_sub}.pt")
    pp_path = os.path.join(laplace_dir,
                           f"prior_precision_{args.laplace_hessian}_{args.laplace_sub}"
                           f"_{args.laplace_prior}_{args.laplace_optim_step}.pt")

    accelerator.print(f"--- Loading Laplace posterior from {laplace_dir} ---")
    H_dict          = torch.load(H_path,  map_location=device, weights_only=False)
    prior_precision = torch.load(pp_path, map_location=device, weights_only=False)
    accelerator.print(f"Prior precision: {prior_precision}")

    # Match Hessian factors to parameter names using the same requires_grad filter
    # as run_gpt_laplace.py (same model setup → same parameter order as la.fit).
    # wrapped = WrappedModel(model) would prefix names with "model.", so we do
    # the same here to keep names consistent with param_variances lookup below.
    trainable_params = [
        (f"model.{n}", p) for n, p in model.named_parameters() if p.requires_grad
    ]

    param_variances = extract_param_variances(H_dict, prior_precision, trainable_params, device)
    for name, var in param_variances.items():
        accelerator.print(f"  {name}: {var.shape}  mean={var.mean():.3e}")

    # Kron factors for improved LoRA VP (KFAC only; None for diagonal Hessian)
    kron_factors = extract_kron_lora_factors(H_dict, trainable_params, device)
    if kron_factors:
        accelerator.print(f"  Extracted Kronecker factors for {len(kron_factors)} parameters")

    lora_A_var = next(v for n, v in param_variances.items() if "lm_head.lora_A" in n)
    lora_B_var = next(v for n, v in param_variances.items() if "lm_head.lora_B" in n)

    # ── Build shared VP modules ───────────────────────────────────────────────
    vp_head = VPLMHead(custom_lm_head, lora_A_var, lora_B_var).to(device)

    backbone = model.base_model.model.model   # LlamaModel
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Backbone VP variants require non-zero backbone LoRA variances (laplace_sub=all).
    # With laplace_sub=last_layer the backbone variance is identically zero, making
    # log_T untrainable (gradient = T * 0 = 0).  Detect and skip automatically.
    has_backbone_var = any(
        v.abs().max().item() > 0
        for n, v in param_variances.items()
        if "lm_head" not in n
    )
    run_global              = RUN_GLOBAL
    run_per_layer           = RUN_PER_LAYER
    run_per_sub_block       = RUN_PER_SUB_BLOCK
    run_per_sub_block_logit = RUN_PER_SUB_BLOCK_LOGIT
    run_per_layer_logit     = RUN_PER_LAYER_LOGIT
    run_per_block_st        = RUN_PER_BLOCK_ST
    if not has_backbone_var and any([run_global, run_per_layer, run_per_sub_block,
                                     run_per_sub_block_logit, run_per_layer_logit,
                                     run_per_block_st]):
        accelerator.print(
            "WARNING: backbone LoRA variances are all zero (laplace_sub=last_layer). "
            "Skipping backbone VP variants — rerun with laplace_sub=all for these."
        )
        run_global = run_per_layer = run_per_sub_block = run_per_sub_block_logit = run_per_layer_logit = run_per_block_st = False

    need_vp_backbone = any([RUN_LOG_S_ONLY, RUN_LOG_S_PER_BLOCK, run_global, run_per_layer,
                            run_per_sub_block, run_per_sub_block_logit, run_per_layer_logit,
                            run_per_block_st])
    if need_vp_backbone:
        accelerator.print(f"--- Building VPBackbone (rms_norm={args.rms_norm_method}, swiglu={args.swiglu_method}, attn={args.attn_var_mode}, kron={bool(kron_factors)}) ---")
        vp_backbone = build_vp_backbone(backbone, param_variances,
                                        args.rms_norm_method, args.swiglu_method,
                                        attn_var_mode=args.attn_var_mode,
                                        kron_factors=kron_factors or None).to(device)
        vp_backbone.eval()
    else:
        vp_backbone = None

    # ── init_log_s: learned in phase-1 (log_s_only), then frozen in later phases ─
    _init_log_s = args.init_log_s if args.init_log_s is not None else -4.0
    # Will be updated to the phase-1 best value before any backbone variant runs.
    _learned_log_s = _init_log_s

    # ── Run variants ──────────────────────────────────────────────────────────
    all_results = []

    _phase1_names = {"log_s_only", "log_s_per_block"}

    variants = [
        (RUN_LOG_S_ONLY,          "log_s_only",           lambda: VPCalibLogSOnly(vp_backbone, vp_head)),
        (RUN_LOG_S_PER_BLOCK,     "log_s_per_block",      lambda: VPCalibLogSPerBlock(vp_backbone, vp_head)),
        (run_global,              "global",               lambda: VPCalibGlobal(vp_backbone, vp_head)),
        (run_per_layer,           "per_layer",            lambda: VPCalibPerLayer(vp_backbone, vp_head)),
        (run_per_sub_block,       "per_sub_block",        lambda: VPCalibPerSubBlock(vp_backbone, vp_head)),
        (run_per_sub_block_logit, "per_sub_block_logit",  lambda: VPCalibPerSubBlockLogit(vp_backbone, vp_head)),
        (run_per_layer_logit,     "per_layer_logit",      lambda: VPCalibPerLayerLogit(vp_backbone, vp_head, fixed_log_s=args.fixed_log_s)),
        (run_per_block_st,        "per_block_st",         lambda: VPCalibPerBlockST(vp_backbone, vp_head)),
    ]

    for flag, name, make_model in variants:
        if not flag:
            continue
        if name == "log_s_only" and args.init_log_s is not None:
            accelerator.print(f"  Skipping log_s_only fit — using init_log_s={args.init_log_s:.4f} directly")
            _learned_log_s = args.init_log_s
            continue
        calib_model = make_model().to(device)

        if args.load_calib_params:
            ckpt = torch.load(args.load_calib_params, map_location=device, weights_only=False)
            with torch.no_grad():
                for n, p in calib_model.named_parameters():
                    if n in ckpt:
                        p.data.copy_(ckpt[n].to(device))
            accelerator.print(f"  Loaded calib params from {args.load_calib_params}")

        if args.init_log_T is not None and hasattr(calib_model, "log_T_list"):
            with torch.no_grad():
                for p in calib_model.log_T_list:
                    p.data.fill_(args.init_log_T)
            accelerator.print(f"  init_log_T={args.init_log_T:.4f} broadcast to all {len(calib_model.log_T_list)} layers")
        if args.init_log_T_logit is not None and hasattr(calib_model, "log_T_logit"):
            with torch.no_grad():
                calib_model.log_T_logit.data.fill_(args.init_log_T_logit)
            accelerator.print(f"  init_log_T_logit={args.init_log_T_logit:.4f}")

        if name not in _phase1_names:
            if not isinstance(calib_model.log_s, nn.Parameter):
                if hasattr(calib_model, "log_s_list") and any(
                    p.requires_grad for p in calib_model.log_s_list
                ):
                    # Single-phase variant with per-block log_s — learned jointly with log_T
                    accelerator.print(
                        f"  single-phase per-block log_s+log_T "
                        f"({calib_model.n_calib_params} params, no phase-1)"
                    )
                    lr_s = None  # run_variant uses args.lr for log_s_list
                else:
                    # Fixed buffer (e.g. VPCalibPerLayerLogit) — no init or freeze needed
                    accelerator.print(f"  log_s fixed buffer = {calib_model.log_s.item():.4f} (not learned)")
                    lr_s = None
            else:
                # Initialise log_s from phase-1 result (scalar or [N] tensor), then freeze.
                _ls = _learned_log_s
                if isinstance(_ls, torch.Tensor) and _ls.numel() > 1:
                    calib_model.log_s = nn.Parameter(_ls.to(device).clone(), requires_grad=False)
                    accelerator.print(f"  Initialising log_s[N] from per-block phase-1 (frozen), mean={_ls.mean():.4f}")
                else:
                    val = float(_ls) if not isinstance(_ls, torch.Tensor) else _ls.item()
                    calib_model.log_s.data.fill_(val)
                    accelerator.print(f"  Initialising log_s = {val:.4f} (from phase-1, frozen)")
                if args.finetune_s:
                    calib_model.log_s.requires_grad_(True)
                    lr_s = args.lr_s_finetune
                else:
                    calib_model.log_s.requires_grad_(False)
                    lr_s = None
                if hasattr(calib_model, "log_s_list"):
                    for p in calib_model.log_s_list:
                        p.requires_grad_(args.finetune_s)
        else:
            lr_s = None  # use default args.lr

        phase_epochs = args.n_epochs_s if name in _phase1_names else args.n_epochs
        result = run_variant(calib_model, calib_loader, val_loader, eval_loader, device, args, name, accelerator, lr_s=lr_s, n_epochs=phase_epochs, test_loader=test_loader)

        # After phase-1, capture the best log_s for all subsequent phases.
        if name in _phase1_names:
            _learned_log_s = result["scalars"]["log_s"]  # scalar or [N] tensor
            if isinstance(_learned_log_s, torch.Tensor) and _learned_log_s.numel() > 1:
                accelerator.print(f"  Phase-1 best log_s[N] mean={_learned_log_s.mean():.4f}")
            else:
                accelerator.print(f"  Phase-1 best log_s = {float(_learned_log_s):.4f}")

        all_results.append(result)
        del calib_model
        torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────────
    laplace_output_dir = f"{args.laplace_output_dir}/step_{args.load_step}"
    os.makedirs(laplace_output_dir, exist_ok=True)

    base_tag = (
        f"vp_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}"
        f"_{args.laplace_optim_step}_rms{args.rms_norm_method}_swiglu{args.swiglu_method}_epochs{args.n_epochs}"
        + (f"_{args.suffix}" if args.suffix else "")
    )
    results_path = os.path.join(
        f"{args.output_dir}/step_{args.load_step}", f"all_results_{base_tag}.json"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    # Use scalars_json (tensor-free) for serialization
    all_results_json = [{**r, "scalars": r["scalars_json"]} for r in all_results]
    with open(results_path, "w") as f:
        json.dump(all_results_json, f, indent=2)
    accelerator.print(f"\nAll results saved to {results_path}")

    if args.results_dir and args.seed_label:
        _suffix_str = f"_{args.suffix}" if args.suffix else ""
        clean_dir  = os.path.join(args.results_dir, args.task_name, args.seed_label)
        os.makedirs(clean_dir, exist_ok=True)
        # Pick the per_layer_logit result (only active variant in eval mode)
        _r = next((r for r in all_results if r.get("variant") == "per_layer_logit"), all_results[-1])
        _ser = lambda m: {k: (v.tolist() if isinstance(v, torch.Tensor) else v)
                          for k, v in m.items()} if m is not None else None

        if args.merge_results_into:
            # Merge val_calibrated / test_calibrated into an existing JSON
            merge_path = os.path.join(clean_dir, args.merge_results_into)
            existing = {}
            if os.path.exists(merge_path):
                with open(merge_path) as f:
                    existing = json.load(f)
            existing[args.val_calib_key] = _ser(_r.get("eval"))
            if _r.get("test") is not None:
                existing[args.test_calib_key] = _ser(_r.get("test"))
            # Reorder: val → test → *_calibrated keys → rest
            _section_order = ["val", "test", args.val_calib_key, args.test_calib_key, "scalars"]
            ordered = {k: existing[k] for k in _section_order if k in existing}
            ordered.update({k: v for k, v in existing.items() if k not in ordered})
            with open(merge_path, "w") as f:
                json.dump(ordered, f, indent=2)
            accelerator.print(f"Calibrated results merged → {merge_path} "
                              f"[{args.val_calib_key}, {args.test_calib_key}]")
        else:
            clean_name = f"{args.swiglu_method}_{args.rms_norm_method}{_suffix_str}.json"
            clean_path = os.path.join(clean_dir, clean_name)
            clean_payload = {
                "val":  _ser(_r.get("eval")),
                **({"test": _ser(_r.get("test"))} if _r.get("test") is not None else {}),
                "scalars": _r["scalars_json"],
            }
            with open(clean_path, "w") as f:
                json.dump(clean_payload, f, indent=2)
            accelerator.print(f"Results saved → {clean_path}")


if __name__ == "__main__":
    main()
