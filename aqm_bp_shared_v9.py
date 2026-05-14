from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.nn.utils import clip_grad_norm_

from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import norm, pearsonr, spearmanr


# =========================================================
# Config
# =========================================================
@dataclass
class CFG:
    DATA_ROOT: str = os.environ.get(
        "AQM_MIMIC_BP_ROOT",
        os.environ.get("AQM_MIMIC_BP_DATA_ROOT", str(Path(__file__).resolve().parent / "data" / "raw" / "MIMIC-BP")),
    )
    PROJECT_ROOT: str = str(Path(__file__).resolve().parent)
    OUTPUT_NAME: str = "mimic_bp_reg_v9"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    SEED: int = 42
    BATCH_SIZE: int = 40
    EPOCHS: int = 80
    LR: float = 1.5e-4
    WEIGHT_DECAY: float = 1.0e-4
    NUM_WORKERS: int = 0
    GRAD_CLIP_NORM: float = 3.5
    WARMUP_EPOCHS: int = 6
    WARMUP_START_FACTOR: float = 0.35
    MIN_LR_RATIO: float = 0.10
    EMA_DECAY: float = 0.997

    SAMPLE_RATE: int = 125
    CROP_SECONDS: int = 10
    CROP_LEN: int = 1250

    EMBED_DIM: int = 128
    EXPERT_DIM: int = 160
    NUM_DIM: int = 8
    QFEAT_DIM: int = 10
    N_CLASSES: int = 4
    PATCH_SIZE: int = 25
    PATCH_STRIDE: int = 12
    PATCH_TOKENS_MAX: int = 96
    TRANSFORMER_HEADS: int = 4
    TRANSFORMER_LAYERS: int = 2
    TOPK_EXPERTS: int = 2
    ROUTER_DENSE_BLEND: float = 0.12
    CREDIBILITY_SMOOTHING: float = 0.04
    FUSED_CREDIBILITY_PRIOR: float = 0.60
    FUSED_CREDIBILITY_FLOOR: float = 0.18

    # regression defaults
    LAMBDA_MAP: float = 0.30
    LAMBDA_NLL: float = 0.15
    LAMBDA_Q: float = 0.02
    LAMBDA_ROUTER: float = 0.02
    LAMBDA_BAL: float = 0.01
    LAMBDA_PPG_AUX: float = 0.05
    LAMBDA_ECG_AUX: float = 0.10
    LAMBDA_CRED: float = 0.08
    LAMBDA_REG_CLS: float = 0.30

    # classification defaults
    LAMBDA_CLS: float = 1.0
    LAMBDA_ORD: float = 0.60
    LAMBDA_PROXY: float = 0.25
    LAMBDA_CENTER: float = 0.10

    LABEL_SMOOTHING: float = 0.02
    ORD_POS_WEIGHT_SCALE: float = 1.0
    FOCAL_GAMMA: float = 2.0
    CRED_TARGET_TEMP: float = 6.0

    MODALITY_DROPOUT_PPG: float = 0.20
    MODALITY_DROPOUT_ECG: float = 0.05
    REG_USE_WEIGHTED_SAMPLER: bool = True
    REG_SAMPLER_POWER: float = 0.65
    EARLY_STOPPING_PATIENCE: int = 18
    CONFORMAL_ALPHA: float = 0.10
    P_DEGRADE: float = 0.35
    AUG_RAMP_EPOCHS: int = 8
    AUG_WARMUP_FACTOR: float = 0.35

    NOISE_STDS: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.20, 0.30, 0.40)
    MISSING_PROBS: Tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00)
    CONFORMAL_ALPHAS: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)

    CLASS_NAMES: Tuple[str, ...] = ("Normal", "Elevated", "Stage1", "Stage2")
    PPG_BAND: Tuple[float, float] = (0.5, 8.0)
    ECG_BAND: Tuple[float, float] = (0.5, 35.0)
    QUALITY_BINS: Tuple[float, float] = (0.33, 0.66)
    ECE_BINS: int = 12
    LATENCY_WARMUP: int = 10
    LATENCY_ITERS: int = 50

    # centers only used for soft reg<->class conversion / class proxy
    CLASS_CENTER_SBP: Tuple[float, ...] = (110.0, 125.0, 135.0, 155.0)
    CLASS_CENTER_DBP: Tuple[float, ...] = (70.0, 75.0, 85.0, 95.0)


# =========================================================
# Utilities
# =========================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_subject_id(s: str) -> str:
    s = str(s).strip()
    if s.startswith("p"):
        return s
    if s.isdigit():
        return f"p{int(s):06d}"
    return s


def read_subject_txt(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [normalize_subject_id(line.strip()) for line in f if line.strip()]


def zscore_1d(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + eps)


def center_crop_or_pad(x: np.ndarray, crop_len: int) -> np.ndarray:
    n = len(x)
    if n == crop_len:
        return x
    if n > crop_len:
        start = (n - crop_len) // 2
        return x[start:start + crop_len]
    pad_left = (crop_len - n) // 2
    pad_right = crop_len - n - pad_left
    return np.pad(x, (pad_left, pad_right), mode="constant")


def random_crop_or_pad(x: np.ndarray, crop_len: int, rng: random.Random) -> np.ndarray:
    n = len(x)
    if n == crop_len:
        return x
    if n > crop_len:
        start = rng.randint(0, n - crop_len)
        return x[start:start + crop_len]
    pad_left = rng.randint(0, crop_len - n)
    pad_right = crop_len - n - pad_left
    return np.pad(x, (pad_left, pad_right), mode="constant")


def safe_bandpass_filter(x: np.ndarray, fs: int, low: float, high: float, order: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 32:
        return x
    nyq = 0.5 * fs
    low = max(low / nyq, 1e-4)
    high = min(high / nyq, 0.999)
    if not (0 < low < high < 1):
        return x
    try:
        b, a = butter(order, [low, high], btype="band")
        y = filtfilt(b, a, x).astype(np.float32)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return x
        return y
    except Exception:
        return x


def robust_quality_features_np(x: np.ndarray, fs: int = 125, qfeat_dim: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 16:
        return np.zeros(qfeat_dim, dtype=np.float32)
    dx = np.diff(x)
    z = zscore_1d(x)
    std = float(np.std(x))
    mad = float(np.mean(np.abs(dx)))
    energy = float(np.mean(x ** 2))
    flat_ratio = float(np.mean(np.abs(dx) < 1e-4))
    extreme_ratio = float(np.mean(np.abs(z) > 3.0))
    q01, q99 = np.percentile(x, [1, 99])
    span = max(q99 - q01, 1e-6)
    clip_ratio = 0.5 * (
        np.mean(x <= (q01 + 0.02 * span)) +
        np.mean(x >= (q99 - 0.02 * span))
    )

    prominence_thr = max(0.10 * np.std(x), 1e-4)
    distance = max(1, int(0.25 * fs))
    peaks, props = find_peaks(x, distance=distance, prominence=prominence_thr)
    peak_density = float(len(peaks) / max(len(x) / fs, 1e-6))
    if len(peaks) >= 2:
        ibi = np.diff(peaks) / fs
        ibi_cv = float(np.std(ibi) / (np.mean(ibi) + 1e-6))
    else:
        ibi_cv = 1.0

    prominences = props.get("prominences", np.asarray([], dtype=np.float32))
    if len(prominences) > 0:
        prom_mean = float(np.mean(prominences))
        prom_cv = float(np.std(prominences) / (np.mean(prominences) + 1e-6))
    else:
        prom_mean = 0.0
        prom_cv = 1.0

    return np.array([
        std, mad, energy, flat_ratio, extreme_ratio,
        float(clip_ratio), peak_density, ibi_cv, prom_mean, prom_cv,
    ], dtype=np.float32)


def tensor_quality_features(signal_batch: torch.Tensor, fs: int = 125, qfeat_dim: int = 10) -> torch.Tensor:
    x = signal_batch[:, 0, :]
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
    z = (x - mean) / std
    dx = x[:, 1:] - x[:, :-1]

    feat_std = x.std(dim=1)
    feat_mad = dx.abs().mean(dim=1)
    feat_energy = (x ** 2).mean(dim=1)
    feat_flat = (dx.abs() < 1e-4).float().mean(dim=1)
    feat_extreme = (z.abs() > 3.0).float().mean(dim=1)
    q01 = torch.quantile(x, 0.01, dim=1)
    q99 = torch.quantile(x, 0.99, dim=1)
    span = (q99 - q01).clamp(min=1e-6)
    low_thr = q01 + 0.02 * span
    high_thr = q99 - 0.02 * span
    feat_clip = 0.5 * (
        (x <= low_thr[:, None]).float().mean(dim=1) +
        (x >= high_thr[:, None]).float().mean(dim=1)
    )

    x_np = x.detach().cpu().numpy()
    peak_density, ibi_cv, prom_mean, prom_cv = [], [], [], []
    for i in range(x.shape[0]):
        feat = robust_quality_features_np(x_np[i], fs=fs, qfeat_dim=qfeat_dim)
        peak_density.append(feat[6])
        ibi_cv.append(feat[7])
        prom_mean.append(feat[8])
        prom_cv.append(feat[9])

    return torch.stack([
        feat_std,
        feat_mad,
        feat_energy,
        feat_flat,
        feat_extreme,
        feat_clip,
        torch.tensor(peak_density, dtype=torch.float32, device=x.device),
        torch.tensor(ibi_cv, dtype=torch.float32, device=x.device),
        torch.tensor(prom_mean, dtype=torch.float32, device=x.device),
        torch.tensor(prom_cv, dtype=torch.float32, device=x.device),
    ], dim=1)


def bp_to_risk_class(sbp: float, dbp: float) -> int:
    if sbp >= 140 or dbp >= 90:
        return 3
    if (130 <= sbp < 140) or (80 <= dbp < 90):
        return 2
    if (120 <= sbp < 130) and dbp < 80:
        return 1
    return 0


def reg_to_class_np(y_reg_pred: np.ndarray) -> np.ndarray:
    out = []
    for i in range(len(y_reg_pred)):
        sbp, dbp = float(y_reg_pred[i, 0]), float(y_reg_pred[i, 1])
        out.append(bp_to_risk_class(sbp, dbp))
    return np.array(out, dtype=np.int64)


def class_prob_to_bp_proxy(y_prob: np.ndarray, cfg: CFG) -> np.ndarray:
    centers = np.stack([
        np.asarray(cfg.CLASS_CENTER_SBP, dtype=np.float32),
        np.asarray(cfg.CLASS_CENTER_DBP, dtype=np.float32),
    ], axis=1)
    return y_prob @ centers


def _regression_to_class_prob_center(y_pred_reg: np.ndarray, uncertainty: Optional[np.ndarray], cfg: CFG) -> np.ndarray:
    centers = np.stack([
        np.asarray(cfg.CLASS_CENTER_SBP, dtype=np.float32),
        np.asarray(cfg.CLASS_CENTER_DBP, dtype=np.float32),
    ], axis=1)
    diff = y_pred_reg[:, None, :] - centers[None, :, :]
    sq = np.sum(diff ** 2, axis=2)
    if uncertainty is None:
        scale = 18.0
    else:
        unc = np.asarray(uncertainty).reshape(-1, 1)
        scale = np.clip(np.sqrt(unc + 1e-6), 8.0, 30.0)
    logits = -sq / (2.0 * (scale ** 2))
    logits = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    prob /= np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)
    return prob.astype(np.float32)


def regression_to_class_prob(y_pred_reg: np.ndarray, uncertainty: Optional[np.ndarray], cfg: CFG) -> np.ndarray:
    y_pred_reg = np.asarray(y_pred_reg, dtype=np.float32)
    if uncertainty is None:
        sigma_base = np.full((y_pred_reg.shape[0],), 12.0, dtype=np.float32)
    else:
        sigma_base = np.clip(np.sqrt(np.asarray(uncertainty, dtype=np.float32).reshape(-1) + 1e-6), 6.0, 22.0)
    sigma_sbp = np.clip(1.10 * sigma_base, 6.5, 24.0)
    sigma_dbp = np.clip(0.85 * sigma_base, 5.0, 18.0)

    sbp = y_pred_reg[:, 0]
    dbp = y_pred_reg[:, 1]
    p_sbp_lt120 = norm.cdf((120.0 - sbp) / sigma_sbp)
    p_sbp_lt130 = norm.cdf((130.0 - sbp) / sigma_sbp)
    p_sbp_lt140 = norm.cdf((140.0 - sbp) / sigma_sbp)
    p_dbp_lt80 = norm.cdf((80.0 - dbp) / sigma_dbp)
    p_dbp_lt90 = norm.cdf((90.0 - dbp) / sigma_dbp)

    p_normal = p_sbp_lt120 * p_dbp_lt80
    p_elevated = np.clip((p_sbp_lt130 - p_sbp_lt120) * p_dbp_lt80, 0.0, 1.0)
    p_stage2 = np.clip(1.0 - (p_sbp_lt140 * p_dbp_lt90), 0.0, 1.0)
    p_stage1 = np.clip(1.0 - p_normal - p_elevated - p_stage2, 0.0, 1.0)

    thresh_prob = np.stack([p_normal, p_elevated, p_stage1, p_stage2], axis=1).astype(np.float32)
    thresh_prob = thresh_prob / np.clip(thresh_prob.sum(axis=1, keepdims=True), 1e-12, None)

    # Blend threshold-aware posteriors with center-based softness to stabilize very uncertain samples.
    center_prob = _regression_to_class_prob_center(y_pred_reg, uncertainty, cfg)
    thresh_blend = float(np.clip(getattr(cfg, "REG_TO_CLASS_THRESHOLD_BLEND", 0.85), 0.0, 1.0))
    center_blend = float(1.0 - thresh_blend)
    prob = thresh_blend * thresh_prob + center_blend * center_prob
    prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)
    return prob.astype(np.float32)


def gaussian_cdf_torch(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def regression_to_class_prob_torch(
    y_pred_reg: torch.Tensor,
    uncertainty: Optional[torch.Tensor],
    cfg: CFG,
) -> torch.Tensor:
    if uncertainty is None:
        sigma_base = torch.full(
            (y_pred_reg.shape[0],),
            12.0,
            dtype=y_pred_reg.dtype,
            device=y_pred_reg.device,
        )
    else:
        unc = uncertainty.reshape(-1).to(dtype=y_pred_reg.dtype)
        sigma_base = torch.sqrt(torch.clamp(unc, min=1e-6)).clamp(6.0, 22.0)
    sigma_sbp = (1.10 * sigma_base).clamp(6.5, 24.0)
    sigma_dbp = (0.85 * sigma_base).clamp(5.0, 18.0)

    sbp = y_pred_reg[:, 0]
    dbp = y_pred_reg[:, 1]
    p_sbp_lt120 = gaussian_cdf_torch((120.0 - sbp) / sigma_sbp)
    p_sbp_lt130 = gaussian_cdf_torch((130.0 - sbp) / sigma_sbp)
    p_sbp_lt140 = gaussian_cdf_torch((140.0 - sbp) / sigma_sbp)
    p_dbp_lt80 = gaussian_cdf_torch((80.0 - dbp) / sigma_dbp)
    p_dbp_lt90 = gaussian_cdf_torch((90.0 - dbp) / sigma_dbp)

    p_normal = p_sbp_lt120 * p_dbp_lt80
    p_elevated = torch.clamp((p_sbp_lt130 - p_sbp_lt120) * p_dbp_lt80, min=0.0, max=1.0)
    p_stage2 = torch.clamp(1.0 - (p_sbp_lt140 * p_dbp_lt90), min=0.0, max=1.0)
    p_stage1 = torch.clamp(1.0 - p_normal - p_elevated - p_stage2, min=0.0, max=1.0)
    thresh_prob = torch.stack([p_normal, p_elevated, p_stage1, p_stage2], dim=1)
    thresh_prob = thresh_prob / thresh_prob.sum(dim=1, keepdim=True).clamp(min=1e-8)

    centers = torch.tensor(
        np.stack(
            [
                np.asarray(cfg.CLASS_CENTER_SBP, dtype=np.float32),
                np.asarray(cfg.CLASS_CENTER_DBP, dtype=np.float32),
            ],
            axis=1,
        ),
        dtype=y_pred_reg.dtype,
        device=y_pred_reg.device,
    )
    diff = y_pred_reg[:, None, :] - centers[None, :, :]
    sq = diff.pow(2).sum(dim=2)
    if uncertainty is None:
        scale = torch.full_like(sigma_base.unsqueeze(1), 18.0)
    else:
        scale = sigma_base.unsqueeze(1).clamp(8.0, 30.0)
    center_prob = torch.softmax(-sq / (2.0 * scale.pow(2)), dim=1)

    thresh_blend = float(max(0.0, min(1.0, getattr(cfg, "REG_TO_CLASS_THRESHOLD_BLEND", 0.85))))
    center_blend = float(1.0 - thresh_blend)
    prob = thresh_blend * thresh_prob + center_blend * center_prob
    prob = prob / prob.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return prob


def build_credibility_prior_torch(avail: torch.Tensor, fused_prior: float = 0.60) -> torch.Tensor:
    device = avail.device
    dtype = avail.dtype
    ppg = avail[:, 0:1]
    ecg = avail[:, 1:2]
    joint = torch.maximum(ppg, ecg)
    both = torch.minimum(ppg, ecg)

    single_share = max(0.0, 1.0 - float(fused_prior))
    both_prior = torch.tensor(
        [0.60 * single_share, 0.40 * single_share, float(fused_prior)],
        device=device,
        dtype=dtype,
    ).unsqueeze(0)
    ppg_only_prior = torch.tensor([0.75, 0.0, 0.25], device=device, dtype=dtype).unsqueeze(0)
    ecg_only_prior = torch.tensor([0.0, 0.75, 0.25], device=device, dtype=dtype).unsqueeze(0)

    prior = ppg_only_prior * (ppg * (1.0 - ecg))
    prior = prior + ecg_only_prior * (ecg * (1.0 - ppg))
    prior = prior + both_prior * both
    prior = prior + torch.cat(
        [
            (ppg <= 0).float() * (ecg <= 0).float(),
            torch.zeros_like(joint),
            torch.zeros_like(joint),
        ],
        dim=1,
    )
    prior = prior / prior.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return prior


def ordinal_targets_from_class(y_cls: torch.Tensor) -> torch.Tensor:
    t0 = (y_cls >= 1).float()
    t1 = (y_cls >= 2).float()
    t2 = (y_cls >= 3).float()
    return torch.stack([t0, t1, t2], dim=1)


def ordinal_logits_to_class_prob(logits_ord: torch.Tensor) -> torch.Tensor:
    p_ge_1 = torch.sigmoid(logits_ord[:, 0])
    p_ge_2 = torch.sigmoid(logits_ord[:, 1])
    p_ge_3 = torch.sigmoid(logits_ord[:, 2])
    p0 = 1.0 - p_ge_1
    p1 = torch.clamp(p_ge_1 - p_ge_2, min=0.0)
    p2 = torch.clamp(p_ge_2 - p_ge_3, min=0.0)
    p3 = torch.clamp(p_ge_3, min=0.0)
    probs = torch.stack([p0, p1, p2, p3], dim=1)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return probs


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(pearsonr(x, y)[0])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    val = spearmanr(x, y).correlation
    return float(0.0 if np.isnan(val) else val)


def concordance_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mt = y_true.mean()
    mp = y_pred.mean()
    vt = y_true.var()
    vp = y_pred.var()
    cov = np.mean((y_true - mt) * (y_pred - mp))
    denom = vt + vp + (mt - mp) ** 2
    if denom < 1e-12:
        return 0.0
    return float(2.0 * cov / denom)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 12) -> float:
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += float(m.mean()) * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    y_onehot = np.eye(n_classes, dtype=np.float32)[y_true.astype(int)]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def specificity_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> List[float]:
    out = []
    for c in range(n_classes):
        tn = np.sum((y_true != c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))
        denom = tn + fp
        out.append(float(tn / denom) if denom > 0 else 0.0)
    return out


def np_trapz_auc(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return 0.0
    order = np.argsort(np.asarray(x))
    return float(np.trapezoid(np.asarray(y)[order], np.asarray(x)[order]))


def clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.clone() if torch.is_tensor(v) else v
    return out


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def ramp_factor(epoch: Optional[int], full_after: int, min_factor: float = 0.0) -> float:
    if epoch is None or full_after <= 0:
        return 1.0
    progress = min(1.0, max(0.0, float(epoch) / float(full_after)))
    return float(min_factor + (1.0 - min_factor) * progress)


def set_warmup_cosine_lr(optimizer: torch.optim.Optimizer, cfg: CFG, epoch: int) -> float:
    warmup_epochs = max(0, min(int(cfg.WARMUP_EPOCHS), int(cfg.EPOCHS) - 1))
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        factor = cfg.WARMUP_START_FACTOR + (1.0 - cfg.WARMUP_START_FACTOR) * (float(epoch) / float(warmup_epochs))
    else:
        denom = max(1, int(cfg.EPOCHS) - warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(denom)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = cfg.MIN_LR_RATIO + (1.0 - cfg.MIN_LR_RATIO) * cosine
    lr = float(cfg.LR * factor)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for key, ema_val in ema_state.items():
            model_val = model_state[key]
            if not torch.is_floating_point(ema_val):
                ema_val.copy_(model_val)
                continue
            ema_val.mul_(self.decay).add_(model_val.detach(), alpha=1.0 - self.decay)


# =========================================================
# Dataset
# =========================================================
class MIMICBPDataset(Dataset):
    def __init__(self, cfg: CFG, subjects: List[str], crop_len: int, mode: str = "train", seed: int = 42):
        self.cfg = cfg
        self.root = Path(cfg.DATA_ROOT)
        self.crop_len = crop_len
        self.mode = mode
        self.rng = random.Random(seed)
        self.ppg_dir = self.root / "ppg"
        self.ecg_dir = self.root / "ecg"
        self.labels_dir = self.root / "labels"
        self.subjects = [normalize_subject_id(s) for s in subjects]
        self.index = []
        self.sample_classes = []
        self.cache: Dict[str, Dict[str, np.ndarray]] = {}
        for sid in self.subjects:
            labels = self._load_labels_only(sid)
            n_seg = labels.shape[0]
            for seg_idx in range(n_seg):
                sbp, dbp = labels[seg_idx]
                cls = bp_to_risk_class(float(sbp), float(dbp))
                self.index.append((sid, seg_idx))
                self.sample_classes.append(cls)
        self.class_counts = np.bincount(np.asarray(self.sample_classes), minlength=cfg.N_CLASSES).tolist()

    def _path(self, folder: Path, sid: str, suffix: str) -> Path:
        candidates = [
            folder / f"{sid}_{suffix}.npy",
            folder / f"{sid}.npy",
            folder / f"{sid}_label.npy" if suffix == "labels" else folder / f"{sid}.npy",
            folder / f"{sid}_labels.npy" if suffix == "labels" else folder / f"{sid}.npy",
        ]
        for c in candidates:
            if c.exists():
                return c
        return folder / f"{sid}_{suffix}.npy"

    def _load_npy(self, path: Path):
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
            arr = arr.item()
        return arr

    def _normalize_wave_shape(self, x) -> np.ndarray:
        if isinstance(x, dict):
            for k in ["signal", "wave", "waves", "data", "x"]:
                if k in x:
                    x = x[k]
                    break
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            if x.shape[0] == 30:
                return x
            if x.shape[1] == 30:
                return x.T
        if x.ndim == 1 and x.size % 30 == 0:
            return x.reshape(30, x.size // 30)
        raise ValueError(f"Unsupported waveform shape: {x.shape}")

    def _labels_from_array(self, arr: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape == (30, 2):
            return arr
        if arr.ndim == 2 and arr.shape == (2, 30):
            return arr.T
        if arr.ndim == 1 and arr.size == 60:
            return arr.reshape(30, 2)
        return None

    def _labels_from_mapping(self, obj: dict) -> Optional[np.ndarray]:
        lowered = {str(k).lower(): v for k, v in obj.items()}
        for k1, k2 in [("sbp", "dbp"), ("sys", "dia"), ("systolic", "diastolic")]:
            if k1 in lowered and k2 in lowered:
                sbp = np.asarray(lowered[k1], dtype=np.float32).reshape(-1)
                dbp = np.asarray(lowered[k2], dtype=np.float32).reshape(-1)
                if len(sbp) == 30 and len(dbp) == 30:
                    return np.stack([sbp, dbp], axis=1)
        return None

    def _load_labels_only(self, sid: str) -> np.ndarray:
        path = self._path(self.labels_dir, sid, "labels")
        obj = self._load_npy(path)
        if isinstance(obj, dict):
            out = self._labels_from_mapping(obj)
            if out is not None:
                return out
        out = self._labels_from_array(obj)
        if out is not None:
            return out
        raise ValueError(f"Unsupported labels format for {sid}: type={type(obj)}")

    def _load_subject(self, sid: str) -> Dict[str, np.ndarray]:
        if sid in self.cache:
            return self.cache[sid]
        ppg = self._normalize_wave_shape(self._load_npy(self._path(self.ppg_dir, sid, "ppg")))
        ecg = self._normalize_wave_shape(self._load_npy(self._path(self.ecg_dir, sid, "ecg")))
        labels = self._load_labels_only(sid)
        obj = {"ppg": ppg, "ecg": ecg, "labels": labels}
        if len(self.cache) > 32:
            self.cache.pop(next(iter(self.cache)))
        self.cache[sid] = obj
        return obj

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        sid, seg_idx = self.index[idx]
        item = self._load_subject(sid)
        ppg = item["ppg"][seg_idx].astype(np.float32)
        ecg = item["ecg"][seg_idx].astype(np.float32)
        label = item["labels"][seg_idx].astype(np.float32)

        if self.mode == "train":
            ppg_crop = random_crop_or_pad(ppg, self.crop_len, self.rng)
            ecg_crop = random_crop_or_pad(ecg, self.crop_len, self.rng)
        else:
            ppg_crop = center_crop_or_pad(ppg, self.crop_len)
            ecg_crop = center_crop_or_pad(ecg, self.crop_len)

        ppg_crop = safe_bandpass_filter(ppg_crop, self.cfg.SAMPLE_RATE, *self.cfg.PPG_BAND)
        ecg_crop = safe_bandpass_filter(ecg_crop, self.cfg.SAMPLE_RATE, *self.cfg.ECG_BAND)

        ppg_qfeat = robust_quality_features_np(ppg_crop, fs=self.cfg.SAMPLE_RATE, qfeat_dim=self.cfg.QFEAT_DIM)
        ecg_qfeat = robust_quality_features_np(ecg_crop, fs=self.cfg.SAMPLE_RATE, qfeat_dim=self.cfg.QFEAT_DIM)
        ppg_norm = zscore_1d(ppg_crop)
        ecg_norm = zscore_1d(ecg_crop)
        y_reg = label
        y_cls = bp_to_risk_class(float(label[0]), float(label[1]))
        num = np.zeros((self.cfg.NUM_DIM,), dtype=np.float32)

        return {
            "ppg": torch.tensor(ppg_norm[None, :], dtype=torch.float32),
            "ecg": torch.tensor(ecg_norm[None, :], dtype=torch.float32),
            "num": torch.tensor(num, dtype=torch.float32),
            "ppg_qfeat": torch.tensor(ppg_qfeat, dtype=torch.float32),
            "ecg_qfeat": torch.tensor(ecg_qfeat, dtype=torch.float32),
            "num_qfeat": torch.zeros((self.cfg.QFEAT_DIM,), dtype=torch.float32),
            "avail": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
            "y_reg": torch.tensor(y_reg, dtype=torch.float32),
            "y_cls": torch.tensor(y_cls, dtype=torch.long),
            "subject_id": sid,
            "seg_idx": int(seg_idx),
        }


# =========================================================
# Model blocks
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualSE1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels)
        self.act = nn.GELU()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return self.act(out)


class MultiScaleStem1D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.b1 = ResidualSE1DBlock(in_channels, 24, stride=2, kernel_size=5)
        self.b2 = ResidualSE1DBlock(in_channels, 24, stride=2, kernel_size=9)
        self.b3 = ResidualSE1DBlock(in_channels, 16, stride=2, kernel_size=15)
        self.merge = nn.Sequential(
            nn.Conv1d(64, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            ResidualSE1DBlock(out_channels, out_channels, stride=2, kernel_size=7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.merge(x)


class AttentivePooling1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1).unsqueeze(-1)
        return torch.sum(weights * x, dim=1)


class PhysiologicalEncoder1D(nn.Module):
    def __init__(self, cfg: CFG, out_dim: int = 128):
        super().__init__()
        self.cfg = cfg
        self.stem = MultiScaleStem1D(1, 64)
        self.patch_proj = nn.Conv1d(
            64,
            cfg.EMBED_DIM,
            kernel_size=cfg.PATCH_SIZE,
            stride=cfg.PATCH_STRIDE,
            padding=cfg.PATCH_SIZE // 2,
            bias=False,
        )
        self.patch_bn = nn.BatchNorm1d(cfg.EMBED_DIM)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.PATCH_TOKENS_MAX, cfg.EMBED_DIM))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.EMBED_DIM,
            nhead=cfg.TRANSFORMER_HEADS,
            dim_feedforward=cfg.EMBED_DIM * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.TRANSFORMER_LAYERS)
        self.pool = AttentivePooling1D(cfg.EMBED_DIM)
        self.fc = nn.Sequential(
            nn.LayerNorm(cfg.EMBED_DIM + 64),
            nn.Linear(cfg.EMBED_DIM + 64, out_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        global_feat = stem.mean(dim=-1)
        patches = self.patch_bn(self.patch_proj(stem)).transpose(1, 2)
        n_tokens = min(patches.shape[1], self.cfg.PATCH_TOKENS_MAX)
        patches = patches[:, :n_tokens, :] + self.pos_embed[:, :n_tokens, :]
        patches = self.transformer(patches)
        pooled = self.pool(patches)
        return self.fc(torch.cat([pooled, global_feat], dim=1))


class NumericEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QualityPredictor(nn.Module):
    def __init__(self, qfeat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(qfeat_dim),
            nn.Linear(qfeat_dim, 32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class Expert(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CredibilityHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(64, 2)
        self.logvar_head = nn.Linear(64, 2)
        nn.init.constant_(self.mean_head.bias[0], 120.0)
        nn.init.constant_(self.mean_head.bias[1], 80.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        mean = self.mean_head(h)
        logvar = torch.clamp(self.logvar_head(h), min=-4.0, max=3.5)
        return mean, logvar


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OrdinalHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseQualityMoEBackbone(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.ppg_enc = PhysiologicalEncoder1D(cfg, cfg.EMBED_DIM)
        self.ecg_enc = PhysiologicalEncoder1D(cfg, cfg.EMBED_DIM)
        self.num_enc = NumericEncoder(cfg.NUM_DIM, cfg.EMBED_DIM)
        self.ppg_q = QualityPredictor(cfg.QFEAT_DIM)
        self.ecg_q = QualityPredictor(cfg.QFEAT_DIM)
        self.num_q = QualityPredictor(cfg.QFEAT_DIM)
        token_dim = cfg.EMBED_DIM + 2
        cross_dim = cfg.EMBED_DIM * 2 + 4
        self.ppg_expert = Expert(token_dim, cfg.EXPERT_DIM)
        self.ecg_expert = Expert(token_dim, cfg.EXPERT_DIM)
        self.joint_expert = Expert(token_dim * 3, cfg.EXPERT_DIM)
        self.cross_expert = Expert(cross_dim, cfg.EXPERT_DIM)
        self.router = nn.Sequential(
            nn.Linear(token_dim * 3 + cross_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 4),
        )

    @staticmethod
    def _topk_softmax(logits: torch.Tensor, k: int = 2, dense_blend: float = 0.0) -> torch.Tensor:
        k = min(k, logits.shape[1])
        topv, topi = torch.topk(logits, k=k, dim=1)
        sparse = torch.full_like(logits, -1e4)
        sparse.scatter_(1, topi, topv)
        sparse_prob = torch.softmax(sparse, dim=1)
        if dense_blend <= 0:
            return sparse_prob
        dense_prob = torch.softmax(logits, dim=1)
        mixed = (1.0 - dense_blend) * sparse_prob + dense_blend * dense_prob
        return mixed / mixed.sum(dim=1, keepdim=True).clamp(min=1e-8)

    def forward(self, ppg, ecg, num, ppg_qfeat, ecg_qfeat, num_qfeat, avail):
        a_ppg = avail[:, 0:1]
        a_ecg = avail[:, 1:2]
        a_num = avail[:, 2:3]

        h_ppg = self.ppg_enc(ppg) * a_ppg
        h_ecg = self.ecg_enc(ecg) * a_ecg
        h_num = self.num_enc(num) * a_num

        q_ppg = self.ppg_q(ppg_qfeat) * a_ppg
        q_ecg = self.ecg_q(ecg_qfeat) * a_ecg
        q_num = self.num_q(num_qfeat) * a_num

        tok_ppg = torch.cat([h_ppg, q_ppg, a_ppg], dim=1)
        tok_ecg = torch.cat([h_ecg, q_ecg, a_ecg], dim=1)
        tok_num = torch.cat([h_num, q_num, a_num], dim=1)
        cross_feat = torch.cat([
            torch.abs(h_ppg - h_ecg),
            h_ppg * h_ecg,
            q_ppg,
            q_ecg,
            a_ppg,
            a_ecg,
        ], dim=1)

        z_ppg = self.ppg_expert(tok_ppg)
        z_ecg = self.ecg_expert(tok_ecg)
        z_joint = self.joint_expert(torch.cat([tok_ppg, tok_ecg, tok_num], dim=1))
        z_cross = self.cross_expert(cross_feat)
        router_in = torch.cat([tok_ppg, tok_ecg, tok_num, cross_feat], dim=1)
        alpha = self._topk_softmax(
            self.router(router_in) / 0.75,
            k=self.cfg.TOPK_EXPERTS,
            dense_blend=self.cfg.ROUTER_DENSE_BLEND,
        )
        z_fused = (
            alpha[:, 0:1] * z_ppg +
            alpha[:, 1:2] * z_ecg +
            alpha[:, 2:3] * z_joint +
            alpha[:, 3:4] * z_cross
        )
        return {
            "h_ppg": h_ppg,
            "h_ecg": h_ecg,
            "h_num": h_num,
            "q_ppg": q_ppg.squeeze(1),
            "q_ecg": q_ecg.squeeze(1),
            "q_num": q_num.squeeze(1),
            "z_ppg": z_ppg,
            "z_ecg": z_ecg,
            "z_joint": z_joint,
            "z_cross": z_cross,
            "z_fused": z_fused,
            "alpha": alpha,
            "avail": avail,
        }


class QMoERegressionNet(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.backbone = SparseQualityMoEBackbone(cfg)
        self.ppg_reg_head = RegressionHead(cfg.EXPERT_DIM)
        self.ecg_reg_head = RegressionHead(cfg.EXPERT_DIM)
        self.fused_reg_head = RegressionHead(cfg.EXPERT_DIM)
        self.ppg_cred = CredibilityHead(cfg.EXPERT_DIM + 4)
        self.ecg_cred = CredibilityHead(cfg.EXPERT_DIM + 4)
        self.fused_cred = CredibilityHead(cfg.EXPERT_DIM + 5)

    def forward(self, ppg, ecg, num, ppg_qfeat, ecg_qfeat, num_qfeat, avail):
        base = self.backbone(ppg, ecg, num, ppg_qfeat, ecg_qfeat, num_qfeat, avail)
        q_ppg = base["q_ppg"].unsqueeze(1)
        q_ecg = base["q_ecg"].unsqueeze(1)
        joint_avail = torch.maximum(avail[:, 0:1], avail[:, 1:2])
        both_avail = torch.minimum(avail[:, 0:1], avail[:, 1:2])

        ppg_mu, ppg_logvar = self.ppg_reg_head(base["z_ppg"])
        ecg_mu, ecg_logvar = self.ecg_reg_head(base["z_ecg"])
        fused_mu, fused_logvar = self.fused_reg_head(base["z_fused"])

        ppg_unc = torch.exp(ppg_logvar).mean(dim=1, keepdim=True)
        ecg_unc = torch.exp(ecg_logvar).mean(dim=1, keepdim=True)
        fused_unc = torch.exp(fused_logvar).mean(dim=1, keepdim=True)
        gate_entropy = -(base["alpha"] * torch.log(base["alpha"] + 1e-8)).sum(dim=1, keepdim=True)

        cred_logits = torch.cat([
            self.ppg_cred(torch.cat([base["z_ppg"], q_ppg, avail[:, 0:1], 1.0 / (ppg_unc + 1e-6), gate_entropy], dim=1)),
            self.ecg_cred(torch.cat([base["z_ecg"], q_ecg, avail[:, 1:2], 1.0 / (ecg_unc + 1e-6), gate_entropy], dim=1)),
            self.fused_cred(torch.cat([base["z_fused"], 0.5 * (q_ppg + q_ecg), joint_avail, 1.0 / (fused_unc + 1e-6), base["alpha"][:, 2:3], gate_entropy], dim=1)),
        ], dim=1)
        cred_mask = torch.cat([avail[:, 0:1], avail[:, 1:2], joint_avail], dim=1)
        cred_logits = cred_logits.masked_fill(cred_mask <= 0, -1e4)
        credibility = torch.softmax(cred_logits / 0.75, dim=1)
        cred_prior = build_credibility_prior_torch(avail, self.cfg.FUSED_CREDIBILITY_PRIOR)
        if self.cfg.CREDIBILITY_SMOOTHING > 0:
            credibility = (1.0 - self.cfg.CREDIBILITY_SMOOTHING) * credibility + self.cfg.CREDIBILITY_SMOOTHING * cred_prior
            credibility = credibility / credibility.sum(dim=1, keepdim=True).clamp(min=1e-8)
        if self.cfg.FUSED_CREDIBILITY_FLOOR > 0:
            fused_floor = self.cfg.FUSED_CREDIBILITY_FLOOR * both_avail
            fused_weight = torch.maximum(credibility[:, 2:3], fused_floor).clamp(max=0.95)
            unimodal = credibility[:, :2]
            unimodal = unimodal / unimodal.sum(dim=1, keepdim=True).clamp(min=1e-8)
            credibility = torch.cat([unimodal * (1.0 - fused_weight), fused_weight], dim=1)

        y_reg = (
            credibility[:, 0:1] * ppg_mu +
            credibility[:, 1:2] * ecg_mu +
            credibility[:, 2:3] * fused_mu
        )
        disagreement = (
            credibility[:, 0:1] * (ppg_mu - y_reg).pow(2) +
            credibility[:, 1:2] * (ecg_mu - y_reg).pow(2) +
            credibility[:, 2:3] * (fused_mu - y_reg).pow(2)
        ).mean(dim=1, keepdim=True)
        uncertainty_proxy = (
            credibility[:, 0:1] * ppg_unc +
            credibility[:, 1:2] * ecg_unc +
            credibility[:, 2:3] * fused_unc
        ) + disagreement
        base.update({
            "ppg_mu": ppg_mu,
            "ppg_logvar": ppg_logvar,
            "ecg_mu": ecg_mu,
            "ecg_logvar": ecg_logvar,
            "fused_mu": fused_mu,
            "fused_logvar": fused_logvar,
            "credibility": credibility,
            "uncertainty_proxy": uncertainty_proxy.squeeze(1),
        })
        return y_reg, base


class QMoEClassificationNet(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.backbone = SparseQualityMoEBackbone(cfg)
        self.ppg_cls_head = ClassificationHead(cfg.EXPERT_DIM, cfg.N_CLASSES)
        self.ecg_cls_head = ClassificationHead(cfg.EXPERT_DIM, cfg.N_CLASSES)
        self.fused_cls_head = ClassificationHead(cfg.EXPERT_DIM, cfg.N_CLASSES)
        self.ord_head = OrdinalHead(cfg.EXPERT_DIM)
        self.ppg_cred = CredibilityHead(cfg.EXPERT_DIM + 4)
        self.ecg_cred = CredibilityHead(cfg.EXPERT_DIM + 4)
        self.fused_cred = CredibilityHead(cfg.EXPERT_DIM + 4)

    def forward(self, ppg, ecg, num, ppg_qfeat, ecg_qfeat, num_qfeat, avail):
        base = self.backbone(ppg, ecg, num, ppg_qfeat, ecg_qfeat, num_qfeat, avail)
        q_ppg = base["q_ppg"].unsqueeze(1)
        q_ecg = base["q_ecg"].unsqueeze(1)
        joint_avail = torch.maximum(avail[:, 0:1], avail[:, 1:2])
        both_avail = torch.minimum(avail[:, 0:1], avail[:, 1:2])
        gate_entropy = -(base["alpha"] * torch.log(base["alpha"] + 1e-8)).sum(dim=1, keepdim=True)

        ppg_logits = self.ppg_cls_head(base["z_ppg"])
        ecg_logits = self.ecg_cls_head(base["z_ecg"])
        fused_logits = self.fused_cls_head(base["z_fused"])
        ord_logits = self.ord_head(base["z_fused"])

        ppg_conf = torch.softmax(ppg_logits, dim=1).max(dim=1, keepdim=True).values
        ecg_conf = torch.softmax(ecg_logits, dim=1).max(dim=1, keepdim=True).values
        fused_conf = torch.softmax(fused_logits, dim=1).max(dim=1, keepdim=True).values
        cred_logits = torch.cat([
            self.ppg_cred(torch.cat([base["z_ppg"], q_ppg, avail[:, 0:1], ppg_conf, gate_entropy], dim=1)),
            self.ecg_cred(torch.cat([base["z_ecg"], q_ecg, avail[:, 1:2], ecg_conf, gate_entropy], dim=1)),
            self.fused_cred(torch.cat([base["z_fused"], 0.5 * (q_ppg + q_ecg), joint_avail, fused_conf, gate_entropy], dim=1)),
        ], dim=1)
        cred_mask = torch.cat([avail[:, 0:1], avail[:, 1:2], joint_avail], dim=1)
        cred_logits = cred_logits.masked_fill(cred_mask <= 0, -1e4)
        credibility = torch.softmax(cred_logits / 0.75, dim=1)
        cred_prior = build_credibility_prior_torch(avail, self.cfg.FUSED_CREDIBILITY_PRIOR)
        if self.cfg.CREDIBILITY_SMOOTHING > 0:
            credibility = (1.0 - self.cfg.CREDIBILITY_SMOOTHING) * credibility + self.cfg.CREDIBILITY_SMOOTHING * cred_prior
            credibility = credibility / credibility.sum(dim=1, keepdim=True).clamp(min=1e-8)
        if self.cfg.FUSED_CREDIBILITY_FLOOR > 0:
            fused_floor = self.cfg.FUSED_CREDIBILITY_FLOOR * both_avail
            fused_weight = torch.maximum(credibility[:, 2:3], fused_floor).clamp(max=0.95)
            unimodal = credibility[:, :2]
            unimodal = unimodal / unimodal.sum(dim=1, keepdim=True).clamp(min=1e-8)
            credibility = torch.cat([unimodal * (1.0 - fused_weight), fused_weight], dim=1)

        logits = (
            credibility[:, 0:1] * ppg_logits +
            credibility[:, 1:2] * ecg_logits +
            credibility[:, 2:3] * fused_logits
        )
        ord_prob = ordinal_logits_to_class_prob(ord_logits)
        prob = 0.70 * torch.softmax(logits, dim=1) + 0.30 * ord_prob
        prob = prob / prob.sum(dim=1, keepdim=True).clamp(min=1e-8)
        bp_proxy_centers = torch.tensor(
            np.stack([np.asarray(self.cfg.CLASS_CENTER_SBP), np.asarray(self.cfg.CLASS_CENTER_DBP)], axis=1),
            dtype=torch.float32,
            device=prob.device,
        )
        bp_proxy = prob @ bp_proxy_centers
        base.update({
            "ppg_logits": ppg_logits,
            "ecg_logits": ecg_logits,
            "fused_logits": fused_logits,
            "ord_logits": ord_logits,
            "credibility": credibility,
            "prob": prob,
            "bp_proxy": bp_proxy,
        })
        return logits, base


# =========================================================
# Loaders / weighting
# =========================================================
def build_loaders(cfg: CFG, task: str = "regression"):
    root = Path(cfg.DATA_ROOT)
    split_dir = root / "splits"
    train_path = split_dir / "train_subjects.txt"
    val_path = split_dir / "val_subjects.txt"
    calib_path = split_dir / "calib_subjects.txt"
    test_path = split_dir / "test_subjects.txt"
    if not all(p.exists() for p in [train_path, val_path, calib_path, test_path]):
        raise FileNotFoundError("Missing split txt files. Run your split generation script first.")
    train_subjects = read_subject_txt(train_path)
    val_subjects = read_subject_txt(val_path)
    calib_subjects = read_subject_txt(calib_path)
    test_subjects = read_subject_txt(test_path)

    ds_train = MIMICBPDataset(cfg, train_subjects, cfg.CROP_LEN, mode="train", seed=cfg.SEED)
    ds_val = MIMICBPDataset(cfg, val_subjects, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)
    ds_calib = MIMICBPDataset(cfg, calib_subjects, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)
    ds_test = MIMICBPDataset(cfg, test_subjects, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)

    pin_memory = cfg.DEVICE == "cuda"
    common_loader_kwargs = {
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": pin_memory,
        "persistent_workers": cfg.NUM_WORKERS > 0,
    }

    if task == "classification":
        counts = np.asarray(ds_train.class_counts, dtype=np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        class_w = 1.0 / np.sqrt(counts)
        sample_w = np.asarray([class_w[c] for c in ds_train.sample_classes], dtype=np.float64)
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
        train_loader = DataLoader(ds_train, batch_size=cfg.BATCH_SIZE, sampler=sampler, **common_loader_kwargs)
    else:
        if cfg.REG_USE_WEIGHTED_SAMPLER:
            counts = np.asarray(ds_train.class_counts, dtype=np.float32)
            counts = np.where(counts == 0, 1.0, counts)
            class_w = 1.0 / np.power(counts, float(cfg.REG_SAMPLER_POWER))
            class_w = class_w / max(class_w.mean(), 1e-6)
            sample_w = np.asarray([class_w[c] for c in ds_train.sample_classes], dtype=np.float64)
            sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
            train_loader = DataLoader(ds_train, batch_size=cfg.BATCH_SIZE, sampler=sampler, **common_loader_kwargs)
        else:
            train_loader = DataLoader(ds_train, batch_size=cfg.BATCH_SIZE, shuffle=True, **common_loader_kwargs)

    val_loader = DataLoader(ds_val, batch_size=cfg.BATCH_SIZE, shuffle=False, **common_loader_kwargs)
    calib_loader = DataLoader(ds_calib, batch_size=cfg.BATCH_SIZE, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(ds_test, batch_size=cfg.BATCH_SIZE, shuffle=False, **common_loader_kwargs)
    return ds_train, train_loader, val_loader, calib_loader, test_loader


def build_class_weights(ds_train: MIMICBPDataset, cfg: CFG, device: str):
    counts = np.asarray(ds_train.class_counts, dtype=np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_range_weights(ds_train: MIMICBPDataset, cfg: CFG, device: str):
    counts = np.asarray(ds_train.class_counts, dtype=np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


# =========================================================
# Training helpers
# =========================================================
def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def apply_asymmetric_modality_dropout(batch, cfg: CFG, epoch: Optional[int] = None):
    batch = clone_batch(batch)
    avail = batch["avail"].clone()
    B = avail.shape[0]
    strength = ramp_factor(epoch, cfg.AUG_RAMP_EPOCHS, cfg.AUG_WARMUP_FACTOR)
    ppg_drop_p = cfg.MODALITY_DROPOUT_PPG * strength
    ecg_drop_p = cfg.MODALITY_DROPOUT_ECG * strength
    for i in range(B):
        if avail[i, 0] > 0 and random.random() < ppg_drop_p:
            avail[i, 0] = 0.0
        if avail[i, 1] > 0 and random.random() < ecg_drop_p:
            avail[i, 1] = 0.0
        if avail[i].sum() == 0:
            avail[i, 0] = 1.0
    batch["ppg"] = batch["ppg"] * avail[:, 0].view(-1, 1, 1)
    batch["ecg"] = batch["ecg"] * avail[:, 1].view(-1, 1, 1)
    batch["ppg_qfeat"] = batch["ppg_qfeat"] * avail[:, 0].view(-1, 1)
    batch["ecg_qfeat"] = batch["ecg_qfeat"] * avail[:, 1].view(-1, 1)
    batch["avail"] = avail
    return batch


def build_router_target(q_ppg_t: torch.Tensor, q_ecg_t: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
    q1 = q_ppg_t * avail[:, 0]
    q2 = q_ecg_t * avail[:, 1]
    only_ppg = ((q1 > 0) & (q2 <= 0)).float()
    only_ecg = ((q2 > 0) & (q1 <= 0)).float()
    both = ((q1 > 0) & (q2 > 0)).float()
    ppg_only = only_ppg * 1.0 + both * (q1 * (1.0 - 0.3 * q2) + 0.05)
    ecg_only = only_ecg * 1.0 + both * (q2 * (1.0 - 0.3 * q1) + 0.05)
    joint = both * (q1 * q2 + 0.10)
    cross = both * (torch.sqrt(torch.clamp(q1 * q2, min=0.0)) * (1.0 - torch.abs(q1 - q2)) + 0.05)
    target = torch.stack([ppg_only, ecg_only, joint, cross], dim=1)
    target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return target


def apply_single_modality_quality_degradation(
    signal_batch: torch.Tensor,
    avail_col: torch.Tensor,
    cfg: CFG,
    epoch: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = signal_batch.clone()
    B, _, L = out.shape
    q_target = avail_col.clone()
    strength = ramp_factor(epoch, cfg.AUG_RAMP_EPOCHS, cfg.AUG_WARMUP_FACTOR)
    degrade_prob = cfg.P_DEGRADE * strength
    severity_max = 0.25 + 0.50 * strength
    for i in range(B):
        if avail_col[i] <= 0:
            q_target[i] = 0.0
            continue
        if random.random() > degrade_prob:
            q_target[i] = 1.0
            continue
        severity = random.uniform(0.08, severity_max)
        x = out[i, 0]
        sig_std = x.std().clamp(min=1e-6)
        op = random.choice(["gaussian", "baseline", "scale", "burst_dropout", "clipping"])
        if op == "gaussian":
            x = x + torch.randn_like(x) * sig_std * (0.08 + 0.24 * severity)
        elif op == "baseline":
            t = torch.linspace(0, 1, L, device=x.device)
            amp = sig_std * (0.04 + 0.24 * severity)
            x = x + amp * torch.sin(2 * math.pi * random.uniform(0.5, 2.0) * t + random.uniform(0, 2 * math.pi))
        elif op == "scale":
            factor = 1.0 + (1.0 if random.random() > 0.5 else -1.0) * (0.08 + 0.35 * severity)
            x = x * factor
        elif op == "burst_dropout":
            seg_len = max(4, int(L * (0.03 + 0.12 * severity)))
            start = random.randint(0, max(0, L - seg_len))
            x = x.clone()
            x[start:start + seg_len] = 0.0
        elif op == "clipping":
            thr = sig_std * max(0.15, (0.92 - 0.45 * severity))
            x = torch.clamp(x, min=-thr, max=thr)
        out[i, 0] = x
        q_target[i] = max(0.08, 1.0 - severity)
    return out, q_target


def apply_training_quality_augmentation(batch: Dict[str, torch.Tensor], cfg: CFG, epoch: Optional[int] = None):
    batch = clone_batch(batch)
    ppg_aug, q_ppg_t = apply_single_modality_quality_degradation(batch["ppg"], batch["avail"][:, 0], cfg, epoch=epoch)
    ecg_aug, q_ecg_t = apply_single_modality_quality_degradation(batch["ecg"], batch["avail"][:, 1], cfg, epoch=epoch)
    batch["ppg"] = ppg_aug
    batch["ecg"] = ecg_aug
    batch["ppg_qfeat"] = tensor_quality_features(batch["ppg"], fs=cfg.SAMPLE_RATE, qfeat_dim=cfg.QFEAT_DIM)
    batch["ecg_qfeat"] = tensor_quality_features(batch["ecg"], fs=cfg.SAMPLE_RATE, qfeat_dim=cfg.QFEAT_DIM)
    q_targets = {
        "q_ppg_t": q_ppg_t,
        "q_ecg_t": q_ecg_t,
        "q_num_t": batch["avail"][:, 2] * 0.0,
    }
    return batch, q_targets


def apply_eval_condition(batch: Dict[str, torch.Tensor], cfg: CFG, drop_modality=None, missing_prob=0.0, noise_std=0.0, batch_seed: int = 0):
    batch = clone_batch(batch)
    rng = np.random.RandomState(batch_seed)
    avail = batch["avail"].clone()
    B = avail.shape[0]
    if missing_prob > 0:
        if drop_modality == "ppg":
            mask = torch.tensor(rng.rand(B) < missing_prob, dtype=torch.bool, device=avail.device)
            avail[mask, 0] = 0.0
        elif drop_modality == "ecg":
            mask = torch.tensor(rng.rand(B) < missing_prob, dtype=torch.bool, device=avail.device)
            avail[mask, 1] = 0.0
    if noise_std > 0:
        if avail[:, 0].sum() > 0:
            batch["ppg"] = batch["ppg"] + torch.randn_like(batch["ppg"]) * noise_std
        if avail[:, 1].sum() > 0:
            batch["ecg"] = batch["ecg"] + torch.randn_like(batch["ecg"]) * noise_std
        batch["ppg_qfeat"] = tensor_quality_features(batch["ppg"], fs=cfg.SAMPLE_RATE, qfeat_dim=cfg.QFEAT_DIM)
        batch["ecg_qfeat"] = tensor_quality_features(batch["ecg"], fs=cfg.SAMPLE_RATE, qfeat_dim=cfg.QFEAT_DIM)
    batch["ppg"] = batch["ppg"] * avail[:, 0].view(-1, 1, 1)
    batch["ecg"] = batch["ecg"] * avail[:, 1].view(-1, 1, 1)
    batch["ppg_qfeat"] = batch["ppg_qfeat"] * avail[:, 0].view(-1, 1)
    batch["ecg_qfeat"] = batch["ecg_qfeat"] * avail[:, 1].view(-1, 1)
    batch["avail"] = avail
    return batch


def heteroscedastic_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    return 0.5 * (inv_var * (mean - target) ** 2 + logvar).mean(dim=1)


def measure_runtime(model: nn.Module, example_batch: Dict[str, torch.Tensor], cfg: CFG):
    model.eval()
    device = cfg.DEVICE
    example_batch = move_batch(example_batch, device)
    if device != "cuda":
        return {
            "n_params_trainable": count_parameters(model),
            "latency_ms_per_batch": float("nan"),
            "latency_ms_per_sample": float("nan"),
            "batch_size_profiled": int(example_batch["ppg"].shape[0]),
            "device": device,
            "flops": "not_computed",
        }
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(cfg.LATENCY_WARMUP):
            _ = model(
                example_batch["ppg"], example_batch["ecg"], example_batch["num"],
                example_batch["ppg_qfeat"], example_batch["ecg_qfeat"], example_batch["num_qfeat"],
                example_batch["avail"],
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(cfg.LATENCY_ITERS):
            _ = model(
                example_batch["ppg"], example_batch["ecg"], example_batch["num"],
                example_batch["ppg_qfeat"], example_batch["ecg_qfeat"], example_batch["num_qfeat"],
                example_batch["avail"],
            )
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / cfg.LATENCY_ITERS
    bsz = int(example_batch["ppg"].shape[0])
    return {
        "n_params_trainable": count_parameters(model),
        "latency_ms_per_batch": float(1000.0 * dt),
        "latency_ms_per_sample": float(1000.0 * dt / bsz),
        "batch_size_profiled": bsz,
        "device": device,
        "flops": "not_computed",
    }


# =========================================================
# Metrics / tables
# =========================================================
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    metrics = {}
    for j, name in enumerate(["sbp", "dbp"]):
        metrics[f"mae_{name}"] = float(np.mean(abs_err[:, j]))
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(err[:, j] ** 2)))
        metrics[f"bias_{name}"] = float(np.mean(err[:, j]))
        metrics[f"sd_error_{name}"] = float(np.std(err[:, j]))
        metrics[f"median_ae_{name}"] = float(np.median(abs_err[:, j]))
        metrics[f"p5_error_{name}"] = float(np.percentile(err[:, j], 5))
        metrics[f"p95_error_{name}"] = float(np.percentile(err[:, j], 95))
        metrics[f"r2_{name}"] = float(r2_score(y_true[:, j], y_pred[:, j])) if np.unique(y_true[:, j]).size > 1 else 0.0
        metrics[f"pearson_{name}"] = safe_corr(y_true[:, j], y_pred[:, j])
        metrics[f"spearman_{name}"] = safe_spearman(y_true[:, j], y_pred[:, j])
        metrics[f"ccc_{name}"] = concordance_corrcoef(y_true[:, j], y_pred[:, j])
        metrics[f"within_5mmhg_{name}"] = float(np.mean(abs_err[:, j] <= 5.0))
        metrics[f"within_10mmhg_{name}"] = float(np.mean(abs_err[:, j] <= 10.0))
        metrics[f"within_15mmhg_{name}"] = float(np.mean(abs_err[:, j] <= 15.0))
    metrics["mae_mean"] = float(0.5 * (metrics["mae_sbp"] + metrics["mae_dbp"]))
    return metrics


def proxy_regression_metrics(y_true_reg: np.ndarray, bp_proxy: np.ndarray, prefix: str = "proxy") -> Dict[str, float]:
    met = regression_metrics(y_true_reg, bp_proxy)
    return {f"{prefix}_{k}": v for k, v in met.items()}


def risk_classification_metrics(y_true_cls: np.ndarray, y_pred_cls: np.ndarray, y_prob: np.ndarray, cfg: CFG, prefix: str = "head") -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_cls, y_pred_cls, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
    per_p, per_r, per_f1, support = precision_recall_fscore_support(y_true_cls, y_pred_cls, labels=list(range(cfg.N_CLASSES)), zero_division=0)
    spec = specificity_per_class(y_true_cls, y_pred_cls, cfg.N_CLASSES)
    metrics = {
        f"cls_acc_{prefix}": float(np.mean(y_true_cls == y_pred_cls)),
        f"cls_balanced_acc_{prefix}": float(balanced_accuracy_score(y_true_cls, y_pred_cls)),
        f"cls_f1_macro_{prefix}": float(macro_f1),
        f"cls_f1_weighted_{prefix}": float(weighted_f1),
        f"cls_precision_macro_{prefix}": float(macro_p),
        f"cls_precision_weighted_{prefix}": float(weighted_p),
        f"cls_recall_macro_{prefix}": float(macro_r),
        f"cls_recall_weighted_{prefix}": float(weighted_r),
        f"cls_kappa_{prefix}": float(cohen_kappa_score(y_true_cls, y_pred_cls)),
        f"cls_mcc_{prefix}": float(matthews_corrcoef(y_true_cls, y_pred_cls)),
        f"cls_ece_{prefix}": expected_calibration_error(y_true_cls, y_prob, n_bins=cfg.ECE_BINS),
        f"cls_brier_{prefix}": multiclass_brier_score(y_true_cls, y_prob, cfg.N_CLASSES),
        f"cls_sensitivity_macro_{prefix}": float(np.mean(per_r)),
        f"cls_specificity_macro_{prefix}": float(np.mean(spec)),
    }
    y_bin = label_binarize(y_true_cls, classes=list(range(cfg.N_CLASSES)))
    for c in range(cfg.N_CLASSES):
        if np.unique(y_bin[:, c]).size < 2:
            metrics[f"cls_auroc_{prefix}_{cfg.CLASS_NAMES[c]}"] = float("nan")
            metrics[f"cls_auprc_{prefix}_{cfg.CLASS_NAMES[c]}"] = float("nan")
        else:
            metrics[f"cls_auroc_{prefix}_{cfg.CLASS_NAMES[c]}"] = float(roc_auc_score(y_bin[:, c], y_prob[:, c]))
            metrics[f"cls_auprc_{prefix}_{cfg.CLASS_NAMES[c]}"] = float(average_precision_score(y_bin[:, c], y_prob[:, c]))
        metrics[f"cls_precision_{prefix}_{cfg.CLASS_NAMES[c]}"] = float(per_p[c])
        metrics[f"cls_recall_{prefix}_{cfg.CLASS_NAMES[c]}"] = float(per_r[c])
        metrics[f"cls_f1_{prefix}_{cfg.CLASS_NAMES[c]}"] = float(per_f1[c])
        metrics[f"cls_specificity_{prefix}_{cfg.CLASS_NAMES[c]}"] = float(spec[c])
        metrics[f"cls_support_{prefix}_{cfg.CLASS_NAMES[c]}"] = int(support[c])
    return metrics


def bp_range_name(sbp: float, dbp: float) -> str:
    if sbp >= 180 or dbp >= 120:
        return "crisis"
    if sbp >= 140 or dbp >= 90:
        return "high"
    if sbp >= 120 or dbp >= 80:
        return "elevated"
    return "normal"


def build_bp_range_table(y_true_reg: np.ndarray, y_pred_reg: np.ndarray) -> List[dict]:
    rows = []
    keys = [bp_range_name(float(sbp), float(dbp)) for sbp, dbp in y_true_reg]
    for key in ["normal", "elevated", "high", "crisis"]:
        m = np.array([k == key for k in keys])
        if m.sum() == 0:
            continue
        met = regression_metrics(y_true_reg[m], y_pred_reg[m])
        rows.append({"bp_range": key, "n": int(m.sum()), **met})
    return rows


def build_subjectwise_error_table(y_true_reg: np.ndarray, y_pred_reg: np.ndarray, subject_ids: List[str]) -> List[dict]:
    uniq = sorted(set(subject_ids))
    rows = []
    for sid in uniq:
        idx = [i for i, s in enumerate(subject_ids) if s == sid]
        yt = y_true_reg[idx]
        yp = y_pred_reg[idx]
        err = np.abs(yp - yt)
        rows.append({
            "subject_id": sid,
            "n_segments": len(idx),
            "mae_sbp": float(np.mean(err[:, 0])),
            "mae_dbp": float(np.mean(err[:, 1])),
            "mean_error_sbp": float(np.mean(yp[:, 0] - yt[:, 0])),
            "mean_error_dbp": float(np.mean(yp[:, 1] - yt[:, 1])),
        })
    return rows


def build_calibration_curve_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 12) -> List[dict]:
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if m.sum() == 0:
            rows.append({"bin": i, "count": 0, "conf_mean": np.nan, "acc_mean": np.nan})
            continue
        rows.append({
            "bin": i,
            "count": int(m.sum()),
            "conf_mean": float(conf[m].mean()),
            "acc_mean": float(acc[m].mean()),
        })
    return rows


def build_conditional_coverage_table(y_true_reg: np.ndarray, low: np.ndarray, high: np.ndarray, quality: np.ndarray, cfg: CFG) -> List[dict]:
    cover = (y_true_reg >= low) & (y_true_reg <= high)
    bp_keys = [bp_range_name(float(sbp), float(dbp)) for sbp, dbp in y_true_reg]
    rows = []
    q1, q2 = np.quantile(quality, [cfg.QUALITY_BINS[0], cfg.QUALITY_BINS[1]])
    groups = {
        "quality_low": quality <= q1,
        "quality_mid": (quality > q1) & (quality <= q2),
        "quality_high": quality > q2,
        "bp_normal": np.array([k == "normal" for k in bp_keys]),
        "bp_elevated": np.array([k == "elevated" for k in bp_keys]),
        "bp_high": np.array([k == "high" for k in bp_keys]),
        "bp_crisis": np.array([k == "crisis" for k in bp_keys]),
    }
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        rows.append({
            "group": name,
            "n": int(mask.sum()),
            "coverage_sbp": float(cover[mask, 0].mean()),
            "coverage_dbp": float(cover[mask, 1].mean()),
            "miw_sbp": float(np.mean(high[mask, 0] - low[mask, 0])),
            "miw_dbp": float(np.mean(high[mask, 1] - low[mask, 1])),
        })
    return rows


def build_selective_table(y_true_cls: np.ndarray, y_prob: np.ndarray) -> List[dict]:
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    rows = []
    for thr in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
        keep = conf >= thr
        if keep.sum() == 0:
            rows.append({"threshold": thr, "coverage": 0.0, "acc": np.nan, "macro_f1": np.nan})
            continue
        rows.append({
            "threshold": thr,
            "coverage": float(keep.mean()),
            "acc": float(np.mean(pred[keep] == y_true_cls[keep])),
            "macro_f1": float(precision_recall_fscore_support(y_true_cls[keep], pred[keep], average="macro", zero_division=0)[2]),
        })
    return rows


# =========================================================
# Evaluation collectors
# =========================================================
@torch.no_grad()
def collect_outputs_regression(model, loader, cfg: CFG, drop_modality=None, missing_prob=0.0, noise_std=0.0):
    model.eval()
    y_true_reg, y_pred_reg = [], []
    y_true_cls, y_pred_cls_from_reg, y_pred_cls_from_reg_hard, y_prob_cls_from_reg = [], [], [], []
    alpha_all, cred_all, quality_all, uncert_all = [], [], [], []
    subject_ids, seg_indices = [], []
    for batch_idx, batch in enumerate(loader):
        subject_ids.extend(list(batch["subject_id"]))
        seg_indices.extend([int(x) for x in batch["seg_idx"]])
        batch = move_batch(batch, cfg.DEVICE)
        batch = apply_eval_condition(batch, cfg, drop_modality=drop_modality, missing_prob=missing_prob, noise_std=noise_std, batch_seed=cfg.SEED + batch_idx)
        pred_reg, aux = model(
            batch["ppg"], batch["ecg"], batch["num"],
            batch["ppg_qfeat"], batch["ecg_qfeat"], batch["num_qfeat"], batch["avail"]
        )
        pred_reg_np = pred_reg.cpu().numpy()
        prob_reg = regression_to_class_prob(pred_reg_np, aux["uncertainty_proxy"].cpu().numpy(), cfg)
        pred_label = prob_reg.argmax(axis=1)
        pred_label_hard = reg_to_class_np(pred_reg_np)
        y_true_reg.append(batch["y_reg"].cpu().numpy())
        y_pred_reg.append(pred_reg_np)
        y_true_cls.append(batch["y_cls"].cpu().numpy())
        y_pred_cls_from_reg.append(pred_label)
        y_pred_cls_from_reg_hard.append(pred_label_hard)
        y_prob_cls_from_reg.append(prob_reg)
        alpha_all.append(aux["alpha"].cpu().numpy())
        cred_all.append(aux["credibility"].cpu().numpy())
        quality_all.append((0.5 * (aux["q_ppg"] + aux["q_ecg"])).cpu().numpy())
        uncert_all.append(aux["uncertainty_proxy"].cpu().numpy())
    y_true_reg = np.concatenate(y_true_reg, axis=0)
    y_pred_reg = np.concatenate(y_pred_reg, axis=0)
    y_true_cls = np.concatenate(y_true_cls, axis=0)
    y_pred_cls_from_reg = np.concatenate(y_pred_cls_from_reg, axis=0)
    y_pred_cls_from_reg_hard = np.concatenate(y_pred_cls_from_reg_hard, axis=0)
    y_prob_cls_from_reg = np.concatenate(y_prob_cls_from_reg, axis=0)
    alpha_all = np.concatenate(alpha_all, axis=0)
    cred_all = np.concatenate(cred_all, axis=0)
    quality_all = np.concatenate(quality_all, axis=0)
    uncert_all = np.concatenate(uncert_all, axis=0)
    reg_met = regression_metrics(y_true_reg, y_pred_reg)
    cls_met = risk_classification_metrics(y_true_cls, y_pred_cls_from_reg, y_prob_cls_from_reg, cfg, prefix="from_reg")
    cls_hard_met = risk_classification_metrics(y_true_cls, y_pred_cls_from_reg_hard, y_prob_cls_from_reg, cfg, prefix="from_reg_hard")
    abs_err_mean = np.abs(y_pred_reg - y_true_reg).mean(axis=1)
    unc_met = {
        "uncertainty_error_corr_pearson": safe_corr(uncert_all, abs_err_mean),
        "uncertainty_error_corr_spearman": safe_spearman(uncert_all, abs_err_mean),
    }
    return {
        "y_true_reg": y_true_reg,
        "y_pred_reg": y_pred_reg,
        "y_true_cls": y_true_cls,
        "y_pred_cls_from_reg": y_pred_cls_from_reg,
        "y_pred_cls_from_reg_hard": y_pred_cls_from_reg_hard,
        "y_prob_cls_from_reg": y_prob_cls_from_reg,
        "alpha": alpha_all,
        "credibility": cred_all,
        "quality": quality_all,
        "uncertainty": uncert_all,
        "subject_ids": subject_ids,
        "seg_indices": seg_indices,
        "metrics_reg": reg_met,
        "metrics_cls_from_reg": cls_met,
        "metrics_cls_from_reg_hard": cls_hard_met,
        "uncertainty_metrics": unc_met,
    }


@torch.no_grad()
def collect_outputs_classification(model, loader, cfg: CFG, drop_modality=None, missing_prob=0.0, noise_std=0.0):
    model.eval()
    y_true_reg, y_true_cls = [], []
    y_prob_all, y_pred_all, bp_proxy_all = [], [], []
    alpha_all, cred_all, quality_all = [], [], []
    subject_ids, seg_indices = [], []
    for batch_idx, batch in enumerate(loader):
        subject_ids.extend(list(batch["subject_id"]))
        seg_indices.extend([int(x) for x in batch["seg_idx"]])
        batch = move_batch(batch, cfg.DEVICE)
        batch = apply_eval_condition(batch, cfg, drop_modality=drop_modality, missing_prob=missing_prob, noise_std=noise_std, batch_seed=cfg.SEED + batch_idx)
        logits, aux = model(
            batch["ppg"], batch["ecg"], batch["num"],
            batch["ppg_qfeat"], batch["ecg_qfeat"], batch["num_qfeat"], batch["avail"]
        )
        prob = aux["prob"].cpu().numpy()
        pred = prob.argmax(axis=1)
        y_true_reg.append(batch["y_reg"].cpu().numpy())
        y_true_cls.append(batch["y_cls"].cpu().numpy())
        y_prob_all.append(prob)
        y_pred_all.append(pred)
        bp_proxy_all.append(aux["bp_proxy"].cpu().numpy())
        alpha_all.append(aux["alpha"].cpu().numpy())
        cred_all.append(aux["credibility"].cpu().numpy())
        quality_all.append((0.5 * (aux["q_ppg"] + aux["q_ecg"])).cpu().numpy())
    y_true_reg = np.concatenate(y_true_reg, axis=0)
    y_true_cls = np.concatenate(y_true_cls, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    bp_proxy_all = np.concatenate(bp_proxy_all, axis=0)
    alpha_all = np.concatenate(alpha_all, axis=0)
    cred_all = np.concatenate(cred_all, axis=0)
    quality_all = np.concatenate(quality_all, axis=0)
    cls_met = risk_classification_metrics(y_true_cls, y_pred_all, y_prob_all, cfg, prefix="cls")
    proxy_met = proxy_regression_metrics(y_true_reg, bp_proxy_all, prefix="proxy")
    return {
        "y_true_reg": y_true_reg,
        "y_true_cls": y_true_cls,
        "y_prob_cls": y_prob_all,
        "y_pred_cls": y_pred_all,
        "bp_proxy": bp_proxy_all,
        "alpha": alpha_all,
        "credibility": cred_all,
        "quality": quality_all,
        "subject_ids": subject_ids,
        "seg_indices": seg_indices,
        "metrics_cls": cls_met,
        "metrics_proxy": proxy_met,
    }


# =========================================================
# Conformal for regression
# =========================================================
def conformal_from_outputs(calib_out: dict, test_out: dict, alpha: float = 0.10):
    calib_err = np.abs(calib_out["y_pred_reg"] - calib_out["y_true_reg"])
    calib_scale = np.sqrt(np.clip(calib_out["uncertainty"], 1e-6, None)).reshape(-1, 1)
    scores = calib_err / np.clip(calib_scale, 1e-6, None)
    q = np.quantile(scores, 1.0 - alpha, axis=0, method="higher")
    test_scale = np.sqrt(np.clip(test_out["uncertainty"], 1e-6, None)).reshape(-1, 1)
    half_width = q.reshape(1, 2) * test_scale
    low = test_out["y_pred_reg"] - half_width
    high = test_out["y_pred_reg"] + half_width
    cover = (test_out["y_true_reg"] >= low) & (test_out["y_true_reg"] <= high)
    met = {
        "coverage_sbp": float(cover[:, 0].mean()),
        "coverage_dbp": float(cover[:, 1].mean()),
        "miw_sbp": float(np.mean(high[:, 0] - low[:, 0])),
        "miw_dbp": float(np.mean(high[:, 1] - low[:, 1])),
    }
    return low, high, met


# =========================================================
# Saving helpers
# =========================================================
def ensure_out_dirs(cfg: CFG):
    out_root = Path(cfg.PROJECT_ROOT) / "outputs" / cfg.OUTPUT_NAME
    figures_dir = out_root / "figures"
    artifacts_dir = out_root / "artifacts"
    tables_dir = out_root / "tables"
    out_root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return out_root, figures_dir, artifacts_dir, tables_dir


def save_epoch_log(csv_path: Path, rows: List[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_rows_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_regression_npz(path: Path, obj: dict):
    np.savez_compressed(
        path,
        y_true_reg=obj["y_true_reg"],
        y_pred_reg=obj["y_pred_reg"],
        y_true_cls=obj["y_true_cls"],
        y_pred_cls_from_reg=obj["y_pred_cls_from_reg"],
        y_prob_cls_from_reg=obj["y_prob_cls_from_reg"],
        alpha=obj["alpha"],
        credibility=obj["credibility"],
        quality=obj["quality"],
        uncertainty=obj["uncertainty"],
        subject_ids=np.array(obj["subject_ids"], dtype=object),
        seg_indices=np.array(obj["seg_indices"], dtype=np.int64),
    )


def save_classification_npz(path: Path, obj: dict):
    np.savez_compressed(
        path,
        y_true_reg=obj["y_true_reg"],
        y_true_cls=obj["y_true_cls"],
        y_prob_cls=obj["y_prob_cls"],
        y_pred_cls=obj["y_pred_cls"],
        bp_proxy=obj["bp_proxy"],
        alpha=obj["alpha"],
        credibility=obj["credibility"],
        quality=obj["quality"],
        subject_ids=np.array(obj["subject_ids"], dtype=object),
        seg_indices=np.array(obj["seg_indices"], dtype=np.int64),
    )


# =========================================================
# Plotting helpers
# =========================================================
def plot_training_curves_reg(epoch_rows: List[dict], fig_dir: Path):
    epochs = [r["epoch"] for r in epoch_rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].plot(epochs, [r["train_loss"] for r in epoch_rows], marker="o")
    axes[0, 0].set_title("Training Total Loss")
    axes[0, 1].plot(epochs, [r["loss_reg"] for r in epoch_rows], marker="o", label="Reg")
    axes[0, 1].plot(epochs, [r["loss_nll"] for r in epoch_rows], marker="s", label="NLL")
    axes[0, 1].plot(epochs, [r["loss_q"] for r in epoch_rows], marker="^", label="Q")
    axes[0, 1].plot(epochs, [r["loss_router"] for r in epoch_rows], marker="d", label="Router")
    axes[0, 1].legend()
    axes[0, 1].set_title("Main Loss Components")
    axes[0, 2].plot(epochs, [r["loss_ecg_aux"] for r in epoch_rows], marker="o", label="ECG aux")
    axes[0, 2].plot(epochs, [r["loss_bal"] for r in epoch_rows], marker="s", label="Balance")
    axes[0, 2].plot(epochs, [r["grad_norm"] for r in epoch_rows], marker="^")
    axes[0, 2].set_title("Auxiliary / Grad")
    axes[1, 0].plot(epochs, [r["mae_sbp"] for r in epoch_rows], marker="o", label="SBP MAE")
    axes[1, 0].plot(epochs, [r["mae_dbp"] for r in epoch_rows], marker="s", label="DBP MAE")
    axes[1, 0].legend(); axes[1, 0].set_title("Validation MAE")
    axes[1, 1].plot(epochs, [r["cls_f1_macro_from_reg"] for r in epoch_rows], marker="o")
    axes[1, 1].set_title("Validation Macro-F1 (from reg)")
    axes[1, 2].plot(epochs, [r["uncertainty_error_corr_pearson"] for r in epoch_rows], marker="o")
    axes[1, 2].set_title("Uncertainty-Error Corr")
    fig.tight_layout(); fig.savefig(fig_dir / "training_curves.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_training_curves_cls(epoch_rows: List[dict], fig_dir: Path):
    epochs = [r["epoch"] for r in epoch_rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].plot(epochs, [r["train_loss"] for r in epoch_rows], marker="o")
    axes[0, 0].set_title("Training Total Loss")
    axes[0, 1].plot(epochs, [r["loss_cls"] for r in epoch_rows], marker="o", label="Focal")
    axes[0, 1].plot(epochs, [r["loss_ord"] for r in epoch_rows], marker="s", label="Ordinal")
    axes[0, 1].plot(epochs, [r["loss_proxy"] for r in epoch_rows], marker="^")
    axes[0, 1].legend(); axes[0, 1].set_title("Task Losses")
    axes[0, 2].plot(epochs, [r["loss_q"] for r in epoch_rows], marker="o", label="Q")
    axes[0, 2].plot(epochs, [r["loss_router"] for r in epoch_rows], marker="s", label="Router")
    axes[0, 2].plot(epochs, [r["loss_bal"] for r in epoch_rows], marker="^")
    axes[0, 2].legend(); axes[0, 2].set_title("Regularizers")
    axes[1, 0].plot(epochs, [r["cls_f1_macro_cls"] for r in epoch_rows], marker="o", label="Macro-F1")
    axes[1, 0].plot(epochs, [r["cls_balanced_acc_cls"] for r in epoch_rows], marker="s", label="Bal-Acc")
    axes[1, 0].legend(); axes[1, 0].set_title("Validation Classification")
    axes[1, 1].plot(epochs, [r["proxy_mae_sbp"] for r in epoch_rows], marker="o", label="Proxy SBP MAE")
    axes[1, 1].plot(epochs, [r["proxy_mae_dbp"] for r in epoch_rows], marker="s", label="Proxy DBP MAE")
    axes[1, 1].legend(); axes[1, 1].set_title("Validation BP Proxy")
    axes[1, 2].plot(epochs, [r["grad_norm"] for r in epoch_rows], marker="o")
    axes[1, 2].set_title("Gradient Norm")
    fig.tight_layout(); fig.savefig(fig_dir / "training_curves.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_scatter_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, fig_dir: Path, filename: str = "scatter_true_vs_pred.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for j, name in enumerate(["SBP", "DBP"]):
        axes[j].scatter(y_true[:, j], y_pred[:, j], s=10, alpha=0.5)
        lo = min(y_true[:, j].min(), y_pred[:, j].min())
        hi = max(y_true[:, j].max(), y_pred[:, j].max())
        axes[j].plot([lo, hi], [lo, hi], linestyle="--")
        axes[j].set_title(f"{name}: Reference-Estimate Agreement")
        axes[j].set_xlabel(f"True {name}")
        axes[j].set_ylabel(f"Pred {name}")
    fig.tight_layout(); fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_bland_altman(y_true: np.ndarray, y_pred: np.ndarray, fig_dir: Path, filename: str = "bland_altman.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for j, name in enumerate(["SBP", "DBP"]):
        mean_val = (y_true[:, j] + y_pred[:, j]) / 2.0
        diff = y_pred[:, j] - y_true[:, j]
        md = diff.mean(); sd = diff.std()
        axes[j].scatter(mean_val, diff, s=10, alpha=0.5)
        axes[j].axhline(md, linestyle="--", label=f"Mean={md:.2f}")
        axes[j].axhline(md + 1.96 * sd, linestyle=":", label=f"+1.96SD={md + 1.96 * sd:.2f}")
        axes[j].axhline(md - 1.96 * sd, linestyle=":", label=f"-1.96SD={md - 1.96 * sd:.2f}")
        axes[j].set_title(f"{name}: Bland-Altman Analysis")
        axes[j].set_xlabel(f"Mean(True, Pred) {name}")
        axes[j].set_ylabel("Pred - True")
        axes[j].legend()
    fig.tight_layout(); fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    fig_dir: Path,
    filename: str,
    title: Optional[str] = None,
    normalize: bool = False,
):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).astype(np.float32)
    if normalize:
        row_sum = np.clip(cm.sum(axis=1, keepdims=True), 1.0, None)
        cm = cm / row_sum
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title or ("Row-normalized Confusion Matrix" if normalize else "Confusion Matrix"))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            label = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, label, ha="center", va="center")
    fig.tight_layout(); fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight"); plt.close(fig)


def _pretty_eval_prefix(prefix: str) -> str:
    mapping = {
        "selected_final": "final operating point",
        "stability_selected": "robust operating point",
        "selected": "final operating point",
        "stability": "robust operating point",
    }
    for needle, pretty in mapping.items():
        if needle in str(prefix):
            return pretty
    return str(prefix).replace("_", " ")


def plot_roc_pr(y_true_cls: np.ndarray, y_prob: np.ndarray, cfg: CFG, fig_dir: Path, prefix: str):
    y_bin = label_binarize(y_true_cls, classes=list(range(cfg.N_CLASSES)))
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    for c in range(cfg.N_CLASSES):
        if np.unique(y_bin[:, c]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{cfg.CLASS_NAMES[c]} (AUC={roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_title(f"Receiver Operating Characteristic: {_pretty_eval_prefix(prefix).title()}")
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate"); ax_roc.legend()
    fig_roc.tight_layout(); fig_roc.savefig(fig_dir / f"roc_curve_{prefix}.png", dpi=300, bbox_inches="tight"); plt.close(fig_roc)

    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    for c in range(cfg.N_CLASSES):
        if np.unique(y_bin[:, c]).size < 2:
            continue
        p, r, _ = precision_recall_curve(y_bin[:, c], y_prob[:, c])
        ap = average_precision_score(y_bin[:, c], y_prob[:, c])
        ax_pr.plot(r, p, label=f"{cfg.CLASS_NAMES[c]} (AP={ap:.3f})")
    ax_pr.set_title(f"Precision-Recall Analysis: {_pretty_eval_prefix(prefix).title()}")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision"); ax_pr.legend()
    fig_pr.tight_layout(); fig_pr.savefig(fig_dir / f"pr_curve_{prefix}.png", dpi=300, bbox_inches="tight"); plt.close(fig_pr)


def plot_calibration(cal_rows: List[dict], fig_dir: Path, filename: str = "calibration_curve.png"):
    x = [r["conf_mean"] for r in cal_rows if not np.isnan(r["conf_mean"])]
    y = [r["acc_mean"] for r in cal_rows if not np.isnan(r["acc_mean"])]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--")
    if x:
        ax.plot(x, y, marker="o")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title("Probability Calibration Diagram")
    fig.tight_layout(); fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_router_heatmap(alpha: np.ndarray, y_true_reg: Optional[np.ndarray], fig_dir: Path, expert_names: List[str], filename: str = "router_heatmap.png"):
    mat = alpha.copy()
    if y_true_reg is not None:
        order = np.argsort(y_true_reg[:, 0])
        mat = mat[order]
    n_show = min(150, mat.shape[0])
    mat = mat[:n_show]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(expert_names))); ax.set_xticklabels(expert_names)
    ax.set_ylabel("Test samples")
    ax.set_title("Expert Routing Allocation Matrix")
    plt.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_noise_robustness(noise_rows: List[dict], fig_dir: Path):
    x = [r["noise_std"] for r in noise_rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if "mae_sbp" in noise_rows[0]:
        axes[0].plot(x, [r["mae_sbp"] for r in noise_rows], marker="o", label="SBP MAE")
        axes[0].plot(x, [r["mae_dbp"] for r in noise_rows], marker="s", label="DBP MAE")
        axes[0].set_ylabel("MAE")
        axes[0].legend(); axes[0].set_title("Noise Stress Test: Regression")
        f1_key = "cls_f1_macro_from_reg" if "cls_f1_macro_from_reg" in noise_rows[0] else "cls_f1_macro_cls"
        acc_key = "cls_acc_from_reg" if "cls_acc_from_reg" in noise_rows[0] else "cls_acc_cls"
        axes[1].plot(x, [r[f1_key] for r in noise_rows], marker="o", label="Macro-F1")
        axes[1].plot(x, [r[acc_key] for r in noise_rows], marker="s", label="Accuracy")
        axes[1].legend(); axes[1].set_title("Noise Stress Test: Classification")
    else:
        axes[0].plot(x, [r["proxy_mae_sbp"] for r in noise_rows], marker="o")
        axes[0].plot(x, [r["proxy_mae_dbp"] for r in noise_rows], marker="s")
        axes[1].plot(x, [r["cls_f1_macro_cls"] for r in noise_rows], marker="o")
    for ax in axes:
        ax.set_xlabel("Gaussian noise std")
    fig.tight_layout(); fig.savefig(fig_dir / "noise_robustness_curve.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_sharpness_vs_coverage(rows: List[dict], fig_dir: Path):
    x = [0.5 * (r["miw_sbp"] + r["miw_dbp"]) for r in rows]
    y = [0.5 * (r["coverage_sbp"] + r["coverage_dbp"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Sharpness proxy (mean interval width)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Sharpness-Coverage Operating Profile")
    fig.tight_layout(); fig.savefig(fig_dir / "sharpness_vs_coverage.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_quality_conditional_coverage(cond_rows: List[dict], fig_dir: Path):
    subset = [r for r in cond_rows if r["group"].startswith("quality_")]
    if not subset:
        return
    labels = [r["group"] for r in subset]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.15, [r["coverage_sbp"] for r in subset], width=0.3, label="SBP")
    ax.bar(x + 0.15, [r["coverage_dbp"] for r in subset], width=0.3, label="DBP")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Coverage"); ax.set_ylim(0, 1)
    ax.set_title("Coverage Across Signal-Quality Strata")
    ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "quality_conditional_coverage.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_uncertainty_error_corr(uncert: np.ndarray, abs_err: np.ndarray, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(uncert, abs_err, s=20, alpha=0.4)
    ax.set_xlabel("Uncertainty proxy")
    ax.set_ylabel("Mean absolute error per sample")
    ax.set_title("Uncertainty-Error Association")
    fig.tight_layout(); fig.savefig(fig_dir / "uncertainty_error_correlation.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_true_pred_hexbin(y_true: np.ndarray, y_pred: np.ndarray, fig_dir: Path, filename: str = "true_pred_hexbin.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for j, name in enumerate(["SBP", "DBP"]):
        hb = axes[j].hexbin(y_true[:, j], y_pred[:, j], gridsize=42, cmap="viridis", mincnt=1)
        lo = min(y_true[:, j].min(), y_pred[:, j].min())
        hi = max(y_true[:, j].max(), y_pred[:, j].max())
        axes[j].plot([lo, hi], [lo, hi], linestyle="--", color="#f4d03f", linewidth=1.2)
        axes[j].set_title(f"{name}: Prediction Density Map")
        axes[j].set_xlabel(f"Reference {name}")
        axes[j].set_ylabel(f"Estimated {name}")
        fig.colorbar(hb, ax=axes[j], fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selective_curve(selective_rows: List[dict], fig_dir: Path):
    coverage = [r["coverage"] for r in selective_rows]
    macro_f1 = [r["macro_f1"] for r in selective_rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverage, macro_f1, marker="o")
    ax.set_xlabel("Retained fraction")
    ax.set_ylabel("Macro-F1 on retained set")
    ax.set_title("Selective Classification Curve")
    fig.tight_layout(); fig.savefig(fig_dir / "selective_curve.png", dpi=300, bbox_inches="tight"); plt.close(fig)
