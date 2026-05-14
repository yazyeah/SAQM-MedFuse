from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import aqm_bp_shared_v9 as shared_v9
import aqm_bp_shared_v10_2_multi_protocol as multi_protocol
import aqm_bp_shared_v11_protocol as v11
import train_aqm_medfuse_mimic_bp_reg_v10_2_common as common_script


METHODS: Dict[str, dict] = {
    "ann_lstm_ecg_ppg": {
        "paper": "Cuffless blood pressure estimation from ECG and PPG using waveform based ANN-LSTM network",
        "venue": "Biomedical Signal Processing and Control, 2019",
        "url": "https://doi.org/10.1016/j.bspc.2019.02.028",
        "modality": "ecg_ppg",
    },
    "ppg_bilstm_attention": {
        "paper": "Deep learning models for cuffless blood pressure monitoring from PPG signals using attention mechanism",
        "venue": "Biomedical Signal Processing and Control, 2021",
        "url": "https://doi.org/10.1016/j.bspc.2020.102301",
        "modality": "ppg",
    },
    "bpnet_cnn": {
        "paper": "BP-Net: Cuff-less and non-invasive blood pressure estimation via a generic deep convolutional architecture",
        "venue": "Biomedical Signal Processing and Control, 2022",
        "url": "https://doi.org/10.1016/j.bspc.2022.103850",
        "modality": "ecg_ppg",
    },
    "mlp_bp_mixer": {
        "paper": "MLP-BP: cuffless BP measurement with PPG and ECG based on MLP-Mixer neural networks",
        "venue": "Biomedical Signal Processing and Control, 2022",
        "url": "https://doi.org/10.1016/j.bspc.2021.103404",
        "modality": "ecg_ppg",
    },
    "piso_transformer": {
        "paper": "PiSO model-selection branch: CNN/LSTM/Transformer DL models for ABP estimation on MIMIC-IV",
        "venue": "IEEE Access, 2026",
        "url": "https://doi.org/10.1109/ACCESS.2026.3665255",
        "modality": "ecg_ppg",
    },
    "mufubp_dual_feature_pfe": {
        "paper": "MuFuBP-Net: multimodal fusion with dual-feature pipeline and probabilistic feature encoder",
        "venue": "IEEE Journal of Biomedical and Health Informatics, 2025",
        "url": "https://doi.org/10.1109/JBHI.2025.3563852",
        "modality": "ecg_ppg",
    },
}

DEFAULT_PROTOCOL_SOURCE = "mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_proto"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_entries(path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((str(row["subject_id"]), int(row["seg_idx"])))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_rows_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if not rows and fieldnames is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def resolve_data_root(raw: str) -> str:
    candidates = [
        Path(os.environ.get("AQM_MIMIC_BP_ROOT", "")),
        Path(os.environ.get("AQM_MIMIC_BP_DATA_ROOT", "")),
        Path(raw),
        Path(__file__).resolve().parent / "data" / "raw" / "MIMIC-BP",
    ]
    for path in candidates:
        if str(path) and (path / "ppg").exists() and (path / "ecg").exists() and (path / "labels").exists():
            return str(path)
    return raw


def make_cfg(args: argparse.Namespace):
    cfg = multi_protocol.make_subject_disjoint_fewshot_cfg()
    cfg.PROJECT_ROOT = str(Path(__file__).resolve().parent)
    cfg.DATA_ROOT = resolve_data_root(str(args.data_root or cfg.DATA_ROOT))
    cfg.OUTPUT_NAME = args.output_name
    cfg.PROTOCOL_ID = f"v10.13_compare_{args.method}"
    cfg.PROTOCOL_NAME = f"v10.13 comparison baseline: {args.method}"
    cfg.BATCH_SIZE = int(args.batch_size)
    cfg.NUM_WORKERS = int(args.num_workers)
    cfg.DEVICE = args.device
    return cfg


def build_datasets(cfg, protocol_source: str):
    source_dir = Path(cfg.PROJECT_ROOT) / "outputs" / protocol_source
    train_entries = read_entries(source_dir / "train_entries.csv")
    val_entries = read_entries(source_dir / "val_query_entries.csv")
    test_entries = read_entries(source_dir / "test_query_entries.csv")
    calib_entries = read_entries(source_dir / "test_support_entries.csv")
    ds_train = v11.IndexedMIMICBPDataset(cfg, train_entries, cfg.CROP_LEN, mode="train", seed=cfg.SEED)
    ds_val = v11.IndexedMIMICBPDataset(cfg, val_entries, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)
    ds_test = v11.IndexedMIMICBPDataset(cfg, test_entries, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)
    ds_calib = v11.IndexedMIMICBPDataset(cfg, calib_entries, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)
    return source_dir, ds_train, ds_val, ds_test, ds_calib


def class_weights_from_dataset(ds, n_classes: int, power: float, device: str) -> torch.Tensor:
    counts = np.asarray(ds.class_counts, dtype=np.float32)
    counts = np.where(counts <= 0, 1.0, counts)
    weights = 1.0 / np.power(counts, power)
    weights = weights / max(float(weights.mean()), 1.0e-6)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_train_loader(ds, cfg, power: float) -> DataLoader:
    counts = np.asarray(ds.class_counts, dtype=np.float32)
    counts = np.where(counts <= 0, 1.0, counts)
    weights = 1.0 / np.power(counts, power)
    weights = weights / max(float(weights.mean()), 1.0e-6)
    sample_w = np.asarray([weights[int(c)] for c in ds.sample_classes], dtype=np.float64)
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    return DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.DEVICE == "cuda",
    )


def make_eval_loader(ds, cfg) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.DEVICE == "cuda",
    )


class AttentivePool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1).unsqueeze(-1)
        return torch.sum(w * x, dim=1)


class ConvFeatureMap(nn.Module):
    def __init__(self, in_ch: int, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 48, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 80, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(80),
            nn.GELU(),
            nn.Conv1d(80, dim, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).transpose(1, 2)


class OutputHeads(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.reg = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.15), nn.Linear(dim, 2))
        self.cls = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(0.15), nn.Linear(dim // 2, n_classes))
        nn.init.constant_(self.reg[-1].bias[0], 120.0)
        nn.init.constant_(self.reg[-1].bias[1], 80.0)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reg(h), self.cls(h)


class AnnLstmECGPPG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.frame = nn.Conv1d(2, 96, 25, stride=10, padding=12)
        self.low = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 160), nn.GELU(), nn.Linear(160, 128), nn.GELU())
        self.lstm = nn.LSTM(128, 96, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
        self.pool = AttentivePool(192)
        self.heads = OutputHeads(192, cfg.N_CLASSES)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        x = torch.cat([batch["ppg"], batch["ecg"]], dim=1)
        z = self.frame(x).transpose(1, 2)
        z = self.low(z)
        z, _ = self.lstm(z)
        h = self.pool(z)
        reg, cls = self.heads(h)
        return reg, cls, {}


class PPGBiLSTMAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = ConvFeatureMap(1, 96)
        self.lstm = nn.LSTM(96, 96, num_layers=2, batch_first=True, bidirectional=True, dropout=0.12)
        self.pool = AttentivePool(192)
        self.heads = OutputHeads(192, cfg.N_CLASSES)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        z = self.conv(batch["ppg"])
        z, _ = self.lstm(z)
        reg, cls = self.heads(self.pool(z))
        return reg, cls, {}


class BPNetCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 48, 11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 96, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 128, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 160, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(160),
            nn.GELU(),
        )
        self.pool = AttentivePool(160)
        self.heads = OutputHeads(160, cfg.N_CLASSES)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        x = torch.cat([batch["ppg"], batch["ecg"]], dim=1)
        h = self.pool(self.conv(x).transpose(1, 2))
        reg, cls = self.heads(h)
        return reg, cls, {}


class MixerBlock(nn.Module):
    def __init__(self, n_tokens: int, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(nn.Linear(n_tokens, n_tokens * 2), nn.GELU(), nn.Linear(n_tokens * 2, n_tokens))
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)
        x = x + self.token_mlp(y).transpose(1, 2)
        x = x + self.channel_mlp(self.norm2(x))
        return x


class MLPBPMixer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        patch = 25
        stride = 25
        n_tokens = 1 + max(0, (int(cfg.CROP_LEN) - patch) // stride)
        dim = 128
        self.embed = nn.Conv1d(2, dim, kernel_size=patch, stride=stride)
        self.blocks = nn.Sequential(*[MixerBlock(n_tokens, dim) for _ in range(4)])
        self.heads = OutputHeads(dim, cfg.N_CLASSES)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        x = torch.cat([batch["ppg"], batch["ecg"]], dim=1)
        z = self.embed(x).transpose(1, 2)
        z = self.blocks(z).mean(dim=1)
        reg, cls = self.heads(z)
        return reg, cls, {}


class PiSOTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = 128
        self.patch = nn.Conv1d(2, dim, kernel_size=25, stride=16, padding=4)
        self.pos = nn.Parameter(torch.zeros(1, 96, dim))
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=256, dropout=0.12, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.pool = AttentivePool(dim)
        self.heads = OutputHeads(dim, cfg.N_CLASSES)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        x = torch.cat([batch["ppg"], batch["ecg"]], dim=1)
        z = self.patch(x).transpose(1, 2)
        z = z + self.pos[:, : z.shape[1], :]
        z = self.encoder(z)
        reg, cls = self.heads(self.pool(z))
        return reg, cls, {}


class MuFuBPDualPFE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = 128
        self.ppg = ConvFeatureMap(1, dim)
        self.ecg = ConvFeatureMap(1, dim)
        self.ppg_pool = AttentivePool(dim)
        self.ecg_pool = AttentivePool(dim)
        qdim = int(getattr(cfg, "QFEAT_DIM", 8))
        self.q_gate = nn.Sequential(nn.Linear(qdim * 2, 64), nn.GELU(), nn.Linear(64, 2), nn.Sigmoid())
        self.cross = nn.MultiheadAttention(dim, num_heads=4, dropout=0.10, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(dim * 4, 256), nn.GELU(), nn.Dropout(0.12), nn.Linear(256, 160), nn.GELU())
        self.mu = nn.Linear(160, 160)
        self.logvar = nn.Linear(160, 160)
        self.heads = OutputHeads(160, cfg.N_CLASSES)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        ppg_seq = self.ppg(batch["ppg"])
        ecg_seq = self.ecg(batch["ecg"])
        cross_ppg, _ = self.cross(ppg_seq, ecg_seq, ecg_seq)
        cross_ecg, _ = self.cross(ecg_seq, ppg_seq, ppg_seq)
        ppg_h = self.ppg_pool(ppg_seq + cross_ppg)
        ecg_h = self.ecg_pool(ecg_seq + cross_ecg)
        gates = self.q_gate(torch.cat([batch["ppg_qfeat"], batch["ecg_qfeat"]], dim=1))
        ppg_h = ppg_h * gates[:, 0:1]
        ecg_h = ecg_h * gates[:, 1:2]
        h = self.fuse(torch.cat([ppg_h, ecg_h, torch.abs(ppg_h - ecg_h), ppg_h * ecg_h], dim=1))
        mu = self.mu(h)
        if self.training:
            std = torch.exp(0.5 * torch.clamp(self.logvar(h), -6.0, 3.0))
            h = mu + torch.randn_like(std) * std
        else:
            h = mu
        reg, cls = self.heads(h)
        return reg, cls, {"gate_ppg": gates[:, 0].detach(), "gate_ecg": gates[:, 1].detach()}


def build_model(method: str, cfg) -> nn.Module:
    if method == "ann_lstm_ecg_ppg":
        return AnnLstmECGPPG(cfg)
    if method == "ppg_bilstm_attention":
        return PPGBiLSTMAttention(cfg)
    if method == "bpnet_cnn":
        return BPNetCNN(cfg)
    if method == "mlp_bp_mixer":
        return MLPBPMixer(cfg)
    if method == "piso_transformer":
        return PiSOTransformer(cfg)
    if method == "mufubp_dual_feature_pfe":
        return MuFuBPDualPFE(cfg)
    raise ValueError(f"Unknown method: {method}")


def move_batch(batch: dict, device: str) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def tail_loss(pred: torch.Tensor, target: torch.Tensor, y_cls: torch.Tensor) -> torch.Tensor:
    under = F.relu(target - pred)
    class_w = pred.new_tensor([0.0, 0.35, 0.95, 1.70])[y_cls]
    sbp_gate = torch.clamp((target[:, 0] - 130.0) / 50.0, 0.0, 1.0)
    dbp_gate = torch.clamp((target[:, 1] - 80.0) / 35.0, 0.0, 1.0)
    return (class_w * (sbp_gate * under[:, 0] / 12.0 + dbp_gate * under[:, 1] / 8.0)).mean()


def compute_loss(
    pred: torch.Tensor,
    logits: torch.Tensor,
    batch: dict,
    class_weights: torch.Tensor,
    cls_weight: float,
    tail_weight: float,
) -> tuple[torch.Tensor, dict]:
    per_sample = F.smooth_l1_loss(pred, batch["y_reg"], reduction="none").sum(dim=1)
    sample_w = class_weights[batch["y_cls"]]
    reg = (per_sample * sample_w).mean()
    cls = F.cross_entropy(logits, batch["y_cls"], weight=class_weights, label_smoothing=0.03)
    tail = tail_loss(pred, batch["y_reg"], batch["y_cls"])
    loss = reg + float(cls_weight) * cls + float(tail_weight) * tail
    return loss, {"loss_reg": float(reg.detach()), "loss_cls": float(cls.detach()), "loss_tail": float(tail.detach())}


def evaluate(model: nn.Module, loader: DataLoader, cfg) -> dict:
    model.eval()
    y_true, y_pred, y_cls, probs, subjects, segs = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch(batch, cfg.DEVICE)
            pred, logits, _ = model(batch_dev)
            y_true.append(batch_dev["y_reg"].detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())
            y_cls.append(batch_dev["y_cls"].detach().cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            subjects.extend([str(x) for x in batch["subject_id"]])
            segs.extend([int(x) for x in batch["seg_idx"]])
    y_true_reg = np.concatenate(y_true, axis=0).astype(np.float32)
    y_pred_reg = np.concatenate(y_pred, axis=0).astype(np.float32)
    y_pred_reg[:, 0] = np.clip(y_pred_reg[:, 0], 70.0, 230.0)
    y_pred_reg[:, 1] = np.clip(y_pred_reg[:, 1], 35.0, 140.0)
    y_true_cls = np.concatenate(y_cls, axis=0).astype(np.int64)
    y_prob = np.concatenate(probs, axis=0).astype(np.float32)
    y_pred_cls = y_prob.argmax(axis=1).astype(np.int64)
    y_prob_from_reg = shared_v9.regression_to_class_prob(y_pred_reg, None, cfg)
    y_pred_cls_from_reg = shared_v9.reg_to_class_np(y_pred_reg).astype(np.int64)
    metrics_reg = shared_v9.regression_metrics(y_true_reg, y_pred_reg)
    metrics_cls = shared_v9.risk_classification_metrics(y_true_cls, y_pred_cls, y_prob, cfg, prefix="selected")
    metrics_cls_from_reg = shared_v9.risk_classification_metrics(
        y_true_cls, y_pred_cls_from_reg, y_prob_from_reg, cfg, prefix="from_reg"
    )
    return {
        "y_true_reg": y_true_reg,
        "y_pred_reg": y_pred_reg,
        "y_true_cls": y_true_cls,
        "y_pred_cls": y_pred_cls,
        "y_prob_cls": y_prob,
        "y_pred_cls_from_reg": y_pred_cls_from_reg,
        "y_prob_cls_from_reg": y_prob_from_reg,
        "subject_ids": np.asarray(subjects),
        "seg_indices": np.asarray(segs, dtype=np.int64),
        "metrics_reg": metrics_reg,
        "metrics_cls": metrics_cls,
        "metrics_cls_from_reg": metrics_cls_from_reg,
    }


def recompute_regression_dependent_outputs(out: dict, cfg) -> dict:
    updated = dict(out)
    y_pred_reg = np.asarray(updated["y_pred_reg"], dtype=np.float32).copy()
    y_pred_reg[:, 0] = np.clip(y_pred_reg[:, 0], 70.0, 230.0)
    y_pred_reg[:, 1] = np.clip(y_pred_reg[:, 1], 35.0, 140.0)
    updated["y_pred_reg"] = y_pred_reg
    updated["y_prob_cls_from_reg"] = shared_v9.regression_to_class_prob(y_pred_reg, None, cfg)
    updated["y_pred_cls_from_reg"] = shared_v9.reg_to_class_np(y_pred_reg).astype(np.int64)
    updated["metrics_reg"] = shared_v9.regression_metrics(updated["y_true_reg"], y_pred_reg)
    updated["metrics_cls_from_reg"] = shared_v9.risk_classification_metrics(
        updated["y_true_cls"],
        updated["y_pred_cls_from_reg"],
        updated["y_prob_cls_from_reg"],
        cfg,
        prefix="from_reg",
    )
    return updated


def apply_support_calibration(test_out: dict, support_out: dict, cfg, shrinkage: float) -> dict:
    shrink = float(np.clip(shrinkage, 0.0, 1.0))
    support_subjects = [str(s) for s in support_out["subject_ids"]]
    residual = support_out["y_true_reg"] - support_out["y_pred_reg"]
    global_offset = residual.mean(axis=0)
    offsets: Dict[str, np.ndarray] = {}
    for sid in sorted(set(support_subjects)):
        mask = np.asarray([s == sid for s in support_subjects], dtype=bool)
        if np.any(mask):
            offsets[sid] = residual[mask].mean(axis=0)

    updated = dict(test_out)
    y_pred_reg = np.asarray(test_out["y_pred_reg"], dtype=np.float32).copy()
    applied = np.zeros_like(y_pred_reg, dtype=np.float32)
    for idx, sid in enumerate([str(s) for s in test_out["subject_ids"]]):
        offset = offsets.get(sid, global_offset)
        applied[idx] = shrink * offset
    y_pred_reg = y_pred_reg + applied
    updated["y_pred_reg"] = y_pred_reg
    updated["support_calibration_offsets"] = applied
    return recompute_regression_dependent_outputs(updated, cfg)


def _fit_affine(x: np.ndarray, y: np.ndarray, ridge: float = 1.0e-2) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or float(np.std(x)) < 1.0e-3:
        return 1.0, float(np.mean(y - x)) if x.size else 0.0
    design = np.stack([x, np.ones_like(x)], axis=1)
    penalty = np.diag([ridge, 0.0])
    try:
        slope, intercept = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        return 1.0, float(np.mean(y - x))
    slope = float(np.clip(slope, 0.35, 1.85))
    intercept = float(np.clip(intercept, -45.0, 45.0))
    return slope, intercept


def apply_support_affine_calibration(test_out: dict, support_out: dict, cfg, shrinkage: float) -> dict:
    shrink = float(np.clip(shrinkage, 0.0, 1.0))
    support_subjects = [str(s) for s in support_out["subject_ids"]]
    global_params = []
    for dim in range(2):
        global_params.append(_fit_affine(support_out["y_pred_reg"][:, dim], support_out["y_true_reg"][:, dim]))

    subject_params: Dict[str, List[tuple[float, float]]] = {}
    for sid in sorted(set(support_subjects)):
        mask = np.asarray([s == sid for s in support_subjects], dtype=bool)
        params = []
        for dim in range(2):
            if np.count_nonzero(mask) >= 3:
                params.append(_fit_affine(support_out["y_pred_reg"][mask, dim], support_out["y_true_reg"][mask, dim]))
            else:
                params.append(global_params[dim])
        subject_params[sid] = params

    updated = dict(test_out)
    y_pred_reg = np.asarray(test_out["y_pred_reg"], dtype=np.float32).copy()
    slopes = np.ones_like(y_pred_reg, dtype=np.float32)
    intercepts = np.zeros_like(y_pred_reg, dtype=np.float32)
    for idx, sid in enumerate([str(s) for s in test_out["subject_ids"]]):
        params = subject_params.get(sid, global_params)
        for dim, (raw_slope, raw_intercept) in enumerate(params):
            slope = 1.0 + shrink * (raw_slope - 1.0)
            intercept = shrink * raw_intercept
            slopes[idx, dim] = slope
            intercepts[idx, dim] = intercept
            y_pred_reg[idx, dim] = slope * y_pred_reg[idx, dim] + intercept
    updated["y_pred_reg"] = y_pred_reg
    updated["support_calibration_slopes"] = slopes
    updated["support_calibration_intercepts"] = intercepts
    return recompute_regression_dependent_outputs(updated, cfg)


def validation_score(out: dict) -> float:
    reg = out["metrics_reg"]
    cls = selected_classification_metrics(out, "regression")
    bp_rows = shared_v9.build_bp_range_table(out["y_true_reg"], out["y_pred_reg"])
    range_map = {str(row["bp_range"]): row for row in bp_rows}
    high = range_map.get("high", {})
    crisis = range_map.get("crisis", {})
    return float(
        reg["mae_mean"]
        + 1.8 * (1.0 - cls["cls_f1_macro_selected"])
        + 0.9 * (1.0 - cls["cls_balanced_acc_selected"])
        + 0.20 * max(0.0, -float(high.get("bias_sbp", 0.0)))
        + 0.35 * max(0.0, -float(crisis.get("bias_sbp", 0.0)))
    )


def selected_classification_arrays(out: dict, source: str) -> tuple[np.ndarray, np.ndarray]:
    if source == "head":
        return out["y_pred_cls"], out["y_prob_cls"]
    return out["y_pred_cls_from_reg"], out["y_prob_cls_from_reg"]


def selected_classification_metrics(out: dict, source: str) -> dict:
    if source == "head":
        return out["metrics_cls"]
    renamed = {}
    for key, value in out["metrics_cls_from_reg"].items():
        renamed[key.replace("_from_reg", "_selected")] = value
    return renamed


def train_model(model: nn.Module, loaders: dict, cfg, args: argparse.Namespace, out_dir: Path) -> tuple[nn.Module, List[dict]]:
    class_weights = class_weights_from_dataset(loaders["train"].dataset, cfg.N_CLASSES, args.class_weight_power, cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state = None
    best_score = float("inf")
    patience = 0
    rows: List[dict] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: Dict[str, List[float]] = {"loss_total": [], "loss_reg": [], "loss_cls": [], "loss_tail": []}
        for batch in loaders["train"]:
            batch = move_batch(batch, cfg.DEVICE)
            pred, logits, _ = model(batch)
            loss, parts = compute_loss(pred, logits, batch, class_weights, args.cls_loss_weight, args.tail_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            losses["loss_total"].append(float(loss.detach()))
            for key, value in parts.items():
                losses[key].append(value)
        val_out = evaluate(model, loaders["val"], cfg)
        score = validation_score(val_out)
        val_cls = selected_classification_metrics(val_out, str(args.classification_source))
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{key: float(np.mean(value)) for key, value in losses.items() if value},
            "val_score": score,
            "val_mae_mean": float(val_out["metrics_reg"]["mae_mean"]),
            "val_acc": float(val_cls["cls_acc_selected"]),
            "val_macro_f1": float(val_cls["cls_f1_macro_selected"]),
            "val_balanced_acc": float(val_cls["cls_balanced_acc_selected"]),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, out_dir / "best_model.pt")
            patience = 0
        else:
            patience += 1
            if patience >= int(args.patience):
                print(f"Early stopping at epoch {epoch}")
                break
    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    model.load_state_dict(best_state)
    return model, rows


def build_error_by_class(out: dict, cfg) -> List[dict]:
    rows = []
    err = out["y_pred_reg"] - out["y_true_reg"]
    for cls_idx, name in enumerate(cfg.CLASS_NAMES):
        mask = out["y_true_cls"] == cls_idx
        if not np.any(mask):
            continue
        rows.append(
            {
                "class": str(name),
                "n": int(mask.sum()),
                "mae_sbp": float(np.abs(err[mask, 0]).mean()),
                "mae_dbp": float(np.abs(err[mask, 1]).mean()),
                "bias_sbp": float(err[mask, 0].mean()),
                "bias_dbp": float(err[mask, 1].mean()),
            }
        )
    return rows


def plot_training(rows: Sequence[dict], fig_dir: Path) -> None:
    if not rows:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(epochs, [float(row["loss_total"]) for row in rows], label="Train loss")
    axes[0].plot(epochs, [float(row["val_score"]) for row in rows], label="Val score")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.25)
    axes[1].plot(epochs, [float(row["val_acc"]) for row in rows], label="Accuracy")
    axes[1].plot(epochs, [float(row["val_macro_f1"]) for row in rows], label="Macro F1")
    axes[1].plot(epochs, [float(row["val_balanced_acc"]) for row in rows], label="Balanced Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "training_comparison_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_classwise_f1(out: dict, cfg, fig_dir: Path, classification_source: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    cls_metrics = selected_classification_metrics(out, classification_source)
    labels, vals = [], []
    for name in cfg.CLASS_NAMES:
        labels.append(str(name))
        vals.append(float(cls_metrics.get(f"cls_f1_selected_{name}", 0.0)))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(np.arange(len(labels)), vals, color=["#4c78a8", "#72b7b2", "#f58518", "#e45756"], alpha=0.82)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Classwise F1")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "classwise_f1_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residual_quantiles(out: dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    err = out["y_pred_reg"] - out["y_true_reg"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for ax, dim, name, color in ((axes[0], 0, "SBP", "#4c78a8"), (axes[1], 1, "DBP", "#f58518")):
        q = np.quantile(out["y_true_reg"][:, dim], [0.0, 0.50, 0.75, 0.90, 0.95, 1.0])
        rows = []
        labels = []
        for i in range(len(q) - 1):
            mask = (out["y_true_reg"][:, dim] >= q[i]) & (out["y_true_reg"][:, dim] <= q[i + 1])
            rows.append(float(err[mask, dim].mean()) if np.any(mask) else 0.0)
            labels.append(f"q{i + 1}")
        ax.bar(np.arange(len(rows)), rows, color=color, alpha=0.78)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean residual (mmHg)")
        ax.set_title(f"{name} residual by BP quantile")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_by_bp_quantile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    method: str,
    paper: dict,
    out: dict,
    val_out: dict,
    train_rows: Sequence[dict],
    cfg,
    out_dir: Path,
    source_dir: Path,
    classification_source: str,
    support_calibration: str,
    support_shrinkage: float,
    runtime_seconds: float,
) -> None:
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    art_dir = out_dir / "artifacts"
    for path in (fig_dir, tbl_dir, art_dir):
        path.mkdir(parents=True, exist_ok=True)

    selected_cls = selected_classification_metrics(out, classification_source)
    selected_val_cls = selected_classification_metrics(val_out, classification_source)
    selected_pred_cls, selected_prob_cls = selected_classification_arrays(out, classification_source)
    metrics = {**out["metrics_reg"], **out["metrics_cls"], **out["metrics_cls_from_reg"]}
    selected_metrics = {**out["metrics_reg"], **selected_cls}
    final_results = {
        "method": method,
        "paper_reference": paper,
        "protocol_source": str(source_dir),
        "classification_source": classification_source,
        "support_calibration": support_calibration,
        "support_shrinkage": float(support_shrinkage),
        "test": metrics,
        "test_selected": selected_metrics,
        "validation": {**val_out["metrics_reg"], **val_out["metrics_cls"], **val_out["metrics_cls_from_reg"]},
        "validation_selected": {**val_out["metrics_reg"], **selected_val_cls},
    }
    write_json(out_dir / "final_results.json", final_results)
    write_json(out_dir / "paper_metrics.json", common_script.build_paper_metrics(out["metrics_reg"]))
    write_json(
        out_dir / "selected_strategy.json",
        {
            "selected_regression_candidate": method,
            "selected_classification_candidate": f"{method}_{classification_source}",
            "classification_source": classification_source,
            "support_calibration": support_calibration,
            "support_shrinkage": float(support_shrinkage),
            "paper_reference": paper,
        },
    )
    write_json(
        out_dir / "runtime_metrics.json",
        {
            "runtime_seconds": float(runtime_seconds),
            "method": method,
            "classification_source": classification_source,
            "support_calibration": support_calibration,
            "support_shrinkage": float(support_shrinkage),
        },
    )
    write_json(
        out_dir / "protocol_manifest.json",
        {
            "protocol_id": cfg.PROTOCOL_ID,
            "protocol_name": cfg.PROTOCOL_NAME,
            "split_protocol": "v10.11_subjectdisjoint_entry_reuse",
            "source_protocol_dir": str(source_dir),
            "classification_source": classification_source,
            "support_calibration": support_calibration,
            "support_shrinkage": float(support_shrinkage),
            "paper_reference": paper,
        },
    )
    write_json(
        out_dir / "protocol_summary.json",
        {
            "method": method,
            "classification_source": classification_source,
            "support_calibration": support_calibration,
            "support_shrinkage": float(support_shrinkage),
            "selected_acc": float(selected_cls["cls_acc_selected"]),
            "selected_macro_f1": float(selected_cls["cls_f1_macro_selected"]),
            "selected_balanced_acc": float(selected_cls["cls_balanced_acc_selected"]),
            "selected_mae_mean": float(out["metrics_reg"]["mae_mean"]),
            "selected_regression_candidate": method,
            "selected_classification_candidate": f"{method}_{classification_source}",
        },
    )
    write_rows_csv(out_dir / "epoch_log.csv", train_rows)
    bp_rows = shared_v9.build_bp_range_table(out["y_true_reg"], out["y_pred_reg"])
    write_rows_csv(tbl_dir / "bp_range_metrics.csv", bp_rows)
    write_rows_csv(tbl_dir / "regression_error_by_class.csv", build_error_by_class(out, cfg))
    write_rows_csv(tbl_dir / "regression_variant_summary.csv", [{"variant": method, **out["metrics_reg"]}])
    write_rows_csv(
        tbl_dir / "classification_variant_summary.csv",
        [{"variant": method, "classification_source": classification_source, **selected_cls}],
    )
    write_rows_csv(
        tbl_dir / "robustness_summary.csv",
        [
            {
                "variant": method,
                "classification_source": classification_source,
                "clean_acc": float(selected_cls["cls_acc_selected"]),
                "clean_macro_f1": float(selected_cls["cls_f1_macro_selected"]),
                "clean_mae_mean": float(out["metrics_reg"]["mae_mean"]),
            }
        ],
    )
    error_cdf = common_script.build_error_cdf_rows(out["y_true_reg"], out["y_pred_reg"])
    write_rows_csv(out_dir / "error_cdf.csv", error_cdf)
    calib_rows = shared_v9.build_calibration_curve_table(out["y_true_cls"], selected_prob_cls)
    write_rows_csv(out_dir / "calibration_curve.csv", calib_rows)
    np.savez_compressed(
        art_dir / "test_outputs_regression_selected.npz",
        y_true_reg=out["y_true_reg"],
        y_pred_reg=out["y_pred_reg"],
        y_true_cls=out["y_true_cls"],
        y_pred_cls_from_reg=out["y_pred_cls_from_reg"],
        y_prob_cls_from_reg=out["y_prob_cls_from_reg"],
        subject_ids=out["subject_ids"],
        seg_indices=out["seg_indices"],
    )
    np.savez_compressed(
        art_dir / "test_outputs_classification_selected.npz",
        y_true_cls=out["y_true_cls"],
        y_pred_cls=selected_pred_cls,
        y_prob_cls=selected_prob_cls,
        classification_source=np.asarray([classification_source]),
        subject_ids=out["subject_ids"],
        seg_indices=out["seg_indices"],
    )

    shared_v9.plot_scatter_true_vs_pred(out["y_true_reg"], out["y_pred_reg"], fig_dir, filename="scatter_true_vs_pred.png")
    shared_v9.plot_bland_altman(out["y_true_reg"], out["y_pred_reg"], fig_dir, filename="bland_altman.png")
    shared_v9.plot_confusion(out["y_true_cls"], selected_pred_cls, list(cfg.CLASS_NAMES), fig_dir, "confusion_matrix_selected_final.png")
    shared_v9.plot_roc_pr(out["y_true_cls"], selected_prob_cls, cfg, fig_dir, prefix="selected_final")
    shared_v9.plot_calibration(calib_rows, fig_dir, filename="calibration_curve.png")
    common_script.plot_error_cdf(error_cdf, fig_dir)
    common_script.plot_bp_range_bias(bp_rows, fig_dir)
    plot_training(train_rows, fig_dir)
    plot_classwise_f1(out, cfg, fig_dir, classification_source)
    plot_residual_quantiles(out, fig_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v10.13 paper-method comparison baselines on the v10.11 protocol split.")
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--output-name", default="")
    parser.add_argument("--protocol-source", default=DEFAULT_PROTOCOL_SOURCE)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--class-weight-power", type=float, default=0.70)
    parser.add_argument("--cls-loss-weight", type=float, default=0.45)
    parser.add_argument("--tail-loss-weight", type=float, default=0.28)
    parser.add_argument(
        "--classification-source",
        choices=("regression", "head"),
        default="regression",
        help="Paper-facing risk class source. Regression maps predicted SBP/DBP to clinical classes.",
    )
    parser.add_argument(
        "--support-calibration",
        choices=("none", "subject_offset", "subject_affine"),
        default="subject_affine",
        help="Use test-support labels for the same subject-disjoint few-shot personalization protocol as v10.18.",
    )
    parser.add_argument(
        "--support-shrinkage",
        type=float,
        default=0.85,
        help="Shrinkage applied to support-set subject offset calibration.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    started_at = time.time()
    args = parse_args()
    paper = METHODS[args.method]
    if not args.output_name:
        args.output_name = f"mimic_bp_compare_v10_13_{args.method}_proto"
    seed_everything(int(args.seed))
    cfg = make_cfg(args)
    cfg.SEED = int(args.seed)
    out_dir = Path(cfg.PROJECT_ROOT) / "outputs" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir, ds_train, ds_val, ds_test, ds_support = build_datasets(cfg, args.protocol_source)
    loaders = {
        "train": make_train_loader(ds_train, cfg, args.class_weight_power),
        "val": make_eval_loader(ds_val, cfg),
        "test": make_eval_loader(ds_test, cfg),
        "support": make_eval_loader(ds_support, cfg),
    }
    model = build_model(args.method, cfg).to(cfg.DEVICE)
    model, train_rows = train_model(model, loaders, cfg, args, out_dir)
    val_out = evaluate(model, loaders["val"], cfg)
    test_out = evaluate(model, loaders["test"], cfg)
    if args.support_calibration == "subject_offset":
        support_out = evaluate(model, loaders["support"], cfg)
        test_out = apply_support_calibration(test_out, support_out, cfg, args.support_shrinkage)
    elif args.support_calibration == "subject_affine":
        support_out = evaluate(model, loaders["support"], cfg)
        test_out = apply_support_affine_calibration(test_out, support_out, cfg, args.support_shrinkage)
    save_outputs(
        args.method,
        paper,
        test_out,
        val_out,
        train_rows,
        cfg,
        out_dir,
        source_dir,
        str(args.classification_source),
        str(args.support_calibration),
        float(args.support_shrinkage),
        time.time() - started_at,
    )
    print(f"Done. Results saved to: {out_dir}")
    print(
        json.dumps(
            {
                "method": args.method,
                "classification_source": args.classification_source,
                "support_calibration": args.support_calibration,
                "support_shrinkage": args.support_shrinkage,
                "test": selected_classification_metrics(test_out, str(args.classification_source)),
                "regression": test_out["metrics_reg"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
