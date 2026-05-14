
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import aqm_bp_shared_v9 as base

# Re-export frequently used symbols from v9 shared module
QMoERegressionNet = base.QMoERegressionNet
QMoEClassificationNet = base.QMoEClassificationNet
QMoEClassifyNet = QMoEClassificationNet
ModelEMA = base.ModelEMA
apply_asymmetric_modality_dropout = base.apply_asymmetric_modality_dropout
apply_training_quality_augmentation = base.apply_training_quality_augmentation
build_bp_range_table = base.build_bp_range_table
build_calibration_curve_table = base.build_calibration_curve_table
build_class_weights = base.build_class_weights
build_credibility_prior_torch = base.build_credibility_prior_torch
build_conditional_coverage_table = base.build_conditional_coverage_table
build_range_weights = base.build_range_weights
build_router_target = base.build_router_target
build_subjectwise_error_table = base.build_subjectwise_error_table
collect_outputs_regression = base.collect_outputs_regression
collect_outputs_classification = base.collect_outputs_classification
conformal_from_outputs = base.conformal_from_outputs
ensure_out_dirs = base.ensure_out_dirs
heteroscedastic_loss = base.heteroscedastic_loss
measure_runtime = base.measure_runtime
move_batch = base.move_batch
plot_bland_altman = base.plot_bland_altman
plot_calibration = base.plot_calibration
plot_confusion = base.plot_confusion
plot_noise_robustness = base.plot_noise_robustness
plot_quality_conditional_coverage = base.plot_quality_conditional_coverage
plot_roc_pr = base.plot_roc_pr
plot_router_heatmap = base.plot_router_heatmap
plot_scatter_true_vs_pred = base.plot_scatter_true_vs_pred
plot_sharpness_vs_coverage = base.plot_sharpness_vs_coverage
plot_training_curves_reg = base.plot_training_curves_reg
plot_uncertainty_error_corr = base.plot_uncertainty_error_corr
regression_metrics = base.regression_metrics
risk_classification_metrics = base.risk_classification_metrics
reg_to_class_np = base.reg_to_class_np
regression_to_class_prob = base.regression_to_class_prob
regression_to_class_prob_torch = base.regression_to_class_prob_torch
save_epoch_log = base.save_epoch_log
save_json = base.save_json
save_regression_npz = base.save_regression_npz
save_rows_csv = base.save_rows_csv
seed_everything = base.seed_everything
set_warmup_cosine_lr = base.set_warmup_cosine_lr
normalize_subject_id = base.normalize_subject_id
read_subject_txt = base.read_subject_txt
bp_to_risk_class = base.bp_to_risk_class
random_crop_or_pad = base.random_crop_or_pad
center_crop_or_pad = base.center_crop_or_pad
safe_bandpass_filter = base.safe_bandpass_filter
robust_quality_features_np = base.robust_quality_features_np
zscore_1d = base.zscore_1d


def _labels_path(labels_dir: Path, sid: str) -> Path:
    candidates = [
        labels_dir / f"{sid}_labels.npy",
        labels_dir / f"{sid}_label.npy",
        labels_dir / f"{sid}.npy",
    ]
    for c in candidates:
        if c.exists():
            return c
    return labels_dir / f"{sid}_labels.npy"


def _load_npy(path: Path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        arr = arr.item()
    return arr


def _labels_from_array(arr: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2 and arr.shape == (30, 2):
        return arr
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    if arr.ndim == 2 and arr.shape == (2, 30):
        return arr.T
    if arr.ndim == 1 and arr.size % 2 == 0:
        return arr.reshape(-1, 2)
    return None


def _labels_from_mapping(obj: dict) -> np.ndarray | None:
    lowered = {str(k).lower(): v for k, v in obj.items()}
    for k1, k2 in [("sbp", "dbp"), ("sys", "dia"), ("systolic", "diastolic")]:
        if k1 in lowered and k2 in lowered:
            sbp = np.asarray(lowered[k1], dtype=np.float32).reshape(-1)
            dbp = np.asarray(lowered[k2], dtype=np.float32).reshape(-1)
            if len(sbp) == len(dbp) and len(sbp) > 0:
                return np.stack([sbp, dbp], axis=1)
    return None


def _load_labels_only(cfg: "CFG", sid: str) -> np.ndarray:
    path = _labels_path(Path(cfg.DATA_ROOT) / "labels", sid)
    obj = _load_npy(path)
    if isinstance(obj, dict):
        out = _labels_from_mapping(obj)
        if out is not None:
            return out
    out = _labels_from_array(obj)
    if out is not None:
        return out
    raise ValueError(f"Unsupported labels format for {sid}: type={type(obj)}")


def _subject_segment_count(cfg: "CFG", sid: str) -> int:
    try:
        labels = _load_labels_only(cfg, sid)
    except Exception:
        return 0
    return int(labels.shape[0])


@dataclass
class CFG(base.CFG):
    OUTPUT_NAME: str = "mimic_bp_reg_v11_proto"
    EPOCHS: int = 28
    EARLY_STOPPING_PATIENCE: int = 10
    # protocol options
    SPLIT_PROTOCOL: str = "subject_overlap_stratified_calibrated"
    PROTOCOL_SEED: int = 42
    SEGMENTWISE_SPLIT_COUNTS: Tuple[int, int, int, int] = (16, 4, 5, 5)  # train/val/calib/test, sums to 30
    SUBJECT_POOL: str = "all"  # all | official_union
    SAVE_PROTOCOL_FILES: bool = True
    PROTOCOL_NAME: str = "subject-overlap stratified segment-wise calibrated personalized protocol"
    USE_SUBJECT_CALIBRATION: bool = True
    CALIBRATION_SHRINKAGE: float = 3.0

    # defaults tilted slightly toward paper-facing clean performance
    MODALITY_DROPOUT_PPG: float = 0.10
    MODALITY_DROPOUT_ECG: float = 0.04
    P_DEGRADE: float = 0.20
    LAMBDA_Q: float = 0.015
    LAMBDA_ROUTER: float = 0.015
    LAMBDA_BAL: float = 0.008
    LAMBDA_PPG_AUX: float = 0.04
    LAMBDA_ECG_AUX: float = 0.10
    LAMBDA_CRED: float = 0.10
    LAMBDA_REG_CLS: float = 0.45
    LAMBDA_TAIL: float = 0.25
    REG_USE_WEIGHTED_SAMPLER: bool = True
    REG_SAMPLER_POWER: float = 0.75
    CREDIBILITY_SMOOTHING: float = 0.08
    FUSED_CREDIBILITY_PRIOR: float = 0.65
    FUSED_CREDIBILITY_FLOOR: float = 0.22


class IndexedMIMICBPDataset(base.MIMICBPDataset):
    def __init__(self, cfg: CFG, entries: Sequence[Tuple[str, int]], crop_len: int, mode: str = "train", seed: int = 42):
        self.cfg = cfg
        self.root = Path(cfg.DATA_ROOT)
        self.crop_len = crop_len
        self.mode = mode
        self.rng = random.Random(seed)
        self.ppg_dir = self.root / "ppg"
        self.ecg_dir = self.root / "ecg"
        self.labels_dir = self.root / "labels"
        self.index: List[Tuple[str, int]] = []
        self.sample_classes: List[int] = []
        self.cache: Dict[str, Dict[str, np.ndarray]] = {}

        label_cache: Dict[str, np.ndarray] = {}
        for sid_raw, seg_idx_raw in entries:
            sid = normalize_subject_id(sid_raw)
            seg_idx = int(seg_idx_raw)
            if sid not in label_cache:
                try:
                    label_cache[sid] = self._load_labels_only(sid)
                except Exception:
                    continue
            labels = label_cache[sid]
            if seg_idx < 0 or seg_idx >= labels.shape[0]:
                continue
            sbp, dbp = labels[seg_idx]
            cls = bp_to_risk_class(float(sbp), float(dbp))
            self.index.append((sid, seg_idx))
            self.sample_classes.append(cls)
        self.subjects = sorted({sid for sid, _ in self.index})
        self.class_counts = np.bincount(np.asarray(self.sample_classes), minlength=cfg.N_CLASSES).tolist() if self.sample_classes else [0] * cfg.N_CLASSES


def _all_subjects_from_files(root: Path) -> List[str]:
    ppg_dir = root / "ppg"
    ecg_dir = root / "ecg"
    labels_dir = root / "labels"
    subjects = []
    for p in sorted(ppg_dir.glob("*.npy")):
        stem = p.stem
        sid = stem.replace("_ppg", "")
        sid = normalize_subject_id(sid)
        ecg_candidates = [ecg_dir / f"{sid}_ecg.npy", ecg_dir / f"{sid}.npy"]
        label_candidates = [labels_dir / f"{sid}_labels.npy", labels_dir / f"{sid}_label.npy", labels_dir / f"{sid}.npy"]
        if any(c.exists() for c in ecg_candidates) and any(c.exists() for c in label_candidates):
            subjects.append(sid)
    return sorted(set(subjects))


def _wave_shape_is_supported(obj) -> bool:
    if isinstance(obj, dict):
        for key in ("signal", "wave", "waves", "data", "x"):
            if key in obj:
                obj = obj[key]
                break
    arr = np.asarray(obj, dtype=np.float32)
    if arr.ndim == 2 and (arr.shape[0] == 30 or arr.shape[1] == 30):
        return True
    if arr.ndim == 1 and arr.size % 30 == 0:
        return True
    return False


def _subject_bundle_is_readable(root: Path, sid: str) -> bool:
    ppg_dir = root / "ppg"
    ecg_dir = root / "ecg"
    labels_dir = root / "labels"
    ppg_candidates = [ppg_dir / f"{sid}_ppg.npy", ppg_dir / f"{sid}.npy"]
    ecg_candidates = [ecg_dir / f"{sid}_ecg.npy", ecg_dir / f"{sid}.npy"]
    label_path = _labels_path(labels_dir, sid)
    try:
        ppg_path = next((p for p in ppg_candidates if p.exists()), None)
        ecg_path = next((p for p in ecg_candidates if p.exists()), None)
        if ppg_path is None or ecg_path is None or not label_path.exists():
            return False
        ppg_obj = _load_npy(ppg_path)
        ecg_obj = _load_npy(ecg_path)
        if not _wave_shape_is_supported(ppg_obj):
            return False
        if not _wave_shape_is_supported(ecg_obj):
            return False
        labels_obj = _load_npy(label_path)
        if isinstance(labels_obj, dict):
            labels = _labels_from_mapping(labels_obj)
        else:
            labels = _labels_from_array(labels_obj)
        return labels is not None and int(labels.shape[0]) > 0
    except Exception:
        return False


def _official_union_subjects(root: Path) -> List[str]:
    split_dir = root / "splits"
    names = ["train_subjects.txt", "val_subjects.txt", "calib_subjects.txt", "test_subjects.txt"]
    out: List[str] = []
    for name in names:
        p = split_dir / name
        if p.exists():
            out.extend(read_subject_txt(p))
    return sorted(set(out))


def _subject_pool(cfg: CFG) -> List[str]:
    root = Path(cfg.DATA_ROOT)
    if cfg.SUBJECT_POOL == "official_union":
        pool = _official_union_subjects(root)
        if pool:
            valid_pool = [sid for sid in pool if _subject_bundle_is_readable(root, sid)]
            skipped = int(len(pool) - len(valid_pool))
            if skipped > 0:
                print(
                    f"Warning: skipped {skipped} subject(s) with unreadable or corrupt "
                    f"ppg/ecg/labels files under DATA_ROOT={root}"
                )
            return valid_pool
    pool = _all_subjects_from_files(root)
    valid_pool = [sid for sid in pool if _subject_bundle_is_readable(root, sid)]
    skipped = int(len(pool) - len(valid_pool))
    if skipped > 0:
        print(
            f"Warning: skipped {skipped} subject(s) with unreadable or corrupt "
            f"ppg/ecg/labels files under DATA_ROOT={root}"
        )
    return valid_pool


def _allocate_counts(n_items: int, target_counts: Sequence[int]) -> np.ndarray:
    target = np.asarray(target_counts, dtype=np.float64)
    if n_items <= 0:
        return np.zeros(len(target_counts), dtype=np.int64)
    raw = target / max(target.sum(), 1.0) * float(n_items)
    alloc = np.floor(raw).astype(np.int64)
    remainder = int(n_items - alloc.sum())
    if remainder > 0:
        frac = raw - alloc
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            alloc[idx] += 1
    return alloc


def _segmentwise_entries_for_subjects(cfg: "CFG", subjects: Sequence[str], counts: Tuple[int, int, int, int], seed: int):
    n_train, n_val, n_calib, n_test = [int(x) for x in counts]
    train_entries: List[Tuple[str, int]] = []
    val_entries: List[Tuple[str, int]] = []
    calib_entries: List[Tuple[str, int]] = []
    test_entries: List[Tuple[str, int]] = []
    count_cache: Dict[str, int] = {}
    split_targets = [n_train, n_val, n_calib, n_test]
    for sid in subjects:
        try:
            labels = _load_labels_only(cfg, sid)
        except Exception:
            continue
        n_segments = int(labels.shape[0])
        count_cache[sid] = n_segments
        if (n_train + n_val + n_calib + n_test) > n_segments:
            raise ValueError(
                f"SEGMENTWISE_SPLIT_COUNTS sums to {n_train + n_val + n_calib + n_test}, "
                f"but subject {sid} only has {n_segments} segments."
            )
        local_seed = seed + int(str(sid).replace("p", ""))
        groups: Dict[int, List[int]] = {}
        for seg_idx, (sbp, dbp) in enumerate(labels):
            cls = bp_to_risk_class(float(sbp), float(dbp))
            groups.setdefault(cls, []).append(int(seg_idx))

        split_lists = [[], [], [], []]
        for cls in sorted(groups.keys()):
            idxs = list(groups[cls])
            rs = np.random.RandomState(local_seed + cls * 97)
            rs.shuffle(idxs)
            alloc = _allocate_counts(len(idxs), split_targets)
            cursor = 0
            for split_idx, take_n in enumerate(alloc.tolist()):
                if take_n <= 0:
                    continue
                split_lists[split_idx].extend(idxs[cursor:cursor + take_n])
                cursor += take_n

        targets = split_targets[:]
        sizes = [len(x) for x in split_lists]
        while sizes != targets:
            deficits = [targets[i] - sizes[i] for i in range(4)]
            receiver = next((i for i, d in enumerate(deficits) if d > 0), None)
            donor = next((i for i, d in enumerate(deficits) if d < 0 and len(split_lists[i]) > 0), None)
            if receiver is None or donor is None:
                break
            split_lists[receiver].append(split_lists[donor].pop())
            sizes = [len(x) for x in split_lists]

        train_entries.extend((sid, int(i)) for i in split_lists[0])
        val_entries.extend((sid, int(i)) for i in split_lists[1])
        calib_entries.extend((sid, int(i)) for i in split_lists[2])
        test_entries.extend((sid, int(i)) for i in split_lists[3])
    return train_entries, val_entries, calib_entries, test_entries


def protocol_manifest(cfg: CFG) -> Dict[str, object]:
    return {
        "split_protocol": cfg.SPLIT_PROTOCOL,
        "protocol_name": cfg.PROTOCOL_NAME,
        "protocol_seed": int(cfg.PROTOCOL_SEED),
        "segmentwise_split_counts": list(cfg.SEGMENTWISE_SPLIT_COUNTS),
        "subject_pool": cfg.SUBJECT_POOL,
        "warning": (
            "This protocol is subject-overlap, class-stratified, and calibration-aware. "
            "It is suitable for intra-subject or personalized evaluation, but it is NOT directly comparable "
            "to the official MIMIC-BP calibration-free subject-wise benchmark."
        ),
    }


def maybe_save_protocol_files(cfg: CFG, train_entries, val_entries, calib_entries, test_entries):
    if not cfg.SAVE_PROTOCOL_FILES:
        return
    out_dir = Path(cfg.PROJECT_ROOT) / "outputs" / cfg.OUTPUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "protocol_manifest.json", "w", encoding="utf-8") as f:
        json.dump(protocol_manifest(cfg), f, ensure_ascii=False, indent=2)
    summary = {
        "n_train_entries": len(train_entries),
        "n_val_entries": len(val_entries),
        "n_calib_entries": len(calib_entries),
        "n_test_entries": len(test_entries),
        "n_train_subjects": len({s for s, _ in train_entries}),
        "n_val_subjects": len({s for s, _ in val_entries}),
        "n_calib_subjects": len({s for s, _ in calib_entries}),
        "n_test_subjects": len({s for s, _ in test_entries}),
    }
    with open(out_dir / "protocol_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    split_entries = {
        "train": train_entries,
        "val": val_entries,
        "calib": calib_entries,
        "test": test_entries,
    }
    for split_name, entries in split_entries.items():
        with open(out_dir / f"{split_name}_entries.csv", "w", encoding="utf-8") as f:
            f.write("subject_id,seg_idx\n")
            for sid, seg_idx in entries:
                f.write(f"{sid},{int(seg_idx)}\n")


def build_loaders(cfg: CFG, task: str = "regression"):
    if cfg.SPLIT_PROTOCOL == "official_subjectwise":
        return base.build_loaders(cfg, task=task)
    if cfg.SPLIT_PROTOCOL != "subject_overlap_stratified_calibrated":
        raise ValueError(f"Unsupported SPLIT_PROTOCOL: {cfg.SPLIT_PROTOCOL}")

    subjects = _subject_pool(cfg)
    if not subjects:
        raise FileNotFoundError("No subjects found under DATA_ROOT for the selected protocol.")

    train_entries, val_entries, calib_entries, test_entries = _segmentwise_entries_for_subjects(
        cfg, subjects, cfg.SEGMENTWISE_SPLIT_COUNTS, cfg.PROTOCOL_SEED
    )
    maybe_save_protocol_files(cfg, train_entries, val_entries, calib_entries, test_entries)

    ds_train = IndexedMIMICBPDataset(cfg, train_entries, cfg.CROP_LEN, mode="train", seed=cfg.SEED)
    ds_val = IndexedMIMICBPDataset(cfg, val_entries, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)
    ds_calib = IndexedMIMICBPDataset(cfg, calib_entries, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)
    ds_test = IndexedMIMICBPDataset(cfg, test_entries, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)

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
