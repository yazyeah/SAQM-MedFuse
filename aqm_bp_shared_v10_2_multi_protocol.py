from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

import aqm_bp_shared_v10_2_protocol as shared
import aqm_bp_shared_v11_protocol as v11


@dataclass
class CFG(shared.CFG):
    PROJECT_ROOT: str = str(Path(__file__).resolve().parent)
    PROTOCOL_ID: str = "v10.2.1_official_subjectwise_calfree"
    PROTOCOL_STRICTNESS_RANK: int = 1
    SUBJECTWISE_SPLIT_RATIOS: Tuple[float, float, float] = (0.80, 0.10, 0.10)
    FEWSHOT_SUPPORT_QUERY_COUNTS: Tuple[int, int] = (5, 25)
    SUBJECT_CALIBRATION_MODE: str = "bias"
    SUBJECT_AFFINE_MIN_SHOTS: int = 3
    SUBJECT_AFFINE_SCALE_CLIP: Tuple[float, float] = (0.85, 1.15)
    REG_TO_CLASS_THRESHOLD_BLEND: float = 0.85
    TAIL_CLASS_WEIGHTS: Tuple[float, float, float, float] = (0.0, 0.20, 0.60, 1.00)
    NORMAL_OVERPRED_WEIGHT: float = 0.0
    TAIL_SBP_SCALE: float = 12.0
    TAIL_DBP_SCALE: float = 8.0
    INIT_CKPT_PATH: str = ""
    PLOT_COMPOSITE_TRAINING_CURVES: bool = True
    SAVE_SPLIT_TRAINING_CURVES: bool = False


@dataclass
class ProtocolLoaders:
    ds_train: object
    train_loader: DataLoader
    val_query_loader: DataLoader
    val_calib_loader: DataLoader
    test_query_loader: DataLoader
    test_calib_loader: DataLoader
    split_datasets: Dict[str, object]
    manifest: Dict[str, object]


def _loader_kwargs(cfg: CFG) -> dict:
    return {
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": cfg.DEVICE == "cuda",
        "persistent_workers": cfg.NUM_WORKERS > 0,
    }


def _make_train_loader(ds_train, cfg: CFG, task: str):
    common_loader_kwargs = _loader_kwargs(cfg)
    if task == "classification":
        counts = np.asarray(ds_train.class_counts, dtype=np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        class_w = 1.0 / np.sqrt(counts)
        sample_w = np.asarray([class_w[c] for c in ds_train.sample_classes], dtype=np.float64)
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
        return DataLoader(ds_train, batch_size=cfg.BATCH_SIZE, sampler=sampler, **common_loader_kwargs)
    if cfg.REG_USE_WEIGHTED_SAMPLER:
        counts = np.asarray(ds_train.class_counts, dtype=np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        class_w = 1.0 / np.power(counts, float(cfg.REG_SAMPLER_POWER))
        class_w = class_w / max(class_w.mean(), 1e-6)
        sample_w = np.asarray([class_w[c] for c in ds_train.sample_classes], dtype=np.float64)
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
        return DataLoader(ds_train, batch_size=cfg.BATCH_SIZE, sampler=sampler, **common_loader_kwargs)
    return DataLoader(ds_train, batch_size=cfg.BATCH_SIZE, shuffle=True, **common_loader_kwargs)


def _make_eval_loader(ds_eval, cfg: CFG):
    return DataLoader(ds_eval, batch_size=cfg.BATCH_SIZE, shuffle=False, **_loader_kwargs(cfg))


def _all_entries_for_subjects(cfg: CFG, subjects: Sequence[str]) -> List[Tuple[str, int]]:
    entries: List[Tuple[str, int]] = []
    for sid in subjects:
        try:
            labels = v11._load_labels_only(cfg, sid)
        except Exception:
            continue
        for seg_idx in range(int(labels.shape[0])):
            entries.append((sid, int(seg_idx)))
    return entries


def _save_protocol_artifacts(
    cfg: CFG,
    manifest: dict,
    split_subjects: Dict[str, Sequence[str]],
    split_entries: Dict[str, Sequence[Tuple[str, int]]],
):
    out_dir = Path(cfg.PROJECT_ROOT) / "outputs" / cfg.OUTPUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "protocol_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    summary = {
        "protocol_id": cfg.PROTOCOL_ID,
        "protocol_rank": int(cfg.PROTOCOL_STRICTNESS_RANK),
        "split_subject_counts": {k: int(len(v)) for k, v in split_subjects.items()},
        "split_entry_counts": {k: int(len(v)) for k, v in split_entries.items()},
    }
    with open(out_dir / "protocol_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    for split_name, subjects in split_subjects.items():
        if not subjects:
            continue
        with open(out_dir / f"{split_name}_subjects.txt", "w", encoding="utf-8") as f:
            for sid in subjects:
                f.write(f"{sid}\n")
    for split_name, entries in split_entries.items():
        with open(out_dir / f"{split_name}_entries.csv", "w", encoding="utf-8") as f:
            f.write("subject_id,seg_idx\n")
            for sid, seg_idx in entries:
                f.write(f"{sid},{int(seg_idx)}\n")


def build_protocol_manifest(cfg: CFG) -> dict:
    if cfg.SPLIT_PROTOCOL == "official_subjectwise":
        return {
            "protocol_id": cfg.PROTOCOL_ID,
            "protocol_rank": int(cfg.PROTOCOL_STRICTNESS_RANK),
            "split_protocol": cfg.SPLIT_PROTOCOL,
            "protocol_name": cfg.PROTOCOL_NAME,
            "subject_pool": cfg.SUBJECT_POOL,
            "use_subject_calibration": bool(cfg.USE_SUBJECT_CALIBRATION),
            "few_shot_support_query_counts": None,
            "warning": (
                "Strict official subject-wise calibration-free benchmark. "
                "This is the hardest and least favorable setting, but it is the cleanest main benchmark."
            ),
        }
    if cfg.SPLIT_PROTOCOL == "subject_disjoint_fewshot_personalized":
        return {
            "protocol_id": cfg.PROTOCOL_ID,
            "protocol_rank": int(cfg.PROTOCOL_STRICTNESS_RANK),
            "split_protocol": cfg.SPLIT_PROTOCOL,
            "protocol_name": cfg.PROTOCOL_NAME,
            "subject_pool": cfg.SUBJECT_POOL,
            "subjectwise_split_ratios": list(cfg.SUBJECTWISE_SPLIT_RATIOS),
            "few_shot_support_query_counts": list(cfg.FEWSHOT_SUPPORT_QUERY_COUNTS),
            "subject_calibration_mode": cfg.SUBJECT_CALIBRATION_MODE,
            "warning": (
                "Subjects are disjoint across train, validation, and test. "
                "Validation and test subjects each receive a small support set for few-shot personalization."
            ),
        }
    return {
        "protocol_id": cfg.PROTOCOL_ID,
        "protocol_rank": int(cfg.PROTOCOL_STRICTNESS_RANK),
        "split_protocol": cfg.SPLIT_PROTOCOL,
        "protocol_name": cfg.PROTOCOL_NAME,
        "subject_pool": cfg.SUBJECT_POOL,
        "segmentwise_split_counts": list(cfg.SEGMENTWISE_SPLIT_COUNTS),
        "few_shot_support_query_counts": list(cfg.FEWSHOT_SUPPORT_QUERY_COUNTS),
        "subject_calibration_mode": cfg.SUBJECT_CALIBRATION_MODE,
        "warning": (
            "Subject-overlap upper-bound protocol. "
            "This is the most favorable personalized setting and should be reported as an upper bound."
        ),
    }


def make_official_cfg() -> CFG:
    return CFG(
        OUTPUT_NAME="mimic_bp_reg_v10_2_1_official_subjectwise_calfree",
        PROTOCOL_ID="v10.2.1_official_subjectwise_calfree",
        PROTOCOL_STRICTNESS_RANK=1,
        SPLIT_PROTOCOL="official_subjectwise",
        PROTOCOL_NAME="v10.2.1 strict official subject-wise calibration-free benchmark",
        SUBJECT_POOL="official_union",
        USE_SUBJECT_CALIBRATION=False,
        MODEL_SELECTION_USE_CALIBRATED_VAL=False,
        FEW_SHOT_SWEEP=(0,),
        REG_TO_CLASS_THRESHOLD_BLEND=0.85,
    )


def make_subject_disjoint_fewshot_cfg() -> CFG:
    return CFG(
        OUTPUT_NAME="mimic_bp_reg_v10_2_2_subject_disjoint_fewshot",
        PROTOCOL_ID="v10.2.2_subject_disjoint_fewshot",
        PROTOCOL_STRICTNESS_RANK=2,
        SPLIT_PROTOCOL="subject_disjoint_fewshot_personalized",
        PROTOCOL_NAME="v10.2.2 subject-disjoint few-shot personalized protocol",
        SUBJECT_POOL="official_union",
        SUBJECTWISE_SPLIT_RATIOS=(0.80, 0.10, 0.10),
        FEWSHOT_SUPPORT_QUERY_COUNTS=(5, 25),
        USE_SUBJECT_CALIBRATION=True,
        SUBJECT_CALIBRATION_MODE="bias",
        CALIBRATION_SHRINKAGE=3.0,
        FEW_SHOT_SWEEP=(0, 1, 3, 5),
        MODEL_SELECTION_USE_CALIBRATED_VAL=True,
        VAL_CAL_SCORE_BLEND=0.60,
        REG_TO_CLASS_THRESHOLD_BLEND=0.88,
    )


def make_subject_overlap_upper_bound_cfg() -> CFG:
    return CFG(
        OUTPUT_NAME="mimic_bp_reg_v10_2_3_subject_overlap_upper_bound",
        PROTOCOL_ID="v10.2.3_subject_overlap_upper_bound",
        PROTOCOL_STRICTNESS_RANK=3,
        SPLIT_PROTOCOL="subject_overlap_stratified_calibrated",
        PROTOCOL_NAME="v10.2.3 subject-overlap upper-bound personalized protocol",
        SUBJECT_POOL="all",
        SEGMENTWISE_SPLIT_COUNTS=(16, 4, 5, 5),
        FEWSHOT_SUPPORT_QUERY_COUNTS=(5, 5),
        USE_SUBJECT_CALIBRATION=True,
        SUBJECT_CALIBRATION_MODE="affine",
        SUBJECT_AFFINE_MIN_SHOTS=3,
        CALIBRATION_SHRINKAGE=2.0,
        FEW_SHOT_SWEEP=(0, 1, 3, 5),
        MODEL_SELECTION_USE_CALIBRATED_VAL=True,
        VAL_CAL_SCORE_BLEND=0.75,
        MODALITY_DROPOUT_PPG=0.05,
        MODALITY_DROPOUT_ECG=0.02,
        P_DEGRADE=0.08,
        LAMBDA_TAIL=0.28,
        LAMBDA_REG_CLS=0.42,
        REG_SAMPLER_POWER=0.82,
        REG_TO_CLASS_THRESHOLD_BLEND=0.90,
    )


def make_v10_2_optimized_cfg() -> CFG:
    project_root = Path(__file__).resolve().parent
    return CFG(
        OUTPUT_NAME="mimic_bp_reg_v10_2_opt_proto",
        PROTOCOL_ID="v10.2_opt_subject_overlap_affine_tail",
        PROTOCOL_STRICTNESS_RANK=3,
        SPLIT_PROTOCOL="subject_overlap_stratified_calibrated",
        PROTOCOL_NAME="v10.2 optimized subject-overlap stratified calibrated protocol",
        SUBJECT_POOL="all",
        SEGMENTWISE_SPLIT_COUNTS=(16, 4, 5, 5),
        FEWSHOT_SUPPORT_QUERY_COUNTS=(5, 5),
        EPOCHS=24,
        EARLY_STOPPING_PATIENCE=16,
        LR=8.0e-5,
        MIN_LR_RATIO=0.04,
        USE_SUBJECT_CALIBRATION=True,
        SUBJECT_CALIBRATION_MODE="affine",
        SUBJECT_AFFINE_MIN_SHOTS=2,
        SUBJECT_AFFINE_SCALE_CLIP=(0.82, 1.20),
        CALIBRATION_SHRINKAGE=2.0,
        FEW_SHOT_SWEEP=(0, 1, 2, 3, 5),
        MODEL_SELECTION_USE_CALIBRATED_VAL=True,
        VAL_CAL_SCORE_BLEND=0.82,
        MODALITY_DROPOUT_PPG=0.05,
        MODALITY_DROPOUT_ECG=0.02,
        P_DEGRADE=0.08,
        LAMBDA_REG_CLS=0.52,
        LAMBDA_TAIL=0.36,
        REG_SAMPLER_POWER=0.92,
        REG_TO_CLASS_THRESHOLD_BLEND=0.92,
        TAIL_CLASS_WEIGHTS=(0.0, 0.35, 0.85, 1.25),
        NORMAL_OVERPRED_WEIGHT=0.30,
        TAIL_SBP_SCALE=11.0,
        TAIL_DBP_SCALE=7.5,
        INIT_CKPT_PATH=str(project_root / "outputs" / "mimic_bp_reg_v10_2_proto" / "best_model.pt"),
    )


def make_v10_2_optimized_long_cfg() -> CFG:
    project_root = Path(__file__).resolve().parent
    resume_path = project_root / "outputs" / "mimic_bp_reg_v10_2_opt_long_proto" / "best_model.pt"
    fallback_path = project_root / "outputs" / "mimic_bp_reg_v10_2_opt_proto" / "best_model.pt"
    return CFG(
        OUTPUT_NAME="mimic_bp_reg_v10_2_opt_long_proto",
        PROTOCOL_ID="v10.2_opt_long_subject_overlap_affine_tail",
        PROTOCOL_STRICTNESS_RANK=3,
        SPLIT_PROTOCOL="subject_overlap_stratified_calibrated",
        PROTOCOL_NAME="v10.2 optimized long-finetune subject-overlap stratified calibrated protocol",
        SUBJECT_POOL="all",
        SEGMENTWISE_SPLIT_COUNTS=(16, 4, 5, 5),
        FEWSHOT_SUPPORT_QUERY_COUNTS=(5, 5),
        EPOCHS=50,
        EARLY_STOPPING_PATIENCE=20,
        LR=4.0e-5,
        MIN_LR_RATIO=0.015,
        USE_SUBJECT_CALIBRATION=True,
        SUBJECT_CALIBRATION_MODE="affine",
        SUBJECT_AFFINE_MIN_SHOTS=2,
        SUBJECT_AFFINE_SCALE_CLIP=(0.80, 1.22),
        CALIBRATION_SHRINKAGE=1.5,
        FEW_SHOT_SWEEP=(0, 1, 2, 3, 5),
        MODEL_SELECTION_USE_CALIBRATED_VAL=True,
        VAL_CAL_SCORE_BLEND=0.86,
        MODALITY_DROPOUT_PPG=0.03,
        MODALITY_DROPOUT_ECG=0.01,
        P_DEGRADE=0.05,
        LAMBDA_REG_CLS=0.58,
        LAMBDA_TAIL=0.42,
        REG_SAMPLER_POWER=0.98,
        REG_TO_CLASS_THRESHOLD_BLEND=0.93,
        TAIL_CLASS_WEIGHTS=(0.0, 0.40, 0.95, 1.35),
        NORMAL_OVERPRED_WEIGHT=0.36,
        TAIL_SBP_SCALE=10.5,
        TAIL_DBP_SCALE=7.0,
        INIT_CKPT_PATH=str(resume_path if resume_path.exists() else fallback_path),
        PLOT_COMPOSITE_TRAINING_CURVES=False,
        SAVE_SPLIT_TRAINING_CURVES=True,
    )


def _build_subject_disjoint_fewshot_loaders(cfg: CFG, task: str = "regression") -> ProtocolLoaders:
    subjects = list(v11._subject_pool(cfg))
    if not subjects:
        raise FileNotFoundError("No subjects found under DATA_ROOT for subject-disjoint few-shot protocol.")
    rs = np.random.RandomState(cfg.PROTOCOL_SEED)
    rs.shuffle(subjects)
    split_counts = v11._allocate_counts(len(subjects), cfg.SUBJECTWISE_SPLIT_RATIOS)
    if len(split_counts) != 3:
        raise ValueError("SUBJECTWISE_SPLIT_RATIOS must contain three values: train/val/test.")
    n_train, n_val, n_test = [int(x) for x in split_counts.tolist()]
    train_subjects = sorted(subjects[:n_train])
    val_subjects = sorted(subjects[n_train:n_train + n_val])
    test_subjects = sorted(subjects[n_train + n_val:n_train + n_val + n_test])

    n_support, n_query = [int(x) for x in cfg.FEWSHOT_SUPPORT_QUERY_COUNTS]
    if (n_support + n_query) > 30:
        raise ValueError("FEWSHOT_SUPPORT_QUERY_COUNTS must sum to <= 30 for MIMIC-BP subjects.")

    train_entries = _all_entries_for_subjects(cfg, train_subjects)
    _, _, val_support_entries, val_query_entries = v11._segmentwise_entries_for_subjects(
        cfg, val_subjects, (0, 0, n_support, n_query), cfg.PROTOCOL_SEED + 1000
    )
    _, _, test_support_entries, test_query_entries = v11._segmentwise_entries_for_subjects(
        cfg, test_subjects, (0, 0, n_support, n_query), cfg.PROTOCOL_SEED + 2000
    )

    ds_train = v11.IndexedMIMICBPDataset(cfg, train_entries, cfg.CROP_LEN, mode="train", seed=cfg.SEED)
    ds_val_support = v11.IndexedMIMICBPDataset(cfg, val_support_entries, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)
    ds_val_query = v11.IndexedMIMICBPDataset(cfg, val_query_entries, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)
    ds_test_support = v11.IndexedMIMICBPDataset(cfg, test_support_entries, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)
    ds_test_query = v11.IndexedMIMICBPDataset(cfg, test_query_entries, cfg.CROP_LEN, mode="eval", seed=cfg.SEED)

    manifest = build_protocol_manifest(cfg)
    _save_protocol_artifacts(
        cfg,
        manifest,
        {
            "train": train_subjects,
            "val": val_subjects,
            "test": test_subjects,
        },
        {
            "train": train_entries,
            "val_support": val_support_entries,
            "val_query": val_query_entries,
            "test_support": test_support_entries,
            "test_query": test_query_entries,
        },
    )
    return ProtocolLoaders(
        ds_train=ds_train,
        train_loader=_make_train_loader(ds_train, cfg, task),
        val_query_loader=_make_eval_loader(ds_val_query, cfg),
        val_calib_loader=_make_eval_loader(ds_val_support, cfg),
        test_query_loader=_make_eval_loader(ds_test_query, cfg),
        test_calib_loader=_make_eval_loader(ds_test_support, cfg),
        split_datasets={
            "train": ds_train,
            "val_support": ds_val_support,
            "val_query": ds_val_query,
            "test_support": ds_test_support,
            "test_query": ds_test_query,
        },
        manifest=manifest,
    )


def build_protocol_loaders(cfg: CFG, task: str = "regression") -> ProtocolLoaders:
    if cfg.SPLIT_PROTOCOL == "official_subjectwise":
        ds_train, train_loader, val_loader, calib_loader, test_loader = shared.build_loaders(cfg, task=task)
        manifest = build_protocol_manifest(cfg)
        return ProtocolLoaders(
            ds_train=ds_train,
            train_loader=train_loader,
            val_query_loader=val_loader,
            val_calib_loader=calib_loader,
            test_query_loader=test_loader,
            test_calib_loader=calib_loader,
            split_datasets={
                "train": ds_train,
                "val": val_loader.dataset,
                "calib": calib_loader.dataset,
                "test": test_loader.dataset,
            },
            manifest=manifest,
        )
    if cfg.SPLIT_PROTOCOL == "subject_disjoint_fewshot_personalized":
        return _build_subject_disjoint_fewshot_loaders(cfg, task=task)
    if cfg.SPLIT_PROTOCOL == "subject_overlap_stratified_calibrated":
        ds_train, train_loader, val_loader, calib_loader, test_loader = shared.build_loaders(cfg, task=task)
        manifest = build_protocol_manifest(cfg)
        return ProtocolLoaders(
            ds_train=ds_train,
            train_loader=train_loader,
            val_query_loader=val_loader,
            val_calib_loader=calib_loader,
            test_query_loader=test_loader,
            test_calib_loader=calib_loader,
            split_datasets={
                "train": ds_train,
                "val": val_loader.dataset,
                "calib": calib_loader.dataset,
                "test": test_loader.dataset,
            },
            manifest=manifest,
        )
    raise ValueError(f"Unsupported v10.2 multi-protocol SPLIT_PROTOCOL: {cfg.SPLIT_PROTOCOL}")
