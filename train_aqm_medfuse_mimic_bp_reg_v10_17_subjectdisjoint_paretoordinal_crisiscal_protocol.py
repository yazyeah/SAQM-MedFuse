from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

import train_aqm_medfuse_mimic_bp_reg_v10_16_subjectdisjoint_clinicalordinal_biasbalanced_protocol as v16


FINAL_OUTPUT_NAME = "mimic_bp_reg_v10_17_subjectdisjoint_paretoordinal_crisiscal_proto"
FINAL_PROTOCOL_ID = "v10.17_subjectdisjoint_paretoordinal_crisiscal"
OPTLONG_FULLTRAIN_OUTPUT = v16.OPTLONG_FULLTRAIN_OUTPUT
DUALMAX_FULLTRAIN_OUTPUT = v16.DUALMAX_FULLTRAIN_OUTPUT
V10_17_SCORE_VERSION = "v10.17_paretoordinal_crisiscal_20260427"

v13 = v16.v13
v12 = v16.v12
meta_script = v16.meta_script
guided_script = meta_script.guided_script

_ORIG_V16_FINAL_OUTPUT_NAME = v16.FINAL_OUTPUT_NAME
_ORIG_V16_FINAL_PROTOCOL_ID = v16.FINAL_PROTOCOL_ID
_ORIG_V16_OPTLONG_FULLTRAIN_OUTPUT = v16.OPTLONG_FULLTRAIN_OUTPUT
_ORIG_V16_DUALMAX_FULLTRAIN_OUTPUT = v16.DUALMAX_FULLTRAIN_OUTPUT
_ORIG_V16_BUILD_NEXTGEN_CFG = v16.build_nextgen_cfg
_ORIG_V16_GENERATE_EXTRA_OUTPUTS = v16.generate_extra_outputs
_ORIG_V16_SEARCH_CLASSIFICATION_ARBITER = v16.search_classification_arbiter_with_progress
_ORIG_V16_VECTOR_HIGH_BIAS_ROWS = v16._vectorized_high_bias_rows
_ORIG_V16_VECTOR_CRISIS_TAIL_ROWS = v16._vectorized_crisis_tail_rows
_ORIG_V16_SEARCH_HIGH_BIAS = v16.tailaware_search_high_bias_calibration_candidates
_ORIG_V16_SEARCH_CRISIS_TAIL = v16.search_crisis_tail_debias_candidates
_ORIG_V16_SAFETY_SCORE = v16.guarded_safety_class_fusion_score
_ORIG_V16_SEARCH_SAFETY = v16.guarded_search_safety_class_fusion
_ORIG_V16_MERGE_TOP_ROWS = v16._merge_top_rows
_ORIG_V16_HIGH_BIAS_SIGNATURE = v16._high_bias_signature
_ORIG_V16_CRISIS_SIGNATURE = v16._crisis_signature
_ORIG_V13_CLASS_SELECTION_SCORE = v13.tailaware_classification_selection_score
_ORIG_V13_CLASS_CANDIDATE_SCORE = v13.tailaware_classification_candidate_score
_ORIG_V13_ROBUST_SCORE = v13.tailaware_robust_classification_score
_ORIG_GUIDED_SEARCH_POLICY = guided_script.search_policy
_ORIG_GUIDED_CLASSIFICATION_CANDIDATE_SCORE = guided_script.classification_candidate_score
_ORIG_META_PREV_ROBUST_SCORE = meta_script.prev_script.robust_classification_score

_ACTIVE_CFG = None
_BASELINE_TARGET_CACHE = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _output_dir() -> Path:
    return _project_root() / "outputs" / FINAL_OUTPUT_NAME


def _tables_dir() -> Path:
    path = _output_dir() / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _figures_dir() -> Path:
    path = _output_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifacts_dir() -> Path:
    path = _output_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _row_float(row: dict, key: str, default: float = 0.0) -> float:
    return _to_float(row.get(key), default)


def _current_cfg():
    cfg = _ACTIVE_CFG
    if cfg is None:
        cfg = getattr(v13.base_script, "_ACTIVE_CFG", None)
    if cfg is None:
        cfg = build_nextgen_cfg()
    return cfg


def _baseline_targets() -> dict:
    global _BASELINE_TARGET_CACHE
    if _BASELINE_TARGET_CACHE is not None:
        return dict(_BASELINE_TARGET_CACHE)
    try:
        baseline = dict(v13._baseline_targets())
    except Exception:
        baseline = {
            "selected_acc": 0.8122807017543859,
            "selected_macro_f1": 0.7334048956487922,
            "selected_balanced_acc": 0.751738213088013,
            "selected_mae_mean": 6.086580038070679,
        }
    _BASELINE_TARGET_CACHE = dict(baseline)
    return dict(baseline)


def _baseline_metric(name: str, default: float) -> float:
    baseline = _baseline_targets()
    return float(baseline.get(name, default))


def _metric_from(metrics: dict, prefix: str, name: str, default: float = 0.0) -> float:
    aliases = {
        "acc": (f"cls_acc_{prefix}", f"acc_{prefix}", "acc"),
        "macro_f1": (f"cls_f1_macro_{prefix}", f"macro_f1_{prefix}", "macro_f1"),
        "balanced_acc": (f"cls_balanced_acc_{prefix}", f"balanced_acc_{prefix}", "balanced_acc"),
        "ece": (f"cls_ece_{prefix}", f"ece_{prefix}", "ece"),
    }
    for key in aliases.get(name, (f"cls_{name}_{prefix}", f"{name}_{prefix}", name)):
        if key in metrics:
            return _to_float(metrics.get(key), default)
    return float(default)


def _class_metric(metrics: dict, prefix: str, metric: str, class_name: str, default: float = 0.0) -> float:
    return _to_float(metrics.get(f"cls_{metric}_{prefix}_{class_name}"), default)


def _class_names(cfg) -> tuple[str, str, str, str]:
    names = tuple(str(x) for x in getattr(cfg, "CLASS_NAMES", ("Normal", "Elevated", "Stage1", "Stage2")))
    if len(names) >= 4:
        return names[:4]
    return ("Normal", "Elevated", "Stage1", "Stage2")


def _summary_from_metrics(metrics: dict, prefix: str, cfg=None) -> dict:
    try:
        summary = dict(v13.prev_script._target_class_summary(metrics, prefix))
    except Exception:
        normal, elevated, stage1, stage2 = _class_names(cfg)
        rare = [
            _class_metric(metrics, prefix, "f1", elevated),
            _class_metric(metrics, prefix, "f1", stage1),
            _class_metric(metrics, prefix, "f1", stage2),
        ]
        summary = {
            "acc": _metric_from(metrics, prefix, "acc"),
            "macro_f1": _metric_from(metrics, prefix, "macro_f1"),
            "balanced_acc": _metric_from(metrics, prefix, "balanced_acc"),
            "ece": _metric_from(metrics, prefix, "ece"),
            "rare_f1_mean": float(np.mean(rare)),
            "rare_f1_min": float(np.min(rare)),
            "elevated_f1": rare[0],
            "stage1_f1": rare[1],
            "stage2_f1": rare[2],
        }
    for key in (
        "acc",
        "macro_f1",
        "balanced_acc",
        "ece",
        "rare_f1_mean",
        "rare_f1_min",
        "elevated_f1",
        "stage1_f1",
        "stage2_f1",
    ):
        summary[key] = _to_float(summary.get(key), 0.0)
    return summary


def _row_class_summary(row: dict, cfg=None, prefix: str = "selected_val") -> dict:
    normal, elevated, stage1, stage2 = _class_names(cfg)
    elevated_f1 = _to_float(
        row.get("elevated_f1", row.get(f"cls_f1_{prefix}_{elevated}", row.get("cls_f1", 0.0))),
        0.0,
    )
    stage1_f1 = _to_float(row.get("stage1_f1", row.get(f"cls_f1_{prefix}_{stage1}", 0.0)), 0.0)
    stage2_f1 = _to_float(row.get("stage2_f1", row.get(f"cls_f1_{prefix}_{stage2}", 0.0)), 0.0)
    rare_values = [elevated_f1, stage1_f1, stage2_f1]
    return {
        "acc": _to_float(row.get("acc", row.get(f"cls_acc_{prefix}")), 0.0),
        "macro_f1": _to_float(row.get("macro_f1", row.get(f"cls_f1_macro_{prefix}")), 0.0),
        "balanced_acc": _to_float(row.get("balanced_acc", row.get(f"cls_balanced_acc_{prefix}")), 0.0),
        "ece": _to_float(row.get("ece", row.get(f"cls_ece_{prefix}")), 0.0),
        "rare_f1_mean": _to_float(row.get("rare_f1_mean"), float(np.mean(rare_values))),
        "rare_f1_min": _to_float(row.get("rare_f1_min"), float(np.min(rare_values))),
        "elevated_f1": elevated_f1,
        "stage1_f1": stage1_f1,
        "stage2_f1": stage2_f1,
        "robust_min_f1": _to_float(row.get("robust_min_f1"), 0.0),
    }


def _classification_pareto_score(summary: dict, cfg, row: dict | None = None) -> float:
    acc = float(summary["acc"])
    macro_f1 = float(summary["macro_f1"])
    balanced = float(summary["balanced_acc"])
    ece = float(summary["ece"])
    rare_mean = float(summary.get("rare_f1_mean", 0.0))
    rare_min = float(summary.get("rare_f1_min", 0.0))
    stage2 = float(summary.get("stage2_f1", 0.0))
    robust_min = float(summary.get("robust_min_f1", 0.0))
    if row is not None:
        robust_min = max(robust_min, _to_float(row.get("robust_min_f1"), 0.0))

    baseline_acc = _baseline_metric("selected_acc", 0.8122807017543859)
    baseline_f1 = _baseline_metric("selected_macro_f1", 0.7334048956487922)
    baseline_bal = _baseline_metric("selected_balanced_acc", 0.751738213088013)
    min_acc = baseline_acc + float(getattr(cfg, "BASELINE_MIN_ACC_MARGIN", 0.0))
    min_f1 = baseline_f1 + float(getattr(cfg, "BASELINE_MIN_F1_MARGIN", 0.0))
    min_bal = baseline_bal + float(getattr(cfg, "BASELINE_MIN_BAL_MARGIN", 0.0))
    asp_acc = float(getattr(cfg, "BASELINE_ASPIRATIONAL_ACC", 0.86))
    asp_f1 = float(getattr(cfg, "BASELINE_ASPIRATIONAL_F1", 0.80))

    acc_gap = max(0.0, min_acc - acc)
    f1_gap = max(0.0, min_f1 - macro_f1)
    bal_gap = max(0.0, min_bal - balanced)
    asp_acc_gap = max(0.0, asp_acc - acc)
    asp_f1_gap = max(0.0, asp_f1 - macro_f1)
    rare_gap = max(0.0, float(getattr(cfg, "CLASSIFIER_RARE_MIN_TARGET", 0.57)) - rare_min)
    robust_gap = max(0.0, float(getattr(cfg, "CLASSIFIER_ROBUST_MIN_TARGET", 0.59)) - robust_min)

    return float(
        10.0 * acc
        + 14.0 * macro_f1
        + 5.0 * balanced
        + 1.4 * rare_min
        + 0.8 * rare_mean
        + 0.65 * stage2
        + 0.75 * robust_min
        - 0.40 * ece
        + 2.0 * max(0.0, acc - baseline_acc)
        + 3.0 * max(0.0, macro_f1 - baseline_f1)
        + 1.2 * max(0.0, balanced - baseline_bal)
        - float(getattr(cfg, "BASELINE_ACC_SHORTFALL_WEIGHT", 140.0)) * acc_gap
        - float(getattr(cfg, "BASELINE_F1_SHORTFALL_WEIGHT", 170.0)) * f1_gap
        - float(getattr(cfg, "BASELINE_BAL_SHORTFALL_WEIGHT", 90.0)) * bal_gap
        - float(getattr(cfg, "BASELINE_ASPIRATIONAL_ACC_WEIGHT", 5.5)) * asp_acc_gap
        - float(getattr(cfg, "BASELINE_ASPIRATIONAL_F1_WEIGHT", 7.0)) * asp_f1_gap
        - 14.0 * rare_gap
        - 8.0 * robust_gap
    )


def _classification_pareto_score_batch(
    acc: np.ndarray,
    macro_f1: np.ndarray,
    balanced: np.ndarray,
    ece: np.ndarray,
    rare_min: np.ndarray,
    rare_mean: np.ndarray,
    stage2: np.ndarray,
    robust_min: np.ndarray | float,
    cfg,
) -> np.ndarray:
    baseline_acc = _baseline_metric("selected_acc", 0.8122807017543859)
    baseline_f1 = _baseline_metric("selected_macro_f1", 0.7334048956487922)
    baseline_bal = _baseline_metric("selected_balanced_acc", 0.751738213088013)
    min_acc = baseline_acc + float(getattr(cfg, "BASELINE_MIN_ACC_MARGIN", 0.0))
    min_f1 = baseline_f1 + float(getattr(cfg, "BASELINE_MIN_F1_MARGIN", 0.0))
    min_bal = baseline_bal + float(getattr(cfg, "BASELINE_MIN_BAL_MARGIN", 0.0))
    asp_acc = float(getattr(cfg, "BASELINE_ASPIRATIONAL_ACC", 0.86))
    asp_f1 = float(getattr(cfg, "BASELINE_ASPIRATIONAL_F1", 0.80))
    robust = np.asarray(robust_min, dtype=np.float64)
    acc_gap = np.maximum(0.0, min_acc - acc)
    f1_gap = np.maximum(0.0, min_f1 - macro_f1)
    bal_gap = np.maximum(0.0, min_bal - balanced)
    asp_acc_gap = np.maximum(0.0, asp_acc - acc)
    asp_f1_gap = np.maximum(0.0, asp_f1 - macro_f1)
    rare_gap = np.maximum(0.0, float(getattr(cfg, "CLASSIFIER_RARE_MIN_TARGET", 0.57)) - rare_min)
    robust_gap = np.maximum(0.0, float(getattr(cfg, "CLASSIFIER_ROBUST_MIN_TARGET", 0.59)) - robust)
    return (
        10.0 * acc
        + 14.0 * macro_f1
        + 5.0 * balanced
        + 1.4 * rare_min
        + 0.8 * rare_mean
        + 0.65 * stage2
        + 0.75 * robust
        - 0.40 * ece
        + 2.0 * np.maximum(0.0, acc - baseline_acc)
        + 3.0 * np.maximum(0.0, macro_f1 - baseline_f1)
        + 1.2 * np.maximum(0.0, balanced - baseline_bal)
        - float(getattr(cfg, "BASELINE_ACC_SHORTFALL_WEIGHT", 140.0)) * acc_gap
        - float(getattr(cfg, "BASELINE_F1_SHORTFALL_WEIGHT", 170.0)) * f1_gap
        - float(getattr(cfg, "BASELINE_BAL_SHORTFALL_WEIGHT", 90.0)) * bal_gap
        - float(getattr(cfg, "BASELINE_ASPIRATIONAL_ACC_WEIGHT", 5.5)) * asp_acc_gap
        - float(getattr(cfg, "BASELINE_ASPIRATIONAL_F1_WEIGHT", 7.0)) * asp_f1_gap
        - 14.0 * rare_gap
        - 8.0 * robust_gap
    )


def _classification_gate(summary: dict, cfg) -> dict:
    baseline_acc = _baseline_metric("selected_acc", 0.8122807017543859)
    baseline_f1 = _baseline_metric("selected_macro_f1", 0.7334048956487922)
    baseline_bal = _baseline_metric("selected_balanced_acc", 0.751738213088013)
    min_acc = baseline_acc + float(getattr(cfg, "BASELINE_MIN_ACC_MARGIN", 0.0))
    min_f1 = baseline_f1 + float(getattr(cfg, "BASELINE_MIN_F1_MARGIN", 0.0))
    min_bal = baseline_bal + float(getattr(cfg, "BASELINE_MIN_BAL_MARGIN", 0.0))
    acc = float(summary["acc"])
    macro_f1 = float(summary["macro_f1"])
    balanced = float(summary["balanced_acc"])
    return {
        "classification_gate_pass": bool(acc >= min_acc and macro_f1 >= min_f1 and balanced >= min_bal),
        "baseline_acc": float(baseline_acc),
        "baseline_macro_f1": float(baseline_f1),
        "baseline_balanced_acc": float(baseline_bal),
        "baseline_min_acc": float(min_acc),
        "baseline_min_macro_f1": float(min_f1),
        "baseline_min_balanced_acc": float(min_bal),
        "baseline_acc_gap": float(max(0.0, min_acc - acc)),
        "baseline_macro_f1_gap": float(max(0.0, min_f1 - macro_f1)),
        "baseline_balanced_acc_gap": float(max(0.0, min_bal - balanced)),
        "baseline_acc_delta": float(acc - baseline_acc),
        "baseline_macro_f1_delta": float(macro_f1 - baseline_f1),
        "baseline_balanced_acc_delta": float(balanced - baseline_bal),
    }


def _annotate_classification_rows(rows: Iterable[dict], cfg, selected_name: str | None = None) -> List[dict]:
    annotated_rows: List[dict] = []
    for row in rows:
        annotated = dict(row)
        summary = _row_class_summary(annotated, cfg, "selected_val")
        gate = _classification_gate(summary, cfg)
        legacy_score = _to_float(annotated.get("score"), 0.0)
        pareto_score = _classification_pareto_score(summary, cfg, annotated)
        annotated.update(
            {
                "legacy_score": float(legacy_score),
                "pareto_score": float(pareto_score),
                "score": float(pareto_score),
                **gate,
            }
        )
        if selected_name is not None:
            annotated["classification_gate_selected"] = int(str(annotated.get("candidate", "")) == selected_name)
        annotated_rows.append(annotated)
    annotated_rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return annotated_rows


def pareto_classification_selection_score(metrics: dict, prefix: str) -> float:
    try:
        legacy = float(_ORIG_V13_CLASS_SELECTION_SCORE(metrics, prefix))
    except Exception:
        legacy = 0.0
    cfg = _current_cfg()
    summary = _summary_from_metrics(metrics, prefix, cfg)
    return float(0.55 * legacy + _classification_pareto_score(summary, cfg))


def pareto_classification_candidate_score(metrics: dict, prefix: str) -> float:
    return float(pareto_classification_selection_score(metrics, prefix))


def pareto_robust_classification_score(
    clean_metrics: dict,
    noise_metrics: dict,
    ecg_metrics: dict,
    ppg_metrics: dict,
    cfg,
) -> float:
    clean_summary = _summary_from_metrics(clean_metrics, "selected_val", cfg)
    noise_f1 = _metric_from(noise_metrics, "selected_noise_val", "macro_f1")
    ecg_f1 = _metric_from(ecg_metrics, "selected_ecg_val", "macro_f1")
    ppg_f1 = _metric_from(ppg_metrics, "selected_ppg_val", "macro_f1")
    robust_min = min(noise_f1, ecg_f1, ppg_f1)
    clean_summary["robust_min_f1"] = robust_min
    robust_gap = max(0.0, float(getattr(cfg, "CLASSIFIER_ROBUST_MIN_TARGET", 0.59)) - robust_min)
    return float(
        _classification_pareto_score(clean_summary, cfg)
        + 0.22 * noise_f1
        + 0.22 * ecg_f1
        + 0.26 * ppg_f1
        + 0.28 * robust_min
        - 8.0 * robust_gap
    )


def guided_classification_candidate_score(metrics: dict, prefix: str) -> float:
    cfg = _current_cfg()
    summary = _summary_from_metrics(metrics, prefix, cfg)
    return float(_classification_pareto_score(summary, cfg))


def _grid(cfg, name: str, default: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(x) for x in getattr(cfg, name, default))
    return values if values else tuple(float(x) for x in default)


def _batched_policy_ece(y_true: np.ndarray, pred: np.ndarray, adj_prob: np.ndarray, cfg) -> np.ndarray:
    bins = int(getattr(cfg, "ECE_BINS", 10))
    n = max(int(y_true.shape[0]), 1)
    conf = adj_prob.max(axis=2)
    correct = pred == y_true[None, :]
    ece = np.zeros(pred.shape[0], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        count = mask.sum(axis=1).astype(np.float64)
        nz = count > 0.0
        if not np.any(nz):
            continue
        bin_acc = (correct & mask).sum(axis=1).astype(np.float64)
        bin_conf = (conf * mask).sum(axis=1).astype(np.float64)
        bin_acc[nz] /= count[nz]
        bin_conf[nz] /= count[nz]
        ece[nz] += (count[nz] / float(n)) * np.abs(bin_acc[nz] - bin_conf[nz])
    return ece


def _guided_metrics_row(
    cfg,
    gamma: float,
    w_elevated: float,
    w_stage1: float,
    w_stage2: float,
    acc: float,
    balanced_acc: float,
    macro_f1: float,
    ece: float,
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    support: np.ndarray,
) -> dict:
    metrics = {
        "cls_acc_guided_calib": float(acc),
        "cls_balanced_acc_guided_calib": float(balanced_acc),
        "cls_f1_macro_guided_calib": float(macro_f1),
        "cls_ece_guided_calib": float(ece),
    }
    rare = f1[1:4]
    summary = {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(balanced_acc),
        "ece": float(ece),
        "rare_f1_mean": float(np.mean(rare)),
        "rare_f1_min": float(np.min(rare)),
        "elevated_f1": float(f1[1]),
        "stage1_f1": float(f1[2]),
        "stage2_f1": float(f1[3]),
        "robust_min_f1": 0.0,
    }
    for idx, name in enumerate(_class_names(cfg)):
        metrics[f"cls_precision_guided_calib_{name}"] = float(precision[idx])
        metrics[f"cls_recall_guided_calib_{name}"] = float(recall[idx])
        metrics[f"cls_f1_guided_calib_{name}"] = float(f1[idx])
        metrics[f"cls_support_guided_calib_{name}"] = int(support[idx])
    score = _classification_pareto_score(summary, cfg)
    return {
        "gamma": float(gamma),
        "w_normal": 1.0,
        "w_elevated": float(w_elevated),
        "w_stage1": float(w_stage1),
        "w_stage2": float(w_stage2),
        "score": float(score),
        **metrics,
    }


def configurable_search_policy(prob: np.ndarray, y_true: np.ndarray, cfg):
    rows = []
    best_row = None
    best_params = None
    best_score_seen = -float("inf")
    materialize_rows = bool(getattr(cfg, "GUIDED_POLICY_MATERIALIZE_ROWS", True))
    gamma_grid = _grid(cfg, "GUIDED_POLICY_GAMMAS", (0.90, 1.00, 1.10, 1.20, 1.30))
    w_elevated_grid = _grid(cfg, "GUIDED_POLICY_ELEVATED_WEIGHTS", (1.00, 1.10, 1.25, 1.40))
    w_stage1_grid = _grid(cfg, "GUIDED_POLICY_STAGE1_WEIGHTS", (1.00, 1.15, 1.30, 1.50))
    w_stage2_grid = _grid(cfg, "GUIDED_POLICY_STAGE2_WEIGHTS", (1.00, 1.20, 1.40, 1.60))
    prob = np.clip(np.asarray(prob, dtype=np.float32), 1.0e-8, None)
    y_true = np.asarray(y_true, dtype=np.int64)
    n_classes = int(getattr(cfg, "N_CLASSES", prob.shape[1]))
    support = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    support_safe = np.maximum(support, 1.0)
    batch_size = max(32, int(getattr(cfg, "GUIDED_POLICY_BATCH_SIZE", 512)))
    weight_pairs = np.asarray(
        [
            (float(w_elevated), float(w_stage1), float(w_stage2))
            for w_elevated in w_elevated_grid
            for w_stage1 in w_stage1_grid
            for w_stage2 in w_stage2_grid
        ],
        dtype=np.float32,
    )
    for gamma in gamma_grid:
        base = np.power(prob, float(gamma), dtype=np.float32)
        for start in range(0, weight_pairs.shape[0], batch_size):
            stop = min(start + batch_size, weight_pairs.shape[0])
            weights = np.ones((stop - start, n_classes), dtype=np.float32)
            weights[:, 1] = weight_pairs[start:stop, 0]
            weights[:, 2] = weight_pairs[start:stop, 1]
            weights[:, 3] = weight_pairs[start:stop, 2]
            weighted = base[None, :, :] * weights[:, None, :]
            denom = np.clip(weighted.sum(axis=2, keepdims=True), 1.0e-8, None)
            adj_prob = weighted / denom
            pred = adj_prob.argmax(axis=2).astype(np.int64)
            correct = pred == y_true[None, :]
            acc = correct.mean(axis=1).astype(np.float64)
            tp = np.stack(
                [((pred == cls_idx) & (y_true[None, :] == cls_idx)).sum(axis=1) for cls_idx in range(n_classes)],
                axis=1,
            ).astype(np.float64)
            pred_count = np.stack([(pred == cls_idx).sum(axis=1) for cls_idx in range(n_classes)], axis=1).astype(np.float64)
            precision = np.divide(tp, np.maximum(pred_count, 1.0), out=np.zeros_like(tp), where=pred_count > 0.0)
            recall = tp / support_safe[None, :]
            denom_f1 = precision + recall
            f1 = np.divide(2.0 * precision * recall, denom_f1, out=np.zeros_like(denom_f1), where=denom_f1 > 0.0)
            macro_f1 = f1.mean(axis=1)
            balanced_acc = recall.mean(axis=1)
            ece = _batched_policy_ece(y_true, pred, adj_prob, cfg)
            rare = f1[:, 1:4]
            rare_mean = rare.mean(axis=1)
            rare_min = rare.min(axis=1)
            score_batch = _classification_pareto_score_batch(
                acc,
                macro_f1,
                balanced_acc,
                ece,
                rare_min,
                rare_mean,
                f1[:, 3],
                0.0,
                cfg,
            )
            local_best_idx = int(np.argmax(score_batch))
            local_best_score = float(score_batch[local_best_idx])
            if local_best_score > best_score_seen:
                w_elevated, w_stage1, w_stage2 = weight_pairs[start + local_best_idx]
                best_score_seen = local_best_score
                best_params = (float(gamma), float(w_elevated), float(w_stage1), float(w_stage2))
            for local_idx in range(stop - start):
                if not materialize_rows:
                    continue
                w_elevated, w_stage1, w_stage2 = weight_pairs[start + local_idx]
                row = _guided_metrics_row(
                    cfg,
                    float(gamma),
                    float(w_elevated),
                    float(w_stage1),
                    float(w_stage2),
                    float(acc[local_idx]),
                    float(balanced_acc[local_idx]),
                    float(macro_f1[local_idx]),
                    float(ece[local_idx]),
                    precision[local_idx],
                    recall[local_idx],
                    f1[local_idx],
                    support,
                )
                row["score"] = float(score_batch[local_idx])
                rows.append(row)
                if best_row is None or float(row["score"]) > float(best_row["score"]):
                    best_row = row
    if best_params is None:
        raise RuntimeError("No guided classification policy candidate was evaluated.")
    if best_row is None:
        gamma, w_elevated, w_stage1, w_stage2 = best_params
        best_row = {
            "gamma": float(gamma),
            "w_normal": 1.0,
            "w_elevated": float(w_elevated),
            "w_stage1": float(w_stage1),
            "w_stage2": float(w_stage2),
            "score": float(best_score_seen),
        }
    best_prob = guided_script.apply_policy(
        prob,
        float(best_row["gamma"]),
        (1.0, float(best_row["w_elevated"]), float(best_row["w_stage1"]), float(best_row["w_stage2"])),
    )
    best_pred = best_prob.argmax(axis=1).astype(np.int64)
    best_metrics = meta_script.stage_script.risk_classification_metrics(
        y_true,
        best_pred,
        best_prob,
        cfg,
        prefix="guided_calib",
    )
    best_score = guided_classification_candidate_score(best_metrics, "guided_calib")
    best_row.update(best_metrics)
    best_row["score"] = float(best_score)
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    if rows:
        rows[0].update(best_row)
    else:
        rows = [dict(best_row)]
    return best_row, rows


def _reg_target(cfg, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _regression_pareto_terms(row: dict, cfg, base_max_delta_attr: str, base_cov_delta_attr: str) -> dict:
    mae = _row_float(row, "mae_mean")
    global_bias = abs(_row_float(row, "bias_sbp")) + 0.70 * abs(_row_float(row, "bias_dbp"))
    high_sbp = _row_float(row, "high_bias_sbp")
    high_dbp = _row_float(row, "high_bias_dbp")
    crisis_sbp = _row_float(row, "crisis_bias_sbp")
    crisis_dbp = _row_float(row, "crisis_bias_dbp")
    coverage_gap = _row_float(row, "coverage_gap")
    mae_target = _reg_target(cfg, "REGRESSION_PARETO_MAE_TARGET", 5.70)
    coverage_target = _reg_target(cfg, "REGRESSION_PARETO_COVERAGE_GAP_MAX", 0.16)
    high_sbp_target = _reg_target(cfg, "HIGH_BIAS_TARGET_ABS_SBP", 4.0)
    high_dbp_target = _reg_target(cfg, "HIGH_BIAS_TARGET_ABS_DBP", 3.0)
    crisis_sbp_target = _reg_target(cfg, "CRISIS_BIAS_TARGET_ABS_SBP", 5.0)
    crisis_dbp_target = _reg_target(cfg, "CRISIS_BIAS_TARGET_ABS_DBP", 4.0)
    return {
        "mae": mae,
        "global_bias": global_bias,
        "coverage_gap": coverage_gap,
        "mae_excess": max(0.0, mae - mae_target),
        "coverage_excess": max(0.0, coverage_gap - coverage_target),
        "high_abs_sbp": abs(high_sbp),
        "high_abs_dbp": abs(high_dbp),
        "high_under_sbp": max(0.0, -high_sbp),
        "high_under_dbp": max(0.0, -high_dbp),
        "high_abs_excess_sbp": max(0.0, abs(high_sbp) - high_sbp_target),
        "high_abs_excess_dbp": max(0.0, abs(high_dbp) - high_dbp_target),
        "crisis_abs_sbp": abs(crisis_sbp),
        "crisis_abs_dbp": abs(crisis_dbp),
        "crisis_under_sbp": max(0.0, -crisis_sbp),
        "crisis_under_dbp": max(0.0, -crisis_dbp),
        "crisis_abs_excess_sbp": max(0.0, abs(crisis_sbp) - crisis_sbp_target),
        "crisis_abs_excess_dbp": max(0.0, abs(crisis_dbp) - crisis_dbp_target),
        "base_mae_delta_limit": _reg_target(cfg, base_max_delta_attr, 0.10),
        "base_cov_delta_limit": _reg_target(cfg, base_cov_delta_attr, 0.03),
    }


def _high_bias_row_score(row: dict, cfg) -> float:
    terms = _regression_pareto_terms(
        row,
        cfg,
        "RELIABILITY_BIAS_MAX_MAE_DELTA",
        "RELIABILITY_BIAS_MAX_COVERAGE_GAP_DELTA",
    )
    shift_pen = 0.035 * abs(_row_float(row, "shift_mean_sbp")) + 0.025 * abs(_row_float(row, "shift_mean_dbp"))
    return float(
        0.14 * terms["mae"]
        + 0.55 * terms["global_bias"]
        + 0.80 * terms["high_abs_sbp"]
        + 0.45 * terms["high_abs_dbp"]
        + 1.85 * terms["crisis_abs_sbp"]
        + 0.95 * terms["crisis_abs_dbp"]
        + 1.40 * terms["high_under_sbp"]
        + 0.70 * terms["high_under_dbp"]
        + 2.65 * terms["crisis_under_sbp"]
        + 1.20 * terms["crisis_under_dbp"]
        + 38.0 * terms["mae_excess"]
        + 14.0 * terms["coverage_excess"]
        + 2.00 * terms["high_abs_excess_sbp"]
        + 1.20 * terms["high_abs_excess_dbp"]
        + 5.50 * terms["crisis_abs_excess_sbp"]
        + 2.60 * terms["crisis_abs_excess_dbp"]
        + shift_pen
    )


def _crisis_tail_row_score(row: dict, cfg) -> float:
    terms = _regression_pareto_terms(
        row,
        cfg,
        "CRISIS_TAIL_FUSION_MAX_MAE_DELTA",
        "CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA",
    )
    guard_activation = _row_float(row, "guard_activation_rate")
    guard_shift = _row_float(row, "guard_shift_mean_sbp")
    guard_overuse = max(0.0, guard_activation - 0.24) * max(0.0, guard_shift - 1.8)
    shift_pen = 0.025 * abs(_row_float(row, "shift_mean_sbp")) + 0.018 * abs(_row_float(row, "shift_mean_dbp"))
    return float(
        0.12 * terms["mae"]
        + 0.45 * terms["global_bias"]
        + 0.55 * terms["high_abs_sbp"]
        + 0.35 * terms["high_abs_dbp"]
        + 3.00 * terms["crisis_abs_sbp"]
        + 1.45 * terms["crisis_abs_dbp"]
        + 1.35 * terms["high_under_sbp"]
        + 0.72 * terms["high_under_dbp"]
        + 3.30 * terms["crisis_under_sbp"]
        + 1.55 * terms["crisis_under_dbp"]
        + 42.0 * terms["mae_excess"]
        + 16.0 * terms["coverage_excess"]
        + 1.25 * terms["high_abs_excess_sbp"]
        + 0.75 * terms["high_abs_excess_dbp"]
        + 7.20 * terms["crisis_abs_excess_sbp"]
        + 3.20 * terms["crisis_abs_excess_dbp"]
        + 2.60 * guard_overuse
        + shift_pen
    )


def _annotate_regression_rows(rows: Iterable[dict], cfg, scorer, selected_name: str | None = None) -> List[dict]:
    annotated_rows: List[dict] = []
    for row in rows:
        annotated = dict(row)
        legacy_score = _to_float(annotated.get("score"), 0.0)
        score = float(scorer(annotated, cfg))
        annotated.update(
            {
                "legacy_score": float(legacy_score),
                "score": float(score),
                "v10_17_score_version": V10_17_SCORE_VERSION,
                "crisis_abs_bias_sbp": abs(_row_float(annotated, "crisis_bias_sbp")),
                "crisis_abs_bias_dbp": abs(_row_float(annotated, "crisis_bias_dbp")),
                "high_abs_bias_sbp": abs(_row_float(annotated, "high_bias_sbp")),
                "high_abs_bias_dbp": abs(_row_float(annotated, "high_bias_dbp")),
            }
        )
        if selected_name is not None:
            annotated["v10_17_selected"] = int(str(annotated.get("candidate", "")) == selected_name)
        annotated_rows.append(annotated)
    annotated_rows.sort(key=lambda item: float(item.get("score", 0.0)))
    return annotated_rows


def _select_regression_row(rows: List[dict], cfg, prefer_crisis: bool = False) -> dict:
    if not rows:
        raise RuntimeError("No regression candidate rows were available.")
    mae_target = _reg_target(cfg, "REGRESSION_PARETO_MAE_TARGET", 5.70)
    cov_target = _reg_target(cfg, "REGRESSION_PARETO_COVERAGE_GAP_MAX", 0.16)
    crisis_sbp_target = _reg_target(cfg, "CRISIS_BIAS_TARGET_ABS_SBP", 5.0)
    crisis_dbp_target = _reg_target(cfg, "CRISIS_BIAS_TARGET_ABS_DBP", 4.0)
    feasible = [
        row
        for row in rows
        if _row_float(row, "mae_mean") <= mae_target + (0.10 if prefer_crisis else 0.0)
        and _row_float(row, "coverage_gap") <= cov_target
        and abs(_row_float(row, "crisis_bias_sbp")) <= crisis_sbp_target
        and abs(_row_float(row, "crisis_bias_dbp")) <= crisis_dbp_target + 1.0
    ]
    if feasible:
        return feasible[0]
    return rows[0]


def _merge_top_rows_keep_identity(existing: List[dict], incoming: Iterable[dict], keep: int) -> List[dict]:
    rows = _ORIG_V16_MERGE_TOP_ROWS(existing, incoming, keep)
    identity = next(
        (
            dict(row)
            for row in list(existing) + list(incoming)
            if str(row.get("candidate", "")) == "identity"
        ),
        None,
    )
    if identity is not None and not any(str(row.get("candidate", "")) == "identity" for row in rows):
        rows = rows[:-1] + [identity] if len(rows) >= keep else rows + [identity]
        rows.sort(key=lambda item: float(item.get("score", 0.0)))
    return rows[:keep]


def _high_bias_signature_v17(cfg) -> dict:
    signature = dict(_ORIG_V16_HIGH_BIAS_SIGNATURE(cfg))
    signature.update(
        {
            "protocol_id": FINAL_PROTOCOL_ID,
            "score_version": V10_17_SCORE_VERSION,
            "crisis_abs_sbp_target": _reg_target(cfg, "CRISIS_BIAS_TARGET_ABS_SBP", 5.0),
            "crisis_abs_dbp_target": _reg_target(cfg, "CRISIS_BIAS_TARGET_ABS_DBP", 4.0),
            "mae_target": _reg_target(cfg, "REGRESSION_PARETO_MAE_TARGET", 5.70),
        }
    )
    return signature


def _crisis_signature_v17(cfg) -> dict:
    signature = dict(_ORIG_V16_CRISIS_SIGNATURE(cfg))
    signature.update(
        {
            "protocol_id": FINAL_PROTOCOL_ID,
            "score_version": V10_17_SCORE_VERSION,
            "crisis_abs_sbp_target": _reg_target(cfg, "CRISIS_BIAS_TARGET_ABS_SBP", 5.0),
            "crisis_abs_dbp_target": _reg_target(cfg, "CRISIS_BIAS_TARGET_ABS_DBP", 4.0),
            "mae_target": _reg_target(cfg, "REGRESSION_PARETO_MAE_TARGET", 5.70),
        }
    )
    return signature


def _vectorized_high_bias_rows_pareto(batch_rows: List[dict], calib_pre: dict, query_pre: dict, base_ref: dict, cfg) -> List[dict]:
    rows = _ORIG_V16_VECTOR_HIGH_BIAS_ROWS(batch_rows, calib_pre, query_pre, base_ref, cfg)
    return _annotate_regression_rows(rows, cfg, _high_bias_row_score)


def _vectorized_crisis_tail_rows_pareto(batch_rows: List[dict], calib_pre: dict, query_pre: dict, base_ref: dict, cfg) -> List[dict]:
    rows = _ORIG_V16_VECTOR_CRISIS_TAIL_ROWS(batch_rows, calib_pre, query_pre, base_ref, cfg)
    return _annotate_regression_rows(rows, cfg, _crisis_tail_row_score)


def tailaware_search_high_bias_calibration_candidates_pareto(
    calib_out: dict,
    calib_cls_prob: np.ndarray,
    query_out: dict,
    query_cls_prob: np.ndarray,
    cfg,
) -> tuple[dict, List[dict]]:
    best, rows = _ORIG_V16_SEARCH_HIGH_BIAS(calib_out, calib_cls_prob, query_out, query_cls_prob, cfg)
    rows = _annotate_regression_rows(rows, cfg, _high_bias_row_score)
    selected = _select_regression_row(rows, cfg, prefer_crisis=False)
    selected_name = str(selected.get("candidate", ""))
    rows = _annotate_regression_rows(rows, cfg, _high_bias_row_score, selected_name=selected_name)
    selected = dict(next((row for row in rows if str(row.get("candidate", "")) == selected_name), selected))
    print(
        "[v10.17] High-bias calibration selected "
        f"{selected_name}: mae={_row_float(selected, 'mae_mean'):.4f}, "
        f"high_sbp={_row_float(selected, 'high_bias_sbp'):.4f}, "
        f"crisis_sbp={_row_float(selected, 'crisis_bias_sbp'):.4f}",
        flush=True,
    )
    return selected, rows


def search_crisis_tail_debias_candidates_pareto(
    calib_out: dict,
    calib_cls_prob: np.ndarray,
    calib_reg_inputs: dict,
    query_out: dict,
    query_cls_prob: np.ndarray,
    query_reg_inputs: dict,
    cfg,
) -> tuple[dict, List[dict]]:
    best, rows = _ORIG_V16_SEARCH_CRISIS_TAIL(
        calib_out,
        calib_cls_prob,
        calib_reg_inputs,
        query_out,
        query_cls_prob,
        query_reg_inputs,
        cfg,
    )
    rows = _annotate_regression_rows(rows, cfg, _crisis_tail_row_score)
    selected = _select_regression_row(rows, cfg, prefer_crisis=True)
    selected_name = str(selected.get("candidate", ""))
    rows = _annotate_regression_rows(rows, cfg, _crisis_tail_row_score, selected_name=selected_name)
    selected = dict(next((row for row in rows if str(row.get("candidate", "")) == selected_name), selected))
    print(
        "[v10.17] Crisis-tail fusion selected "
        f"{selected_name}: mae={_row_float(selected, 'mae_mean'):.4f}, "
        f"crisis_sbp={_row_float(selected, 'crisis_bias_sbp'):.4f}, "
        f"crisis_dbp={_row_float(selected, 'crisis_bias_dbp'):.4f}",
        flush=True,
    )
    return selected, rows


def search_classification_arbiter_pareto(
    arbiter_bundle: dict,
    calib_selected_prob: np.ndarray,
    calib_stability_prob: np.ndarray,
    query_selected_prob_map: Dict[str, np.ndarray],
    query_stability_prob_map: Dict[str, np.ndarray],
    y_map: Dict[str, np.ndarray],
    cfg,
) -> tuple[dict, List[dict]]:
    best, rows = _ORIG_V16_SEARCH_CLASSIFICATION_ARBITER(
        arbiter_bundle,
        calib_selected_prob,
        calib_stability_prob,
        query_selected_prob_map,
        query_stability_prob_map,
        y_map,
        cfg,
    )
    rows = _annotate_classification_rows(rows, cfg)
    baseline_pass = [row for row in rows if bool(row.get("classification_gate_pass", False))]
    selected = dict(baseline_pass[0] if baseline_pass else rows[0])
    selected_name = str(selected.get("candidate", ""))
    decision = "baseline_gate_pass" if baseline_pass else "best_available_no_gate_pass"
    selected["classification_gate_decision"] = decision
    selected["classification_gate_selected"] = 1
    rows = _annotate_classification_rows(rows, cfg, selected_name=selected_name)
    for row in rows:
        row["classification_gate_decision"] = decision if str(row.get("candidate", "")) == selected_name else "searched"
    selected = dict(next((row for row in rows if str(row.get("candidate", "")) == selected_name), selected))
    selected["classification_gate_decision"] = decision
    selected["classification_gate_selected"] = 1
    print(
        "[v10.17] Classification arbiter selected "
        f"{selected_name}: acc={_row_float(selected, 'acc'):.6f}, "
        f"macro_f1={_row_float(selected, 'macro_f1'):.6f}, "
        f"gate={decision}",
        flush=True,
    )
    return selected, rows


def guarded_safety_class_fusion_score_pareto(metrics: dict, prefix: str, cfg) -> float:
    try:
        legacy = float(_ORIG_V16_SAFETY_SCORE(metrics, prefix, cfg))
    except Exception:
        legacy = 0.0
    summary = _summary_from_metrics(metrics, prefix, cfg)
    return float(0.35 * legacy + _classification_pareto_score(summary, cfg))


def guarded_search_safety_class_fusion_pareto(query_cls_prob: np.ndarray, query_reg_out: dict, cfg) -> tuple[dict, List[dict]]:
    best, rows = _ORIG_V16_SEARCH_SAFETY(query_cls_prob, query_reg_out, cfg)
    annotated = _annotate_classification_rows(rows, cfg)
    identity = next((row for row in annotated if str(row.get("candidate", "")) == "identity"), None)
    baseline_pass = [row for row in annotated if bool(row.get("classification_gate_pass", False))]
    if baseline_pass:
        selected = dict(baseline_pass[0])
        decision = "baseline_gate_pass"
    elif identity is not None:
        selected = dict(identity)
        decision = "identity_fallback_no_gate_pass"
    else:
        selected = dict(annotated[0])
        decision = "best_available_no_identity"
    selected_name = str(selected.get("candidate", ""))
    for row in annotated:
        is_selected = str(row.get("candidate", "")) == selected_name
        row["safety_gate_selected"] = int(is_selected)
        row["safety_gate_decision"] = decision if is_selected else "searched"
    selected = dict(next((row for row in annotated if str(row.get("candidate", "")) == selected_name), selected))
    selected["safety_gate_selected"] = 1
    selected["safety_gate_decision"] = decision
    annotated.sort(key=lambda row: (int(row.get("safety_gate_selected", 0)), float(row.get("score", 0.0))), reverse=True)
    print(
        "[v10.17] Safety class fusion selected "
        f"{selected_name}: acc={_row_float(selected, 'acc', _row_float(selected, 'cls_acc_selected_val')):.6f}, "
        f"macro_f1={_row_float(selected, 'macro_f1', _row_float(selected, 'cls_f1_macro_selected_val')):.6f}, "
        f"gate={decision}",
        flush=True,
    )
    return selected, annotated


def _unique(existing: Sequence[float], extra: Sequence[float]) -> tuple[float, ...]:
    return tuple(dict.fromkeys(tuple(float(x) for x in existing) + tuple(float(x) for x in extra)))


def build_nextgen_cfg():
    global _ACTIVE_CFG
    cfg = _ORIG_V16_BUILD_NEXTGEN_CFG()
    cfg.OUTPUT_NAME = FINAL_OUTPUT_NAME
    cfg.PROTOCOL_ID = FINAL_PROTOCOL_ID
    cfg.PROTOCOL_NAME = (
        "v10.17 subject-disjoint Pareto-ordinal crisis-calibrated protocol "
        "(reuse v10.13/v10.16 backbones + hard v10.11 gates + exact vectorized tail searches)"
    )
    cfg.FULLTRAIN_OPTLONG_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
    cfg.FULLTRAIN_DUALMAX_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT
    cfg.WARMSTART_CANDIDATES = tuple(
        dict.fromkeys(
            tuple(getattr(cfg, "WARMSTART_CANDIDATES", ()))
            + (
                "mimic_bp_reg_v10_16_subjectdisjoint_clinicalordinal_biasbalanced_proto",
                "mimic_bp_reg_v10_13_subjectdisjoint_piso_uncertainty_moe_tailaware_proto",
                "mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_proto",
            )
        )
    )

    cfg.BASELINE_AWARE_SELECTOR_ENABLE = True
    cfg.BASELINE_MIN_ACC_MARGIN = max(float(getattr(cfg, "BASELINE_MIN_ACC_MARGIN", 0.0)), 0.0020)
    cfg.BASELINE_MIN_F1_MARGIN = max(float(getattr(cfg, "BASELINE_MIN_F1_MARGIN", 0.0)), 0.0020)
    cfg.BASELINE_MIN_BAL_MARGIN = max(float(getattr(cfg, "BASELINE_MIN_BAL_MARGIN", 0.0)), 0.0015)
    cfg.BASELINE_ACC_SHORTFALL_WEIGHT = max(float(getattr(cfg, "BASELINE_ACC_SHORTFALL_WEIGHT", 30.0)), 140.0)
    cfg.BASELINE_F1_SHORTFALL_WEIGHT = max(float(getattr(cfg, "BASELINE_F1_SHORTFALL_WEIGHT", 34.0)), 170.0)
    cfg.BASELINE_BAL_SHORTFALL_WEIGHT = max(float(getattr(cfg, "BASELINE_BAL_SHORTFALL_WEIGHT", 0.0)), 90.0)
    cfg.BASELINE_ASPIRATIONAL_ACC = max(float(getattr(cfg, "BASELINE_ASPIRATIONAL_ACC", 0.86)), 0.86)
    cfg.BASELINE_ASPIRATIONAL_F1 = max(float(getattr(cfg, "BASELINE_ASPIRATIONAL_F1", 0.80)), 0.80)
    cfg.BASELINE_ASPIRATIONAL_ACC_WEIGHT = max(float(getattr(cfg, "BASELINE_ASPIRATIONAL_ACC_WEIGHT", 3.4)), 5.5)
    cfg.BASELINE_ASPIRATIONAL_F1_WEIGHT = max(float(getattr(cfg, "BASELINE_ASPIRATIONAL_F1_WEIGHT", 3.8)), 7.0)
    cfg.CLASSIFIER_RARE_MIN_TARGET = max(float(getattr(cfg, "CLASSIFIER_RARE_MIN_TARGET", 0.0)), 0.57)
    cfg.CLASSIFIER_ROBUST_MIN_TARGET = max(float(getattr(cfg, "CLASSIFIER_ROBUST_MIN_TARGET", 0.0)), 0.59)

    cfg.HEAD_EPOCHS = max(int(getattr(cfg, "HEAD_EPOCHS", 120)), 220)
    cfg.HEAD_PATIENCE = max(int(getattr(cfg, "HEAD_PATIENCE", 40)), 80)
    cfg.HEAD_MIN_EPOCHS = max(int(getattr(cfg, "HEAD_MIN_EPOCHS", 60)), 88)
    cfg.HEAD_LR = min(max(float(getattr(cfg, "HEAD_LR", 3.0e-5)), 2.0e-5), 3.2e-5)
    cfg.HEAD_FOCAL_GAMMA = max(float(getattr(cfg, "HEAD_FOCAL_GAMMA", 1.4)), 1.9)
    cfg.HEAD_LABEL_SMOOTHING = min(float(getattr(cfg, "HEAD_LABEL_SMOOTHING", 0.04)), 0.025)
    cfg.HEAD_KD_WEIGHT = min(float(getattr(cfg, "HEAD_KD_WEIGHT", 0.10)), 0.06)
    cfg.HEAD_ORD_WEIGHT = max(float(getattr(cfg, "HEAD_ORD_WEIGHT", 0.30)), 0.42)
    cfg.HEAD_BP_WEIGHT = max(float(getattr(cfg, "HEAD_BP_WEIGHT", 0.03)), 0.055)
    cfg.HEAD_DROPOUT = min(max(float(getattr(cfg, "HEAD_DROPOUT", 0.14)), 0.12), 0.16)
    cfg.HEAD_ELEVATED_REPEAT = max(int(getattr(cfg, "HEAD_ELEVATED_REPEAT", 2)), 4)
    cfg.HEAD_STAGE1_REPEAT = max(int(getattr(cfg, "HEAD_STAGE1_REPEAT", 3)), 5)
    cfg.HEAD_STAGE2_REPEAT = max(int(getattr(cfg, "HEAD_STAGE2_REPEAT", 4)), 6)
    cfg.HEAD_CLEAN_ACC_WEIGHT = max(float(getattr(cfg, "HEAD_CLEAN_ACC_WEIGHT", 0.0)), 3.0)
    cfg.HEAD_CLEAN_F1_WEIGHT = max(float(getattr(cfg, "HEAD_CLEAN_F1_WEIGHT", 0.0)), 3.4)
    cfg.HEAD_CLEAN_BALANCED_WEIGHT = max(float(getattr(cfg, "HEAD_CLEAN_BALANCED_WEIGHT", 0.0)), 1.35)
    cfg.HEAD_SELECTION_RANK_MODE = "target_gap_first"
    cfg.HEAD_TARGET_RARE_MIN_WEIGHT = max(float(getattr(cfg, "HEAD_TARGET_RARE_MIN_WEIGHT", 0.0)), 0.95)
    cfg.HEAD_TARGET_STAGE2_WEIGHT = max(float(getattr(cfg, "HEAD_TARGET_STAGE2_WEIGHT", 0.0)), 0.70)
    cfg.HEAD_TARGET_GAP_WEIGHT = max(float(getattr(cfg, "HEAD_TARGET_GAP_WEIGHT", 0.0)), 2.7)
    cfg.HEAD_TARGET_ROBUST_GAP_WEIGHT = max(float(getattr(cfg, "HEAD_TARGET_ROBUST_GAP_WEIGHT", 0.0)), 1.2)

    cfg.GUIDED_POLICY_GAMMAS = (0.78, 0.84, 0.90, 1.00, 1.10, 1.22, 1.36, 1.52)
    cfg.GUIDED_POLICY_ELEVATED_WEIGHTS = (0.88, 0.96, 1.00, 1.08, 1.18, 1.32, 1.50, 1.72)
    cfg.GUIDED_POLICY_STAGE1_WEIGHTS = (0.88, 0.98, 1.00, 1.12, 1.28, 1.48, 1.72)
    cfg.GUIDED_POLICY_STAGE2_WEIGHTS = (0.86, 0.96, 1.00, 1.14, 1.34, 1.58, 1.88, 2.20)
    cfg.GUIDED_POLICY_BATCH_SIZE = 4096

    cfg.CLS_ARBITER_SCALES = _unique(getattr(cfg, "CLS_ARBITER_SCALES", ()), (0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00))
    cfg.CLS_ARBITER_BETAS = _unique(getattr(cfg, "CLS_ARBITER_BETAS", ()), (0.80, 1.00, 1.40, 1.80, 2.20, 2.80))
    cfg.CLS_ARBITER_FLOORS = _unique(getattr(cfg, "CLS_ARBITER_FLOORS", ()), (0.00, 0.03, 0.06, 0.10))
    cfg.CLS_ARBITER_AGREE_SHRINKS = _unique(getattr(cfg, "CLS_ARBITER_AGREE_SHRINKS", ()), (0.00, 0.05, 0.12, 0.20, 0.30))
    cfg.CLS_ARBITER_PROGRESS_EVERY = 120

    cfg.SAFETY_CLASS_FUSION_MAX_WEIGHT = min(max(float(getattr(cfg, "SAFETY_CLASS_FUSION_MAX_WEIGHT", 0.42)), 0.50), 0.56)
    cfg.SAFETY_EVIDENTIAL_SCALES = (0.0, 0.04, 0.08, 0.12, 0.18, 0.26, 0.36, 0.48)
    cfg.SAFETY_EVIDENTIAL_BETAS = (0.50, 0.70, 0.90, 1.10, 1.40)
    cfg.SAFETY_EVIDENTIAL_DISAGREE_GAINS = (0.0, 0.35, 0.70, 1.05)
    cfg.SAFETY_EVIDENTIAL_HIGH_GAINS = (0.0, 0.25, 0.50, 0.80, 1.15)
    cfg.SAFETY_EVIDENTIAL_CRISIS_GAINS = (0.0, 0.35, 0.70, 1.10, 1.55)
    cfg.SAFETY_EVIDENTIAL_RELIABILITY_FLOORS = (0.0, 0.04, 0.08, 0.14)
    cfg.SAFETY_EVIDENTIAL_STAGE1_BIASES = (0.0, 0.02, 0.04, 0.07)
    cfg.SAFETY_EVIDENTIAL_STAGE2_BIASES = (0.0, 0.04, 0.08, 0.12, 0.18)
    cfg.SAFETY_CLASS_FUSION_MIN_ACC_GAIN = 0.0
    cfg.SAFETY_CLASS_FUSION_MIN_F1_GAIN = 0.0
    cfg.SAFETY_CLASS_FUSION_MAX_ACC_DROP = 0.0
    cfg.SAFETY_CLASS_FUSION_MAX_F1_DROP = 0.0

    cfg.REGRESSION_PARETO_MAE_TARGET = 5.70
    cfg.REGRESSION_PARETO_COVERAGE_GAP_MAX = 0.16
    cfg.CRISIS_BIAS_TARGET_ABS_SBP = 5.0
    cfg.CRISIS_BIAS_TARGET_ABS_DBP = 4.0
    cfg.HIGH_BIAS_TARGET_ABS_SBP = 4.0
    cfg.HIGH_BIAS_TARGET_ABS_DBP = 3.0
    cfg.BASELINE_CRISIS_SBP_TARGET = -4.5
    cfg.BASELINE_CRISIS_DBP_TARGET = -2.5
    cfg.BASELINE_HIGH_SBP_TARGET = -2.5
    cfg.BASELINE_HIGH_DBP_TARGET = -1.5
    cfg.BASELINE_CRISIS_TARGET_WEIGHT = max(float(getattr(cfg, "BASELINE_CRISIS_TARGET_WEIGHT", 4.60)), 6.00)
    cfg.BASELINE_HIGH_TARGET_WEIGHT = max(float(getattr(cfg, "BASELINE_HIGH_TARGET_WEIGHT", 1.25)), 1.80)
    cfg.BASELINE_BIAS_WORSE_TOL_SBP = min(float(getattr(cfg, "BASELINE_BIAS_WORSE_TOL_SBP", 0.65)), 0.50)
    cfg.BASELINE_BIAS_WORSE_TOL_DBP = min(float(getattr(cfg, "BASELINE_BIAS_WORSE_TOL_DBP", 0.45)), 0.35)
    cfg.RELIABILITY_BIAS_MAX_MAE_DELTA = min(float(getattr(cfg, "RELIABILITY_BIAS_MAX_MAE_DELTA", 0.08)), 0.08)
    cfg.RELIABILITY_BIAS_MAX_COVERAGE_GAP_DELTA = min(float(getattr(cfg, "RELIABILITY_BIAS_MAX_COVERAGE_GAP_DELTA", 0.025)), 0.025)
    cfg.CRISIS_TAIL_FUSION_MAX_MAE_DELTA = min(float(getattr(cfg, "CRISIS_TAIL_FUSION_MAX_MAE_DELTA", 0.10)), 0.10)
    cfg.CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA = min(float(getattr(cfg, "CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA", 0.03)), 0.03)

    cfg.RELIABILITY_BIAS_SCALES = (0.0, 0.08, 0.16, 0.25, 0.38, 0.55)
    cfg.RELIABILITY_BIAS_BETAS = (0.80, 1.00, 1.25, 1.55)
    cfg.RELIABILITY_BIAS_RELIABILITY_FLOORS = (0.0, 0.05, 0.10)
    cfg.RELIABILITY_BIAS_DISAGREE_GAINS = (0.0, 0.35, 0.75)
    cfg.RELIABILITY_BIAS_HIGH_GAINS = (0.0, 0.25, 0.55, 0.90)
    cfg.RELIABILITY_BIAS_CRISIS_GAINS = (0.0, 0.35, 0.70, 1.10, 1.65)
    cfg.RELIABILITY_BIAS_NEGATIVE_FRACS = (0.0, 0.06, 0.12)
    cfg.RELIABILITY_BIAS_HIGH_THRESHOLDS = (0.08, 0.14, 0.22, 0.32)
    cfg.RELIABILITY_BIAS_CRISIS_THRESHOLDS = (0.04, 0.10, 0.18, 0.30)
    cfg.RELIABILITY_BIAS_HIGH_FLOOR_SBP = (0.0, 2.0, 4.0)
    cfg.RELIABILITY_BIAS_CRISIS_FLOOR_SBP = (0.0, 3.0, 6.0, 9.0, 12.0)

    cfg.CRISIS_TAIL_FUSION_HIGH_THRESHOLDS = (0.08, 0.14, 0.22)
    cfg.CRISIS_TAIL_FUSION_CRISIS_THRESHOLDS = (0.04, 0.10, 0.18, 0.30)
    cfg.CRISIS_TAIL_FUSION_GAMMAS = (0.85, 1.00, 1.25)
    cfg.CRISIS_TAIL_FUSION_SBP_QUANTILES = (0.82, 0.88, 0.92, 0.96)
    cfg.CRISIS_TAIL_FUSION_DBP_QUANTILES = (0.76, 0.84, 0.92)
    cfg.CRISIS_TAIL_FUSION_CRISIS_GAINS = (0.55, 0.85, 1.20, 1.70, 2.35)
    cfg.CRISIS_TAIL_FUSION_SBP_MARGINS = (-2.0, 0.0, 2.0, 4.0, 6.0, 9.0)
    cfg.CRISIS_TAIL_FUSION_DBP_MARGINS = (-1.5, 0.0, 1.5, 3.0)
    cfg.CRISIS_TAIL_FUSION_UNCERTAINTY_GAINS = (0.0, 0.20, 0.45)
    cfg.CRISIS_TAIL_FUSION_MODEL_SCALES = (0.65, 0.95, 1.25)
    cfg.CRISIS_TAIL_FUSION_EXPERT_GAINS = (0.45, 0.85, 1.30)
    cfg.CRISIS_TAIL_UNDEREST_WEIGHT_SBP = max(float(getattr(cfg, "CRISIS_TAIL_UNDEREST_WEIGHT_SBP", 6.5)), 8.0)
    cfg.CRISIS_TAIL_UNDEREST_WEIGHT_DBP = max(float(getattr(cfg, "CRISIS_TAIL_UNDEREST_WEIGHT_DBP", 1.2)), 1.6)

    cfg.VECTORFAST_HIGH_BIAS_KEEP_ROWS = 8192
    cfg.VECTORFAST_CRISIS_KEEP_ROWS = 8192
    cfg.VECTORFAST_CACHE_FLUSH_EVERY = 12
    cfg.VECTORFAST_EXACT_EXHAUSTIVE = True
    cfg.V10_17_SCORE_VERSION = V10_17_SCORE_VERSION

    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    _tables_dir()
    _figures_dir()
    _artifacts_dir()
    _ACTIVE_CFG = cfg
    return cfg


def generate_v1017_audit(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    summary = _read_json(output_dir / "protocol_summary.json")
    bp_rows = _read_csv_rows(output_dir / "tables" / "bp_range_metrics.csv")
    bp_map = {str(row.get("bp_range", "")): row for row in bp_rows}
    cfg = build_nextgen_cfg()
    baseline = _baseline_targets()
    selected_acc = _to_float(summary.get("selected_acc"))
    selected_f1 = _to_float(summary.get("selected_macro_f1"))
    selected_bal = _to_float(summary.get("selected_balanced_acc"))
    crisis = bp_map.get("crisis", {})
    high = bp_map.get("high", {})
    audit = {
        "protocol_id": FINAL_PROTOCOL_ID,
        "score_version": V10_17_SCORE_VERSION,
        "baseline_targets": baseline,
        "aspirational_targets": {
            "accuracy": float(getattr(cfg, "BASELINE_ASPIRATIONAL_ACC", 0.86)),
            "macro_f1": float(getattr(cfg, "BASELINE_ASPIRATIONAL_F1", 0.80)),
        },
        "selected_metrics": {
            "acc": selected_acc,
            "macro_f1": selected_f1,
            "balanced_acc": selected_bal,
            "mae_mean": _to_float(summary.get("selected_mae_mean")),
            "beats_v10_11_acc": bool(selected_acc >= _to_float(baseline.get("selected_acc"))),
            "beats_v10_11_macro_f1": bool(selected_f1 >= _to_float(baseline.get("selected_macro_f1"))),
            "beats_v10_11_balanced_acc": bool(selected_bal >= _to_float(baseline.get("selected_balanced_acc"))),
        },
        "tail_bias_targets": {
            "crisis_abs_sbp_max": float(getattr(cfg, "CRISIS_BIAS_TARGET_ABS_SBP", 5.0)),
            "crisis_abs_dbp_max": float(getattr(cfg, "CRISIS_BIAS_TARGET_ABS_DBP", 4.0)),
            "high_abs_sbp_max": float(getattr(cfg, "HIGH_BIAS_TARGET_ABS_SBP", 4.0)),
            "high_abs_dbp_max": float(getattr(cfg, "HIGH_BIAS_TARGET_ABS_DBP", 3.0)),
        },
        "bp_range_bias": {
            "high": {
                "n": _to_int(high.get("n")),
                "bias_sbp": _row_float(high, "bias_sbp"),
                "bias_dbp": _row_float(high, "bias_dbp"),
                "mae_mean": _row_float(high, "mae_mean"),
            },
            "crisis": {
                "n": _to_int(crisis.get("n")),
                "bias_sbp": _row_float(crisis, "bias_sbp"),
                "bias_dbp": _row_float(crisis, "bias_dbp"),
                "mae_mean": _row_float(crisis, "mae_mean"),
                "crisis_sbp_abs_le_target": bool(
                    abs(_row_float(crisis, "bias_sbp")) <= float(getattr(cfg, "CRISIS_BIAS_TARGET_ABS_SBP", 5.0))
                ),
                "crisis_dbp_abs_le_target": bool(
                    abs(_row_float(crisis, "bias_dbp")) <= float(getattr(cfg, "CRISIS_BIAS_TARGET_ABS_DBP", 4.0))
                ),
            },
        },
        "search_strategy": {
            "classification_arbiter_candidates": int(v16._classification_arbiter_total(cfg)),
            "high_bias_candidates": int(v16._high_bias_total(cfg)),
            "crisis_tail_candidates": int(v16._crisis_total(cfg)),
            "mode": "exact exhaustive traversal with vectorized batch scoring, resume caches, and v10.17 Pareto re-ranking",
            "cache_files": {
                "high_bias": str(v16._search_cache_path("v10_16_high_bias")),
                "crisis_tail": str(v16._search_cache_path("v10_16_crisis_tail")),
            },
        },
        "paper_readiness_gate": bool(
            selected_acc >= _to_float(baseline.get("selected_acc"))
            and selected_f1 >= _to_float(baseline.get("selected_macro_f1"))
            and abs(_row_float(crisis, "bias_sbp")) <= float(getattr(cfg, "CRISIS_BIAS_TARGET_ABS_SBP", 5.0))
            and abs(_row_float(crisis, "bias_dbp")) <= max(5.0, float(getattr(cfg, "CRISIS_BIAS_TARGET_ABS_DBP", 4.0)))
        ),
        "notes": [
            "The script reuses the same trained backbone/full-train artifacts as v10.16/v10.13 and keeps the output schema aligned with the existing protocol.",
            "Classification selection is baseline-gated against v10.11; if no candidate clears the gate, the audit marks it explicitly.",
            "High-bias and crisis-tail searches still traverse every configured candidate; acceleration is vectorized scoring and checkpointed cache, not pruning.",
            "Crisis-bin sample count is reported because very small bins make tail bias unstable and should be disclosed in the paper tables.",
        ],
    }
    _write_json(output_dir / "v10_17_paretoordinal_crisiscal_audit.json", audit)


def generate_extra_outputs(output_dir: Path) -> None:
    _ORIG_V16_GENERATE_EXTRA_OUTPUTS(output_dir)
    generate_v1017_audit(output_dir)


def main() -> None:
    global _ACTIVE_CFG
    originals = {
        "v16_final_output": v16.FINAL_OUTPUT_NAME,
        "v16_protocol_id": v16.FINAL_PROTOCOL_ID,
        "v16_optlong": v16.OPTLONG_FULLTRAIN_OUTPUT,
        "v16_dualmax": v16.DUALMAX_FULLTRAIN_OUTPUT,
        "v16_build_nextgen": v16.build_nextgen_cfg,
        "v16_generate_extra": v16.generate_extra_outputs,
        "v16_cls_arbiter": v16.search_classification_arbiter_with_progress,
        "v16_vector_high": v16._vectorized_high_bias_rows,
        "v16_vector_crisis": v16._vectorized_crisis_tail_rows,
        "v16_search_high": v16.tailaware_search_high_bias_calibration_candidates,
        "v16_search_crisis": v16.search_crisis_tail_debias_candidates,
        "v16_safety_score": v16.guarded_safety_class_fusion_score,
        "v16_safety_search": v16.guarded_search_safety_class_fusion,
        "v16_merge_top": v16._merge_top_rows,
        "v16_high_sig": v16._high_bias_signature,
        "v16_crisis_sig": v16._crisis_signature,
        "v13_selection": v13.tailaware_classification_selection_score,
        "v13_candidate": v13.tailaware_classification_candidate_score,
        "v13_robust": v13.tailaware_robust_classification_score,
        "guided_policy": guided_script.search_policy,
        "guided_candidate": guided_script.classification_candidate_score,
        "meta_prev_robust": meta_script.prev_script.robust_classification_score,
        "active_cfg": _ACTIVE_CFG,
    }
    try:
        v16.FINAL_OUTPUT_NAME = FINAL_OUTPUT_NAME
        v16.FINAL_PROTOCOL_ID = FINAL_PROTOCOL_ID
        v16.OPTLONG_FULLTRAIN_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
        v16.DUALMAX_FULLTRAIN_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT
        v16.build_nextgen_cfg = build_nextgen_cfg
        v16.generate_extra_outputs = generate_extra_outputs
        v16.search_classification_arbiter_with_progress = search_classification_arbiter_pareto
        v16._vectorized_high_bias_rows = _vectorized_high_bias_rows_pareto
        v16._vectorized_crisis_tail_rows = _vectorized_crisis_tail_rows_pareto
        v16.tailaware_search_high_bias_calibration_candidates = tailaware_search_high_bias_calibration_candidates_pareto
        v16.search_crisis_tail_debias_candidates = search_crisis_tail_debias_candidates_pareto
        v16.guarded_safety_class_fusion_score = guarded_safety_class_fusion_score_pareto
        v16.guarded_search_safety_class_fusion = guarded_search_safety_class_fusion_pareto
        v16._merge_top_rows = _merge_top_rows_keep_identity
        v16._high_bias_signature = _high_bias_signature_v17
        v16._crisis_signature = _crisis_signature_v17
        v13.tailaware_classification_selection_score = pareto_classification_selection_score
        v13.tailaware_classification_candidate_score = pareto_classification_candidate_score
        v13.tailaware_robust_classification_score = pareto_robust_classification_score
        guided_script.search_policy = configurable_search_policy
        guided_script.classification_candidate_score = guided_classification_candidate_score
        meta_script.prev_script.robust_classification_score = pareto_robust_classification_score
        v16.main()
    finally:
        v16.FINAL_OUTPUT_NAME = originals["v16_final_output"]
        v16.FINAL_PROTOCOL_ID = originals["v16_protocol_id"]
        v16.OPTLONG_FULLTRAIN_OUTPUT = originals["v16_optlong"]
        v16.DUALMAX_FULLTRAIN_OUTPUT = originals["v16_dualmax"]
        v16.build_nextgen_cfg = originals["v16_build_nextgen"]
        v16.generate_extra_outputs = originals["v16_generate_extra"]
        v16.search_classification_arbiter_with_progress = originals["v16_cls_arbiter"]
        v16._vectorized_high_bias_rows = originals["v16_vector_high"]
        v16._vectorized_crisis_tail_rows = originals["v16_vector_crisis"]
        v16.tailaware_search_high_bias_calibration_candidates = originals["v16_search_high"]
        v16.search_crisis_tail_debias_candidates = originals["v16_search_crisis"]
        v16.guarded_safety_class_fusion_score = originals["v16_safety_score"]
        v16.guarded_search_safety_class_fusion = originals["v16_safety_search"]
        v16._merge_top_rows = originals["v16_merge_top"]
        v16._high_bias_signature = originals["v16_high_sig"]
        v16._crisis_signature = originals["v16_crisis_sig"]
        v13.tailaware_classification_selection_score = originals["v13_selection"]
        v13.tailaware_classification_candidate_score = originals["v13_candidate"]
        v13.tailaware_robust_classification_score = originals["v13_robust"]
        guided_script.search_policy = originals["guided_policy"]
        guided_script.classification_candidate_score = originals["guided_candidate"]
        meta_script.prev_script.robust_classification_score = originals["meta_prev_robust"]
        _ACTIVE_CFG = originals["active_cfg"]


if __name__ == "__main__":
    main()
