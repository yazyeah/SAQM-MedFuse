from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

import aqm_bp_shared_v9 as shared_v9
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_dualanchor_meta_stack_protocol as meta_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_dualbackbone_bridge_protocol as bridge_script
import train_aqm_medfuse_mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_protocol as prev_script


FINAL_OUTPUT_NAME = "mimic_bp_reg_v10_12_subjectdisjoint_piso_uncertainty_moe_fulltrain_reliabilitybias_proto"
FINAL_PROTOCOL_ID = "v10.12_subjectdisjoint_piso_uncertainty_moe_fulltrain_reliabilitybias"
OPTLONG_FULLTRAIN_OUTPUT = "mimic_bp_reg_v10_12_opt_long_fulltrain_proto"
DUALMAX_FULLTRAIN_OUTPUT = "mimic_bp_reg_v10_12_optlong_stageaware_dualmax_fulltrain_proto"

INSPIRATION_PAPERS = [
    {
        "paper": "Pattern Recognition 2025 reliability-aware uncertainty-guided fusion",
        "design_hook": "use reliability and disagreement jointly when deciding how strongly to trust auxiliary evidence",
        "url": "https://doi.org/10.1016/j.patcog.2024.110993",
    },
    {
        "paper": "Information Fusion 2025 deep evidential reliability learning",
        "design_hook": "fuse classifier and regression-induced class evidence in log-probability space with reliability gating",
        "url": "https://doi.org/10.1016/j.inffus.2024.102648",
    },
    {
        "paper": "Information Fusion 2026 multi-stage cuffless BP estimation",
        "design_hook": "bias correction and decision fusion should explicitly respect subject-specific high-risk tails",
        "url": "https://doi.org/10.1016/j.inffus.2025.103764",
    },
]

_ORIG_PREV_BUILD_CFG = prev_script.build_nextgen_cfg
_ORIG_PREV_BUILD_OPTLONG_CFG = prev_script.build_optlong_fulltrain_cfg
_ORIG_PREV_BUILD_DUALMAX_CFG = prev_script.build_dualmax_fulltrain_cfg
_ORIG_PREV_TARGET_SCORE = prev_script.targeted_classification_selection_score
_ORIG_PREV_TARGET_CANDIDATE_SCORE = prev_script.targeted_classification_candidate_score
_ORIG_PREV_TARGET_ROBUST_SCORE = prev_script.targeted_robust_classification_score

_BASELINE_OUTPUT_NAME = "mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_proto"


def _tail_bias_penalty(bp_range_rows: List[dict]) -> float:
    fn = getattr(prev_script, "tail_bias_penalty", None)
    if fn is None:
        fn = getattr(getattr(meta_script, "prev_script", None), "tail_bias_penalty", None)
    if fn is not None:
        return float(fn(bp_range_rows))

    row_map = {str(row.get("bp_range", "")): row for row in bp_range_rows}
    penalty = 0.0
    normal = row_map.get("normal")
    if normal is not None:
        penalty += 0.25 * max(0.0, float(normal.get("bias_sbp", 0.0)))
        penalty += 0.12 * max(0.0, float(normal.get("bias_dbp", 0.0)))
    elevated = row_map.get("elevated")
    if elevated is not None:
        penalty += 0.40 * max(0.0, -float(elevated.get("bias_sbp", 0.0)))
        penalty += 0.20 * max(0.0, -float(elevated.get("bias_dbp", 0.0)))
    high = row_map.get("high")
    if high is not None:
        penalty += 0.85 * max(0.0, -float(high.get("bias_sbp", 0.0)))
        penalty += 0.45 * max(0.0, -float(high.get("bias_dbp", 0.0)))
    crisis = row_map.get("crisis")
    if crisis is not None:
        penalty += 1.10 * max(0.0, -float(crisis.get("bias_sbp", 0.0)))
        penalty += 0.65 * max(0.0, -float(crisis.get("bias_dbp", 0.0)))
    return float(penalty)


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _output_dir() -> Path:
    return _project_root() / "outputs" / FINAL_OUTPUT_NAME


def _baseline_output_dir() -> Path:
    return _project_root() / "outputs" / _BASELINE_OUTPUT_NAME


def _tables_dir() -> Path:
    path = _output_dir() / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _figures_dir() -> Path:
    path = _output_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _patch_prev_outputs() -> None:
    prev_script.FINAL_OUTPUT_NAME = FINAL_OUTPUT_NAME
    prev_script.FINAL_PROTOCOL_ID = FINAL_PROTOCOL_ID
    prev_script.OPTLONG_FULLTRAIN_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
    prev_script.DUALMAX_FULLTRAIN_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT


def _ensure_expected_dirs() -> None:
    for name in (FINAL_OUTPUT_NAME, OPTLONG_FULLTRAIN_OUTPUT, DUALMAX_FULLTRAIN_OUTPUT):
        (_project_root() / "outputs" / name).mkdir(parents=True, exist_ok=True)


def _write_run_status(status: str, stage: str, extra: dict | None = None) -> None:
    _ensure_expected_dirs()
    payload = {
        "status": str(status),
        "stage": str(stage),
        "protocol_id": FINAL_PROTOCOL_ID,
        "main_output": str(_output_dir()),
        "fulltrain_optlong_output": str(_project_root() / "outputs" / OPTLONG_FULLTRAIN_OUTPUT),
        "fulltrain_dualmax_output": str(_project_root() / "outputs" / DUALMAX_FULLTRAIN_OUTPUT),
    }
    if extra:
        payload.update(extra)
    with (_output_dir() / "run_status.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _is_incomplete_fulltrain_cache(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return False
    files = list(path.glob("*"))
    if not files:
        return False
    return not (path / "epoch_log.csv").exists()


def _archive_incomplete_fulltrain_cache(path: Path) -> Path | None:
    if not _is_incomplete_fulltrain_cache(path):
        return None
    candidate = path.with_name(path.name + "_stale_incomplete")
    idx = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_stale_incomplete_{idx}")
        idx += 1
    path.rename(candidate)
    return candidate


def _prepare_fulltrain_cache_dirs() -> List[str]:
    archived: List[str] = []
    for name in (OPTLONG_FULLTRAIN_OUTPUT, DUALMAX_FULLTRAIN_OUTPUT):
        path = _project_root() / "outputs" / name
        archived_path = _archive_incomplete_fulltrain_cache(path)
        if archived_path is not None:
            archived.append(str(archived_path))
    return archived


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _baseline_targets() -> dict:
    summary = _load_json(_baseline_output_dir() / "protocol_summary.json")
    return {
        "selected_acc": float(summary.get("selected_acc", 0.8122807017543859)),
        "selected_macro_f1": float(summary.get("selected_macro_f1", 0.7334048956487922)),
        "selected_balanced_acc": float(summary.get("selected_balanced_acc", 0.751738213088013)),
        "stability_acc": float(summary.get("stability_acc", 0.8013157894736842)),
        "stability_macro_f1": float(summary.get("stability_macro_f1", 0.7257862236501365)),
    }


def _load_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_row(rows: List[dict], key: str, value: str) -> dict | None:
    for row in rows:
        if str(row.get(key, "")) == str(value):
            return row
    return None


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    lo = float(np.percentile(x, 5.0))
    hi = float(np.percentile(x, 95.0))
    if hi <= lo + 1.0e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _signal_ramp(signal: np.ndarray, threshold: float, gamma: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    scaled = np.clip((signal - float(threshold)) / max(1.0e-6, 1.0 - float(threshold)), 0.0, 1.0)
    return np.power(scaled, float(gamma), dtype=np.float32)


def _class_proxy(prob: np.ndarray, cfg) -> np.ndarray:
    return shared_v9.class_prob_to_bp_proxy(np.asarray(prob, dtype=np.float32), cfg).astype(np.float32)


def _tag(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _run_self_subprocess(args: List[str]) -> int:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        cwd=str(_project_root()),
        check=False,
    )
    return int(proc.returncode)


def build_optlong_fulltrain_cfg():
    _patch_prev_outputs()
    cfg = _ORIG_PREV_BUILD_OPTLONG_CFG()
    cfg.OUTPUT_NAME = OPTLONG_FULLTRAIN_OUTPUT
    cfg.PROTOCOL_ID = "v10.12_optlong_fulltrain"
    cfg.PROTOCOL_NAME = "v10.12 opt-long warm-start full retraining"
    return cfg


def build_dualmax_fulltrain_cfg(base_builder=None):
    _patch_prev_outputs()
    cfg = _ORIG_PREV_BUILD_DUALMAX_CFG(base_builder)
    cfg.OUTPUT_NAME = DUALMAX_FULLTRAIN_OUTPUT
    cfg.PROTOCOL_ID = "v10.12_optlong_stageaware_dualmax_fulltrain"
    cfg.PROTOCOL_NAME = "v10.12 stage-aware dualmax warm-start full retraining"
    return cfg


def build_nextgen_cfg():
    _patch_prev_outputs()
    cfg = _ORIG_PREV_BUILD_CFG()
    cfg.OUTPUT_NAME = FINAL_OUTPUT_NAME
    cfg.PROTOCOL_ID = FINAL_PROTOCOL_ID
    cfg.PROTOCOL_NAME = (
        "v10.12 subject-disjoint PiSO-inspired uncertainty-MoE full-train reliability-bias protocol "
        "(v10.11 crisis-debias stack + signed reliability-aware bias calibration + evidential safety fusion)"
    )
    cfg.FULLTRAIN_OPTLONG_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
    cfg.FULLTRAIN_DUALMAX_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT
    cfg.WARMSTART_CANDIDATES = tuple(cfg.WARMSTART_CANDIDATES) + (prev_script.FINAL_OUTPUT_NAME,)

    cfg.HEAD_EPOCHS = max(int(cfg.HEAD_EPOCHS), 168)
    cfg.HEAD_PATIENCE = max(int(cfg.HEAD_PATIENCE), 48)
    cfg.HEAD_MIN_EPOCHS = max(int(cfg.HEAD_MIN_EPOCHS), 56)
    cfg.HEAD_LR = min(float(cfg.HEAD_LR), 3.0e-5)
    cfg.HEAD_TARGET_RARE_MIN_WEIGHT = max(float(cfg.HEAD_TARGET_RARE_MIN_WEIGHT), 0.54)
    cfg.HEAD_TARGET_STAGE2_WEIGHT = max(float(cfg.HEAD_TARGET_STAGE2_WEIGHT), 0.28)
    cfg.HEAD_TARGET_GAP_WEIGHT = max(float(cfg.HEAD_TARGET_GAP_WEIGHT), 1.35)
    cfg.HEAD_TARGET_ROBUST_GAP_WEIGHT = max(float(cfg.HEAD_TARGET_ROBUST_GAP_WEIGHT), 0.70)
    cfg.HEAD_ROBUST_NOISE_WEIGHT = max(float(cfg.HEAD_ROBUST_NOISE_WEIGHT), 0.14)
    cfg.HEAD_ROBUST_ECG_WEIGHT = max(float(cfg.HEAD_ROBUST_ECG_WEIGHT), 0.18)
    cfg.HEAD_ROBUST_PPG_WEIGHT = max(float(cfg.HEAD_ROBUST_PPG_WEIGHT), 0.25)
    cfg.HEAD_ROBUST_MIN_WEIGHT = max(float(cfg.HEAD_ROBUST_MIN_WEIGHT), 0.24)
    cfg.HEAD_ELEVATED_REPEAT = max(int(getattr(cfg, "HEAD_ELEVATED_REPEAT", 1)), 2)
    cfg.HEAD_STAGE1_REPEAT = max(int(getattr(cfg, "HEAD_STAGE1_REPEAT", 1)), 3)
    cfg.HEAD_STAGE2_REPEAT = max(int(getattr(cfg, "HEAD_STAGE2_REPEAT", 1)), 4)

    cfg.RELIABILITY_BIAS_SCALES = (0.35, 0.55, 0.75)
    cfg.RELIABILITY_BIAS_BETAS = (0.75, 1.00)
    cfg.RELIABILITY_BIAS_RELIABILITY_FLOORS = (0.10, 0.25)
    cfg.RELIABILITY_BIAS_DISAGREE_GAINS = (0.00, 0.45, 0.90)
    cfg.RELIABILITY_BIAS_HIGH_GAINS = (0.60, 1.10)
    cfg.RELIABILITY_BIAS_CRISIS_GAINS = (1.10, 1.80)
    cfg.RELIABILITY_BIAS_NEGATIVE_FRACS = (0.12, 0.20)
    cfg.RELIABILITY_BIAS_HIGH_THRESHOLDS = (0.18, 0.30)
    cfg.RELIABILITY_BIAS_CRISIS_THRESHOLDS = (0.05, 0.12)
    cfg.RELIABILITY_BIAS_HIGH_FLOOR_SBP = (0.0, 1.5)
    cfg.RELIABILITY_BIAS_CRISIS_FLOOR_SBP = (2.0, 5.0)
    cfg.RELIABILITY_BIAS_MAX_MAE_DELTA = 0.22
    cfg.RELIABILITY_BIAS_MAX_COVERAGE_GAP_DELTA = 0.05

    cfg.SAFETY_EVIDENTIAL_SCALES = (0.12, 0.24, 0.40, 0.58)
    cfg.SAFETY_EVIDENTIAL_BETAS = (0.75, 1.00)
    cfg.SAFETY_EVIDENTIAL_DISAGREE_GAINS = (0.20, 0.70)
    cfg.SAFETY_EVIDENTIAL_HIGH_GAINS = (0.35, 0.85)
    cfg.SAFETY_EVIDENTIAL_CRISIS_GAINS = (0.60, 1.20)
    cfg.SAFETY_EVIDENTIAL_RELIABILITY_FLOORS = (0.08, 0.20)
    cfg.SAFETY_EVIDENTIAL_STAGE1_BIASES = (0.00, 0.04)
    cfg.SAFETY_EVIDENTIAL_STAGE2_BIASES = (0.00, 0.08)
    cfg.SAFETY_CLASS_FUSION_MAX_WEIGHT = max(float(cfg.SAFETY_CLASS_FUSION_MAX_WEIGHT), 0.78)
    cfg.SAFETY_CLASS_FUSION_ELEVATED_RECALL_WEIGHT = 0.18
    cfg.SAFETY_CLASS_FUSION_ELEVATED_F1_WEIGHT = 0.10
    cfg.SAFETY_CLASS_FUSION_STAGE1_F1_WEIGHT = 0.08
    cfg.SAFETY_CLASS_FUSION_ECE_WEIGHT = 0.06
    return cfg


def promotion_aware_classification_selection_score(metrics: Dict[str, float], prefix: str) -> float:
    base_score = float(_ORIG_PREV_TARGET_SCORE(metrics, prefix))
    summary = prev_script._target_class_summary(metrics, prefix)
    baseline = _baseline_targets()
    selected_acc = float(baseline["selected_acc"])
    selected_f1 = float(baseline["selected_macro_f1"])
    selected_bal = float(baseline["selected_balanced_acc"])

    acc = float(summary["acc"])
    macro_f1 = float(summary["macro_f1"])
    bal_acc = float(summary["balanced_acc"])
    elevated_f1 = float(summary["elevated_f1"])
    stage1_f1 = float(summary["stage1_f1"])
    stage2_f1 = float(summary["stage2_f1"])
    rare_min = float(summary["rare_f1_min"])

    acc_gain = acc - selected_acc
    f1_gain = macro_f1 - selected_f1
    bal_gain = bal_acc - selected_bal
    acc_shortfall = max(0.0, selected_acc - acc)
    f1_shortfall = max(0.0, selected_f1 - macro_f1)
    bal_shortfall = max(0.0, selected_bal - bal_acc)

    return float(
        base_score
        + 3.2 * acc_gain
        + 4.0 * f1_gain
        + 1.5 * bal_gain
        + 0.45 * elevated_f1
        + 0.55 * stage1_f1
        + 0.70 * stage2_f1
        + 0.35 * rare_min
        - 7.0 * acc_shortfall
        - 8.0 * f1_shortfall
        - 3.0 * bal_shortfall
    )


def promotion_aware_classification_candidate_score(metrics: Dict[str, float], prefix: str) -> float:
    return float(promotion_aware_classification_selection_score(metrics, prefix))


def promotion_aware_robust_classification_score(
    clean_metrics: Dict[str, float],
    noise_metrics: Dict[str, float],
    ecg_metrics: Dict[str, float],
    ppg_metrics: Dict[str, float],
    cfg,
) -> float:
    noise_f1 = float(noise_metrics["cls_f1_macro_selected_noise_val"])
    ecg_f1 = float(ecg_metrics["cls_f1_macro_selected_ecg_val"])
    ppg_f1 = float(ppg_metrics["cls_f1_macro_selected_ppg_val"])
    robust_min = min(noise_f1, ecg_f1, ppg_f1)
    baseline = _baseline_targets()
    clean_score = promotion_aware_classification_candidate_score(clean_metrics, "selected_val")
    clean_acc = float(clean_metrics["cls_acc_selected_val"])
    clean_f1 = float(clean_metrics["cls_f1_macro_selected_val"])
    return float(
        clean_score
        + 0.18 * noise_f1
        + 0.22 * ecg_f1
        + 0.28 * ppg_f1
        + 0.24 * robust_min
        + 2.0 * max(0.0, clean_acc - float(baseline["selected_acc"]))
        + 2.5 * max(0.0, clean_f1 - float(baseline["selected_macro_f1"]))
        - 3.0 * max(0.0, float(baseline["selected_acc"]) - clean_acc)
        - 4.0 * max(0.0, float(baseline["selected_macro_f1"]) - clean_f1)
    )


def apply_high_bias_calibration(
    row: dict,
    reg_out: dict,
    cls_prob: np.ndarray,
    cfg,
):
    candidate = str(row["candidate"])
    if candidate == "identity":
        out = dict(reg_out)
        out["high_bias_cal_shift_mean_sbp"] = 0.0
        out["high_bias_cal_shift_mean_dbp"] = 0.0
        out["high_bias_cal_reliability_mean"] = 0.0
        return out

    pred = np.asarray(reg_out["y_pred_reg"], dtype=np.float32)
    base_prob = bridge_script.normalize_prob(np.asarray(cls_prob, dtype=np.float32))
    reg_prob = bridge_script.normalize_prob(shared_v9.regression_to_class_prob(pred, reg_out.get("uncertainty"), cfg))
    proxy_delta = _class_proxy(base_prob, cfg) - pred
    disagreement = 0.5 * np.abs(reg_prob - base_prob).sum(axis=1)

    uncertainty = np.asarray(reg_out.get("uncertainty", np.zeros(len(pred))), dtype=np.float32).reshape(-1)
    quality = np.asarray(reg_out.get("quality", np.ones(len(pred))), dtype=np.float32).reshape(-1)
    uncertainty_norm = _normalize01(uncertainty)
    quality_norm = _normalize01(quality)
    reliability = float(row["reliability_floor"]) + (1.0 - float(row["reliability_floor"])) * quality_norm * (1.0 - 0.70 * uncertainty_norm)

    high_signal, crisis_signal = meta_script.risk_guard_signals(reg_out, base_prob)
    high_gate = _signal_ramp(high_signal, float(row["high_threshold"]), float(row["beta"]))
    crisis_gate = _signal_ramp(crisis_signal, float(row["crisis_threshold"]), float(row["beta"]))
    disagreement_scale = 1.0 + float(row["disagree_gain"]) * np.power(np.clip(disagreement, 1.0e-6, 1.0), float(row["beta"]))
    risk_scale = 1.0 + float(row["high_gain"]) * high_gate + float(row["crisis_gain"]) * crisis_gate

    delta = float(row["scale"]) * reliability.reshape(-1, 1) * disagreement_scale.reshape(-1, 1) * risk_scale.reshape(-1, 1) * proxy_delta
    max_shift_sbp = float(getattr(cfg, "RISK_GUARD_MAX_SHIFT_SBP", 12.0))
    max_shift_dbp = float(getattr(cfg, "RISK_GUARD_MAX_SHIFT_DBP", 8.0))
    delta[:, 0] = np.clip(delta[:, 0], -float(row["negative_frac"]) * max_shift_sbp, max_shift_sbp)
    delta[:, 1] = np.clip(delta[:, 1], -float(row["negative_frac"]) * max_shift_dbp, max_shift_dbp)

    high_floor_sbp = float(row["high_floor_sbp"]) * high_gate
    crisis_floor_sbp = float(row["crisis_floor_sbp"]) * crisis_gate
    delta[:, 0] = np.maximum(delta[:, 0], high_floor_sbp + crisis_floor_sbp)
    delta[:, 1] = np.maximum(delta[:, 1], 0.35 * high_floor_sbp + 0.45 * crisis_floor_sbp)

    corrected = meta_script.stage_script.clipped_regression_prediction(pred + delta)
    out = meta_script.stage_script.clone_regression_output(reg_out, corrected, cfg)
    out["high_bias_cal_shift_mean_sbp"] = float(delta[:, 0].mean())
    out["high_bias_cal_shift_mean_dbp"] = float(delta[:, 1].mean())
    out["high_bias_cal_reliability_mean"] = float(reliability.mean())
    return out


def high_bias_calibration_cost(calib_out: dict, query_out: dict, base_ref: dict, cfg):
    conformal = meta_script.stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    bp_range_rows = meta_script.stage_script.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"])
    clinical_pen = meta_script.clinical_underestimation_penalty(bp_range_rows)
    tail_pen = _tail_bias_penalty(bp_range_rows)
    range_map = {str(row["bp_range"]): row for row in bp_range_rows}
    high_row = range_map.get("high", {})
    crisis_row = range_map.get("crisis", {})
    reg = query_out["metrics_reg"]

    high_abs_pen = 0.85 * abs(float(high_row.get("bias_sbp", 0.0))) + 0.45 * abs(float(high_row.get("bias_dbp", 0.0)))
    crisis_abs_pen = 0.45 * abs(float(crisis_row.get("bias_sbp", 0.0))) + 0.25 * abs(float(crisis_row.get("bias_dbp", 0.0)))
    crisis_under_pen = 1.40 * max(0.0, -float(crisis_row.get("bias_sbp", 0.0))) + 0.55 * max(0.0, -float(crisis_row.get("bias_dbp", 0.0)))
    mae_excess = max(0.0, float(reg["mae_mean"]) - float(base_ref["mae_mean"]) - float(cfg.RELIABILITY_BIAS_MAX_MAE_DELTA))
    cov_excess = max(
        0.0,
        float(conformal["coverage_gap"]) - float(base_ref["coverage_gap"]) - float(cfg.RELIABILITY_BIAS_MAX_COVERAGE_GAP_DELTA),
    )
    score = float(
        clinical_pen
        + 0.55 * tail_pen
        + high_abs_pen
        + crisis_abs_pen
        + crisis_under_pen
        + 18.0 * mae_excess
        + 8.0 * cov_excess
        + 0.012 * float(reg["mae_mean"])
        + 0.06 * (abs(float(reg["bias_sbp"])) + abs(float(reg["bias_dbp"])))
    )
    return float(score), conformal, bp_range_rows, float(clinical_pen), float(tail_pen)


def search_high_bias_calibration_candidates(
    calib_out: dict,
    calib_cls_prob: np.ndarray,
    query_out: dict,
    query_cls_prob: np.ndarray,
    cfg,
) -> tuple[dict, List[dict]]:
    base_conformal = meta_script.stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    base_bp_range_rows = meta_script.stage_script.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"])
    base_clinical_pen = meta_script.clinical_underestimation_penalty(base_bp_range_rows)
    base_tail_pen = _tail_bias_penalty(base_bp_range_rows)
    base_ref = {
        "mae_mean": float(query_out["metrics_reg"]["mae_mean"]),
        "coverage_gap": float(base_conformal["coverage_gap"]),
    }
    range_map = {str(row["bp_range"]): row for row in base_bp_range_rows}
    rows: List[dict] = [
        {
            "candidate": "identity",
            "scale": 0.0,
            "beta": 1.0,
            "reliability_floor": 0.0,
            "disagree_gain": 0.0,
            "high_gain": 0.0,
            "crisis_gain": 0.0,
            "negative_frac": 0.0,
            "high_threshold": 0.0,
            "crisis_threshold": 0.0,
            "high_floor_sbp": 0.0,
            "crisis_floor_sbp": 0.0,
            "score": float(base_clinical_pen + 0.55 * base_tail_pen + 0.012 * float(query_out["metrics_reg"]["mae_mean"])),
            "clinical_under_penalty": float(base_clinical_pen),
            "tail_bias_penalty": float(base_tail_pen),
            "high_bias_sbp": float(range_map.get("high", {}).get("bias_sbp", 0.0)),
            "high_bias_dbp": float(range_map.get("high", {}).get("bias_dbp", 0.0)),
            "crisis_bias_sbp": float(range_map.get("crisis", {}).get("bias_sbp", 0.0)),
            "crisis_bias_dbp": float(range_map.get("crisis", {}).get("bias_dbp", 0.0)),
            "shift_mean_sbp": 0.0,
            "shift_mean_dbp": 0.0,
            "reliability_mean": 0.0,
            **query_out["metrics_reg"],
            **base_conformal,
        }
    ]

    for scale in tuple(float(x) for x in cfg.RELIABILITY_BIAS_SCALES):
        for beta in tuple(float(x) for x in cfg.RELIABILITY_BIAS_BETAS):
            for reliability_floor in tuple(float(x) for x in cfg.RELIABILITY_BIAS_RELIABILITY_FLOORS):
                for disagree_gain in tuple(float(x) for x in cfg.RELIABILITY_BIAS_DISAGREE_GAINS):
                    for high_gain in tuple(float(x) for x in cfg.RELIABILITY_BIAS_HIGH_GAINS):
                        for crisis_gain in tuple(float(x) for x in cfg.RELIABILITY_BIAS_CRISIS_GAINS):
                            for negative_frac in tuple(float(x) for x in cfg.RELIABILITY_BIAS_NEGATIVE_FRACS):
                                for high_threshold in tuple(float(x) for x in cfg.RELIABILITY_BIAS_HIGH_THRESHOLDS):
                                    for crisis_threshold in tuple(float(x) for x in cfg.RELIABILITY_BIAS_CRISIS_THRESHOLDS):
                                        for high_floor_sbp in tuple(float(x) for x in cfg.RELIABILITY_BIAS_HIGH_FLOOR_SBP):
                                            for crisis_floor_sbp in tuple(float(x) for x in cfg.RELIABILITY_BIAS_CRISIS_FLOOR_SBP):
                                                row = {
                                                    "candidate": (
                                                        f"relbias_s{_tag(scale)}_b{_tag(beta)}_rf{_tag(reliability_floor)}"
                                                        f"_dg{_tag(disagree_gain)}_hg{_tag(high_gain)}_cg{_tag(crisis_gain)}"
                                                        f"_nf{_tag(negative_frac)}_ht{_tag(high_threshold)}_ct{_tag(crisis_threshold)}"
                                                        f"_hf{_tag(high_floor_sbp)}_cf{_tag(crisis_floor_sbp)}"
                                                    ),
                                                    "scale": float(scale),
                                                    "beta": float(beta),
                                                    "reliability_floor": float(reliability_floor),
                                                    "disagree_gain": float(disagree_gain),
                                                    "high_gain": float(high_gain),
                                                    "crisis_gain": float(crisis_gain),
                                                    "negative_frac": float(negative_frac),
                                                    "high_threshold": float(high_threshold),
                                                    "crisis_threshold": float(crisis_threshold),
                                                    "high_floor_sbp": float(high_floor_sbp),
                                                    "crisis_floor_sbp": float(crisis_floor_sbp),
                                                }
                                                calib_adj = apply_high_bias_calibration(row, calib_out, calib_cls_prob, cfg)
                                                query_adj = apply_high_bias_calibration(row, query_out, query_cls_prob, cfg)
                                                score, conformal, bp_range_rows, clinical_pen, tail_pen = high_bias_calibration_cost(
                                                    calib_adj,
                                                    query_adj,
                                                    base_ref,
                                                    cfg,
                                                )
                                                row_map = {str(item["bp_range"]): item for item in bp_range_rows}
                                                rows.append(
                                                    {
                                                        **row,
                                                        "score": float(score),
                                                        "clinical_under_penalty": float(clinical_pen),
                                                        "tail_bias_penalty": float(tail_pen),
                                                        "high_bias_sbp": float(row_map.get("high", {}).get("bias_sbp", 0.0)),
                                                        "high_bias_dbp": float(row_map.get("high", {}).get("bias_dbp", 0.0)),
                                                        "crisis_bias_sbp": float(row_map.get("crisis", {}).get("bias_sbp", 0.0)),
                                                        "crisis_bias_dbp": float(row_map.get("crisis", {}).get("bias_dbp", 0.0)),
                                                        "shift_mean_sbp": float(query_adj.get("high_bias_cal_shift_mean_sbp", 0.0)),
                                                        "shift_mean_dbp": float(query_adj.get("high_bias_cal_shift_mean_dbp", 0.0)),
                                                        "reliability_mean": float(query_adj.get("high_bias_cal_reliability_mean", 0.0)),
                                                        **query_adj["metrics_reg"],
                                                        **conformal,
                                                    }
                                                )

    rows.sort(key=lambda row: float(row["score"]))
    return rows[0], rows


def safety_class_fusion_score(metrics: dict, prefix: str, cfg) -> float:
    elevated_name = str(cfg.CLASS_NAMES[1])
    stage1_name = str(cfg.CLASS_NAMES[2])
    stage2_name = str(cfg.CLASS_NAMES[3])
    return float(
        promotion_aware_classification_selection_score(metrics, prefix)
        + float(getattr(cfg, "SAFETY_CLASS_FUSION_ELEVATED_RECALL_WEIGHT", 0.0))
        * float(metrics.get(f"cls_recall_{prefix}_{elevated_name}", 0.0))
        + float(getattr(cfg, "SAFETY_CLASS_FUSION_ELEVATED_F1_WEIGHT", 0.0))
        * float(metrics.get(f"cls_f1_{prefix}_{elevated_name}", 0.0))
        + float(getattr(cfg, "SAFETY_CLASS_FUSION_STAGE1_RECALL_WEIGHT", 0.0))
        * float(metrics.get(f"cls_recall_{prefix}_{stage1_name}", 0.0))
        + float(getattr(cfg, "SAFETY_CLASS_FUSION_STAGE1_F1_WEIGHT", 0.0))
        * float(metrics.get(f"cls_f1_{prefix}_{stage1_name}", 0.0))
        + float(getattr(cfg, "SAFETY_CLASS_FUSION_STAGE2_RECALL_WEIGHT", 0.0))
        * float(metrics.get(f"cls_recall_{prefix}_{stage2_name}", 0.0))
        + float(getattr(cfg, "SAFETY_CLASS_FUSION_STAGE2_F1_WEIGHT", 0.0))
        * float(metrics.get(f"cls_f1_{prefix}_{stage2_name}", 0.0))
        - float(getattr(cfg, "SAFETY_CLASS_FUSION_ECE_WEIGHT", 0.0))
        * float(metrics.get(f"cls_ece_{prefix}", 0.0))
    )


def apply_safety_class_fusion_prob(
    row: dict,
    base_prob: np.ndarray,
    reg_out: dict,
    cfg,
) -> dict:
    base_prob = bridge_script.normalize_prob(np.asarray(base_prob, dtype=np.float32))
    reg_prob = bridge_script.normalize_prob(
        shared_v9.regression_to_class_prob(np.asarray(reg_out["y_pred_reg"], dtype=np.float32), reg_out.get("uncertainty"), cfg)
    )
    disagreement = 0.5 * np.abs(reg_prob - base_prob).sum(axis=1)
    high_signal, crisis_signal = meta_script.risk_guard_signals(reg_out, base_prob)
    base_pred = base_prob.argmax(axis=1).astype(np.int64)
    reg_pred = reg_prob.argmax(axis=1).astype(np.int64)
    risk_upshift = np.clip((reg_pred - base_pred).astype(np.float32) / max(1, int(cfg.N_CLASSES) - 1), 0.0, 1.0)

    candidate = str(row["candidate"])
    if candidate == "identity" or float(row.get("fusion_scale", 0.0)) <= 0.0:
        weight = np.zeros(len(base_prob), dtype=np.float32)
        fused_prob = base_prob
    else:
        quality = np.asarray(reg_out.get("quality", np.ones(len(base_prob))), dtype=np.float32).reshape(-1)
        uncertainty = np.asarray(reg_out.get("uncertainty", np.zeros(len(base_prob))), dtype=np.float32).reshape(-1)
        reliability = float(row["reliability_floor"]) + (1.0 - float(row["reliability_floor"])) * _normalize01(quality) * (1.0 - 0.70 * _normalize01(uncertainty))
        disagree_flag = (base_pred != reg_pred).astype(np.float32)
        weight = float(row["fusion_scale"]) * np.power(np.clip(disagreement + 0.25 * risk_upshift, 1.0e-6, 1.0), float(row["fusion_beta"]))
        weight *= reliability
        weight *= 1.0 + float(row["fusion_disagree_gain"]) * disagree_flag
        weight *= 1.0 + float(row["fusion_high_gain"]) * high_signal + float(row["fusion_crisis_gain"]) * crisis_signal
        weight = np.clip(weight.astype(np.float32), 0.0, float(getattr(cfg, "SAFETY_CLASS_FUSION_MAX_WEIGHT", 0.78)))

        log_prob = (1.0 - weight.reshape(-1, 1)) * np.log(np.clip(base_prob, 1.0e-6, 1.0))
        log_prob += weight.reshape(-1, 1) * np.log(np.clip(reg_prob, 1.0e-6, 1.0))
        log_prob[:, 2] += float(row["stage1_bias"]) * reliability * (0.35 * high_signal + 0.65 * risk_upshift)
        log_prob[:, 3] += float(row["stage2_bias"]) * reliability * (0.55 * high_signal + crisis_signal + 0.70 * risk_upshift)
        log_prob = log_prob - log_prob.max(axis=1, keepdims=True)
        fused_prob = np.exp(log_prob)
        fused_prob = bridge_script.normalize_prob(fused_prob)

    return {
        "prob": np.asarray(fused_prob, dtype=np.float32),
        "weight": np.asarray(weight, dtype=np.float32),
        "reg_prob": np.asarray(reg_prob, dtype=np.float32),
        "disagreement": np.asarray(disagreement, dtype=np.float32),
        "high_signal": np.asarray(high_signal, dtype=np.float32),
        "crisis_signal": np.asarray(crisis_signal, dtype=np.float32),
        "risk_upshift": np.asarray(risk_upshift, dtype=np.float32),
    }


def search_safety_class_fusion(
    query_cls_prob: np.ndarray,
    query_reg_out: dict,
    cfg,
) -> tuple[dict, List[dict]]:
    rows: List[dict] = []
    y_true = np.asarray(query_reg_out["y_true_cls"], dtype=np.int64)
    elevated_name = str(cfg.CLASS_NAMES[1])
    stage1_name = str(cfg.CLASS_NAMES[2])
    stage2_name = str(cfg.CLASS_NAMES[3])

    identity_diag = apply_safety_class_fusion_prob({"candidate": "identity", "fusion_scale": 0.0}, query_cls_prob, query_reg_out, cfg)
    identity_prob = identity_diag["prob"]
    identity_metrics = meta_script.stage_script.risk_classification_metrics(
        y_true,
        identity_prob.argmax(axis=1).astype(np.int64),
        identity_prob,
        cfg,
        prefix="selected_val",
    )
    rows.append(
        {
            "candidate": "identity",
            "fusion_scale": 0.0,
            "fusion_beta": 0.0,
            "fusion_disagree_gain": 0.0,
            "fusion_high_gain": 0.0,
            "fusion_crisis_gain": 0.0,
            "reliability_floor": 0.0,
            "stage1_bias": 0.0,
            "stage2_bias": 0.0,
            "score": float(safety_class_fusion_score(identity_metrics, "selected_val", cfg)),
            "mean_weight": 0.0,
            "high_weight_mean": 0.0,
            "crisis_weight_mean": 0.0,
            "cls_elevated_recall": float(identity_metrics.get(f"cls_recall_selected_val_{elevated_name}", 0.0)),
            "cls_stage1_recall": float(identity_metrics.get(f"cls_recall_selected_val_{stage1_name}", 0.0)),
            "cls_stage2_recall": float(identity_metrics.get(f"cls_recall_selected_val_{stage2_name}", 0.0)),
            **identity_metrics,
        }
    )

    for scale in tuple(float(x) for x in cfg.SAFETY_EVIDENTIAL_SCALES):
        for beta in tuple(float(x) for x in cfg.SAFETY_EVIDENTIAL_BETAS):
            for disagree_gain in tuple(float(x) for x in cfg.SAFETY_EVIDENTIAL_DISAGREE_GAINS):
                for high_gain in tuple(float(x) for x in cfg.SAFETY_EVIDENTIAL_HIGH_GAINS):
                    for crisis_gain in tuple(float(x) for x in cfg.SAFETY_EVIDENTIAL_CRISIS_GAINS):
                        for reliability_floor in tuple(float(x) for x in cfg.SAFETY_EVIDENTIAL_RELIABILITY_FLOORS):
                            for stage1_bias in tuple(float(x) for x in cfg.SAFETY_EVIDENTIAL_STAGE1_BIASES):
                                for stage2_bias in tuple(float(x) for x in cfg.SAFETY_EVIDENTIAL_STAGE2_BIASES):
                                    row = {
                                        "candidate": (
                                            f"evidfuse_s{_tag(scale)}_b{_tag(beta)}_dg{_tag(disagree_gain)}"
                                            f"_hg{_tag(high_gain)}_cg{_tag(crisis_gain)}_rf{_tag(reliability_floor)}"
                                            f"_s1{_tag(stage1_bias)}_s2{_tag(stage2_bias)}"
                                        ),
                                        "fusion_scale": float(scale),
                                        "fusion_beta": float(beta),
                                        "fusion_disagree_gain": float(disagree_gain),
                                        "fusion_high_gain": float(high_gain),
                                        "fusion_crisis_gain": float(crisis_gain),
                                        "reliability_floor": float(reliability_floor),
                                        "stage1_bias": float(stage1_bias),
                                        "stage2_bias": float(stage2_bias),
                                    }
                                    diag = apply_safety_class_fusion_prob(row, query_cls_prob, query_reg_out, cfg)
                                    prob = diag["prob"]
                                    metrics = meta_script.stage_script.risk_classification_metrics(
                                        y_true,
                                        prob.argmax(axis=1).astype(np.int64),
                                        prob,
                                        cfg,
                                        prefix="selected_val",
                                    )
                                    rows.append(
                                        {
                                            **row,
                                            "score": float(safety_class_fusion_score(metrics, "selected_val", cfg)),
                                            "mean_weight": float(diag["weight"].mean()),
                                            "high_weight_mean": float(diag["weight"][diag["high_signal"] > 0.0].mean()) if np.any(diag["high_signal"] > 0.0) else 0.0,
                                            "crisis_weight_mean": float(diag["weight"][diag["crisis_signal"] > 0.0].mean()) if np.any(diag["crisis_signal"] > 0.0) else 0.0,
                                            "cls_elevated_recall": float(metrics.get(f"cls_recall_selected_val_{elevated_name}", 0.0)),
                                            "cls_stage1_recall": float(metrics.get(f"cls_recall_selected_val_{stage1_name}", 0.0)),
                                            "cls_stage2_recall": float(metrics.get(f"cls_recall_selected_val_{stage2_name}", 0.0)),
                                            **metrics,
                                        }
                                    )

    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows[0], rows


def _plot_bias_stage_trace(rows: List[dict], fig_path: Path) -> None:
    stage_order = ["pre_guard", "post_guard", "post_bias_cal", "post_crisis_fusion", "final_selected"]
    ranges = ["elevated", "high", "crisis"]
    palette = {"elevated": "#2874a6", "high": "#d68910", "crisis": "#c0392b"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for j, metric in enumerate(["sbp_bias", "dbp_bias"]):
        for bp_range in ranges:
            subset = [row for row in rows if str(row["bp_range"]) == bp_range]
            lookup = {str(row["stage"]): float(row[metric]) for row in subset}
            axes[j].plot(stage_order, [lookup.get(stage, np.nan) for stage in stage_order], marker="o", linewidth=2.0, label=bp_range.title(), color=palette[bp_range])
        axes[j].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        axes[j].set_title(f"{metric.split('_')[0].upper()} Bias Stage Trace")
        axes[j].set_ylabel("Bias (Pred - True, mmHg)")
        axes[j].grid(True, linestyle="--", alpha=0.25)
        axes[j].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=True)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_internal_tradeoff(rows: List[dict], fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    x = [float(row["mae_mean"]) for row in rows]
    y = [float(row["macro_f1"]) for row in rows]
    s = [240.0 * float(row["acc"]) for row in rows]
    ax.scatter(x, y, s=s, c=np.linspace(0.2, 0.9, len(rows)), cmap="viridis", alpha=0.9, edgecolors="white", linewidths=0.8)
    for row in rows:
        ax.text(float(row["mae_mean"]) + 0.015, float(row["macro_f1"]) + 0.0015, str(row["label"]), fontsize=9)
    ax.set_xlabel("Regression MAE mean (mmHg)")
    ax.set_ylabel("Classification Macro-F1")
    ax.set_title("Internal Operating Tradeoff")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_uncertainty_bias_profile(rows: List[dict], fig_path: Path) -> None:
    deciles = [int(row["decile"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for j, (bias_key, mae_key, title, bar_color, line_color) in enumerate(
        [
            ("bias_sbp", "mae_sbp", "SBP", "#5dade2", "#1f618d"),
            ("bias_dbp", "mae_dbp", "DBP", "#f5b041", "#af601a"),
        ]
    ):
        bias = [float(row[bias_key]) for row in rows]
        mae = [float(row[mae_key]) for row in rows]
        axes[j].bar(deciles, bias, color=bar_color, alpha=0.72)
        axes[j].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax2 = axes[j].twinx()
        ax2.plot(deciles, mae, color=line_color, marker="o", linewidth=2.0)
        axes[j].set_title(f"{title}: Uncertainty-Stratified Bias/MAE")
        axes[j].set_xlabel("Uncertainty decile")
        axes[j].set_ylabel("Bias (Pred - True, mmHg)")
        ax2.set_ylabel("MAE (mmHg)")
        axes[j].grid(True, linestyle="--", alpha=0.20)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_bias_stage_trace(output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    guard_rows = _load_csv_rows(tables_dir / "clinical_guard_bp_range_comparison.csv")
    bias_rows = _load_csv_rows(tables_dir / "high_bias_calibration_bp_range_comparison.csv")
    crisis_rows = _load_csv_rows(tables_dir / "crisis_tail_fusion_bp_range_comparison.csv")
    final_rows = _load_csv_rows(tables_dir / "bp_range_metrics.csv")
    if not guard_rows or not crisis_rows or not final_rows:
        return

    staged_rows: List[dict] = []
    for row in guard_rows:
        bp_range = str(row["bp_range"])
        staged_rows.append({"stage": "pre_guard", "bp_range": bp_range, "sbp_bias": float(row["base_bias_sbp"]), "dbp_bias": float(row["base_bias_dbp"])})
        staged_rows.append({"stage": "post_guard", "bp_range": bp_range, "sbp_bias": float(row["guarded_bias_sbp"]), "dbp_bias": float(row["guarded_bias_dbp"])})
    for row in bias_rows:
        staged_rows.append({"stage": "post_bias_cal", "bp_range": str(row["bp_range"]), "sbp_bias": float(row["guarded_bias_sbp"]), "dbp_bias": float(row["guarded_bias_dbp"])})
    for row in crisis_rows:
        staged_rows.append({"stage": "post_crisis_fusion", "bp_range": str(row["bp_range"]), "sbp_bias": float(row["guarded_bias_sbp"]), "dbp_bias": float(row["guarded_bias_dbp"])})
    for row in final_rows:
        staged_rows.append({"stage": "final_selected", "bp_range": str(row["bp_range"]), "sbp_bias": float(row["bias_sbp"]), "dbp_bias": float(row["bias_dbp"])})

    _write_csv(tables_dir / "bias_stage_trace.csv", staged_rows, ["stage", "bp_range", "sbp_bias", "dbp_bias"])
    _plot_bias_stage_trace(staged_rows, output_dir / "figures" / "bias_stage_trajectory.png")


def generate_internal_tradeoff(output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    cls_rows = _load_csv_rows(tables_dir / "classification_variant_summary.csv")
    reg_rows = _load_csv_rows(tables_dir / "regression_variant_summary.csv")
    if not cls_rows or not reg_rows:
        return

    cls_map = {str(row["variant"]): row for row in cls_rows}
    reg_map = {str(row["variant"]): row for row in reg_rows}
    candidates = [
        ("Opt-long ref", "optlong_from_reg", "optlong_corrected"),
        ("Dualmax ref", "dualmax_hybrid", "dualmax_corrected"),
        ("Stability selected", "stability_selected", "stability_selected"),
        ("Pre-fusion selected", "selected_prefusion", "meta_selected"),
        ("Final selected", "selected_final", "meta_selected"),
    ]
    rows: List[dict] = []
    for label, cls_variant, reg_variant in candidates:
        if cls_variant not in cls_map or reg_variant not in reg_map:
            continue
        rows.append(
            {
                "label": label,
                "classification_variant": cls_variant,
                "regression_variant": reg_variant,
                "acc": float(cls_map[cls_variant]["acc"]),
                "macro_f1": float(cls_map[cls_variant]["macro_f1"]),
                "balanced_acc": float(cls_map[cls_variant]["balanced_acc"]),
                "mae_mean": float(reg_map[reg_variant]["mae_mean"]),
                "bias_sbp": float(reg_map[reg_variant]["bias_sbp"]),
                "bias_dbp": float(reg_map[reg_variant]["bias_dbp"]),
            }
        )
    if not rows:
        return

    _write_csv(
        tables_dir / "internal_operating_tradeoff.csv",
        rows,
        ["label", "classification_variant", "regression_variant", "acc", "macro_f1", "balanced_acc", "mae_mean", "bias_sbp", "bias_dbp"],
    )
    _plot_internal_tradeoff(rows, output_dir / "figures" / "internal_operating_tradeoff.png")


def generate_uncertainty_bias_profile(output_dir: Path) -> None:
    artifact_path = output_dir / "artifacts" / "test_outputs_regression_selected.npz"
    if not artifact_path.exists():
        return
    with np.load(artifact_path) as arr:
        y_true = np.asarray(arr["y_true_reg"], dtype=np.float32)
        y_pred = np.asarray(arr["y_pred_reg"], dtype=np.float32)
        uncertainty = np.asarray(arr["uncertainty"], dtype=np.float32).reshape(-1)

    order = np.argsort(uncertainty)
    buckets = np.array_split(order, 10)
    rows: List[dict] = []
    for decile, idx in enumerate(buckets, start=1):
        if len(idx) == 0:
            continue
        err = y_pred[idx] - y_true[idx]
        abs_err = np.abs(err)
        rows.append(
            {
                "decile": int(decile),
                "n": int(len(idx)),
                "uncertainty_mean": float(uncertainty[idx].mean()),
                "bias_sbp": float(err[:, 0].mean()),
                "bias_dbp": float(err[:, 1].mean()),
                "mae_sbp": float(abs_err[:, 0].mean()),
                "mae_dbp": float(abs_err[:, 1].mean()),
            }
        )
    _write_csv(
        output_dir / "tables" / "uncertainty_bias_profile.csv",
        rows,
        ["decile", "n", "uncertainty_mean", "bias_sbp", "bias_dbp", "mae_sbp", "mae_dbp"],
    )
    _plot_uncertainty_bias_profile(rows, output_dir / "figures" / "uncertainty_bias_profile.png")


def update_literature_alignment(output_dir: Path) -> None:
    path = output_dir / "protocol_literature_alignment.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data["v10_12_inspirations"] = INSPIRATION_PAPERS
    data["v10_12_design_summary"] = [
        "Signed reliability-aware bias calibration lets the post-hoc corrector move predictions both upward and downward instead of only applying positive crisis/high shifts.",
        "Evidential safety fusion mixes classifier and regression-induced class evidence in log-probability space, with stronger fusion only when disagreement and reliability jointly support it.",
        "The additional plots expose how bias evolves across guard, bias calibration, crisis fusion, and final selection, which is useful for the paper narrative.",
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def generate_extra_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    for fn in (generate_bias_stage_trace, generate_internal_tradeoff, generate_uncertainty_bias_profile, update_literature_alignment):
        try:
            fn(output_dir)
        except Exception as exc:
            print(f"[v10.12] Skipped extra artifact step {fn.__name__}: {exc}")


def main():
    original_prev_build_cfg = prev_script.build_nextgen_cfg
    original_prev_build_optlong_cfg = prev_script.build_optlong_fulltrain_cfg
    original_prev_build_dualmax_cfg = prev_script.build_dualmax_fulltrain_cfg
    original_prev_run_self = prev_script._run_self_subprocess
    original_prev_final_output = prev_script.FINAL_OUTPUT_NAME
    original_prev_protocol_id = prev_script.FINAL_PROTOCOL_ID
    original_prev_optlong_output = prev_script.OPTLONG_FULLTRAIN_OUTPUT
    original_prev_dualmax_output = prev_script.DUALMAX_FULLTRAIN_OUTPUT
    original_prev_target_score = prev_script.targeted_classification_selection_score
    original_prev_target_candidate_score = prev_script.targeted_classification_candidate_score
    original_prev_target_robust_score = prev_script.targeted_robust_classification_score
    original_meta_apply_high_bias = meta_script.apply_high_bias_calibration
    original_meta_high_bias_cost = meta_script.high_bias_calibration_cost
    original_meta_search_high_bias = meta_script.search_high_bias_calibration_candidates
    original_meta_apply_safety = meta_script.apply_safety_class_fusion_prob
    original_meta_safety_score = meta_script.safety_class_fusion_score
    original_meta_search_safety = meta_script.search_safety_class_fusion

    try:
        _ensure_expected_dirs()
        archived_dirs = _prepare_fulltrain_cache_dirs()
        _write_run_status("running", "bootstrap", {"baseline_targets": _baseline_targets(), "archived_incomplete_dirs": archived_dirs})
        _patch_prev_outputs()
        prev_script.build_nextgen_cfg = build_nextgen_cfg
        prev_script.build_optlong_fulltrain_cfg = build_optlong_fulltrain_cfg
        prev_script.build_dualmax_fulltrain_cfg = build_dualmax_fulltrain_cfg
        prev_script._run_self_subprocess = _run_self_subprocess
        prev_script.targeted_classification_selection_score = promotion_aware_classification_selection_score
        prev_script.targeted_classification_candidate_score = promotion_aware_classification_candidate_score
        prev_script.targeted_robust_classification_score = promotion_aware_robust_classification_score
        meta_script.apply_high_bias_calibration = apply_high_bias_calibration
        meta_script.high_bias_calibration_cost = high_bias_calibration_cost
        meta_script.search_high_bias_calibration_candidates = search_high_bias_calibration_candidates
        meta_script.apply_safety_class_fusion_prob = apply_safety_class_fusion_prob
        meta_script.safety_class_fusion_score = safety_class_fusion_score
        meta_script.search_safety_class_fusion = search_safety_class_fusion
        _write_run_status("running", "backbone_and_protocol")
        for archived_dir in archived_dirs:
            print(f"[v10.12] Archived incomplete cache dir to force a clean retrain: {archived_dir}")
        prev_script.main()
        _write_run_status("running", "extra_artifacts")
        generate_extra_outputs(_output_dir())
        _write_run_status("completed", "done", {"protocol_summary_exists": bool((_output_dir() / "protocol_summary.json").exists())})
    except Exception as exc:
        _write_run_status("failed", "exception", {"error": str(exc)})
        raise
    finally:
        prev_script.build_nextgen_cfg = original_prev_build_cfg
        prev_script.build_optlong_fulltrain_cfg = original_prev_build_optlong_cfg
        prev_script.build_dualmax_fulltrain_cfg = original_prev_build_dualmax_cfg
        prev_script._run_self_subprocess = original_prev_run_self
        prev_script.FINAL_OUTPUT_NAME = original_prev_final_output
        prev_script.FINAL_PROTOCOL_ID = original_prev_protocol_id
        prev_script.OPTLONG_FULLTRAIN_OUTPUT = original_prev_optlong_output
        prev_script.DUALMAX_FULLTRAIN_OUTPUT = original_prev_dualmax_output
        prev_script.targeted_classification_selection_score = original_prev_target_score
        prev_script.targeted_classification_candidate_score = original_prev_target_candidate_score
        prev_script.targeted_robust_classification_score = original_prev_target_robust_score
        meta_script.apply_high_bias_calibration = original_meta_apply_high_bias
        meta_script.high_bias_calibration_cost = original_meta_high_bias_cost
        meta_script.search_high_bias_calibration_candidates = original_meta_search_high_bias
        meta_script.apply_safety_class_fusion_prob = original_meta_apply_safety
        meta_script.safety_class_fusion_score = original_meta_safety_score
        meta_script.search_safety_class_fusion = original_meta_search_safety


def _dispatch_subprocess_mode(argv: List[str]) -> int | None:
    _patch_prev_outputs()
    prev_script.build_optlong_fulltrain_cfg = build_optlong_fulltrain_cfg
    prev_script.build_dualmax_fulltrain_cfg = build_dualmax_fulltrain_cfg
    return prev_script._dispatch_subprocess_mode(argv)


if __name__ == "__main__":
    dispatch_code = _dispatch_subprocess_mode(sys.argv[1:])
    if dispatch_code is None:
        main()
    else:
        raise SystemExit(dispatch_code)
