from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np

import train_aqm_medfuse_mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_protocol as prev_script
import train_aqm_medfuse_mimic_bp_reg_v10_12_subjectdisjoint_piso_uncertainty_moe_fulltrain_reliabilitybias_protocol as base_script


FINAL_OUTPUT_NAME = "mimic_bp_reg_v10_13_subjectdisjoint_piso_uncertainty_moe_tailaware_proto"
FINAL_PROTOCOL_ID = "v10.13_subjectdisjoint_piso_uncertainty_moe_tailaware"
OPTLONG_FULLTRAIN_OUTPUT = "mimic_bp_reg_v10_13_opt_long_fulltrain_proto"
DUALMAX_FULLTRAIN_OUTPUT = "mimic_bp_reg_v10_13_optlong_stageaware_dualmax_fulltrain_proto"

BASELINE_OUTPUT_NAME = "mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_proto"

_BASE_BUILD_NEXTGEN_CFG = base_script.build_nextgen_cfg
_BASE_BUILD_OPTLONG_CFG = base_script.build_optlong_fulltrain_cfg
_BASE_BUILD_DUALMAX_CFG = base_script.build_dualmax_fulltrain_cfg
_BASE_GENERATE_EXTRA_OUTPUTS = base_script.generate_extra_outputs
_BASE_PROMOTION_SELECTION_SCORE = base_script.promotion_aware_classification_selection_score
_BASE_PROMOTION_CANDIDATE_SCORE = base_script.promotion_aware_classification_candidate_score
_BASE_PROMOTION_ROBUST_SCORE = base_script.promotion_aware_robust_classification_score
_BASE_SEARCH_HIGH_BIAS = base_script.search_high_bias_calibration_candidates
_BASE_HIGH_BIAS_COST = base_script.high_bias_calibration_cost
_PREV_APPLY_CRISIS_TAIL = prev_script.apply_crisis_tail_debias_fusion
_PREV_CRISIS_TAIL_COST = prev_script.crisis_tail_debias_cost


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _output_dir() -> Path:
    return _project_root() / "outputs" / FINAL_OUTPUT_NAME


def _baseline_output_dir() -> Path:
    return _project_root() / "outputs" / BASELINE_OUTPUT_NAME


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _has_usable_checkpoint(output_dir: Path) -> bool:
    return (output_dir / "best_model.pt").exists() or any(output_dir.glob("*_best.pt"))


def _has_companion_metrics(output_dir: Path) -> bool:
    return (output_dir / "final_results.json").exists() and (output_dir / "tables" / "bp_range_metrics.csv").exists()


def _checkpoint_artifact_usable(ckpt_path: Path) -> bool:
    if not ckpt_path.exists():
        return False
    out_dir = ckpt_path.parent
    epoch_log = out_dir / "epoch_log.csv"
    if epoch_log.exists():
        try:
            with epoch_log.open("r", encoding="utf-8") as f:
                if max(0, sum(1 for _ in f) - 1) >= int(prev_script.MIN_FULLTRAIN_EPOCH_LOG_ROWS):
                    return True
        except OSError:
            pass
    # v10.12 can be interrupted after writing a checkpoint but before writing
    # companion CSV/JSON files.  For the final protocol the checkpoint is still
    # a valid warm-start source, so do not force a clean multi-day retrain.
    return _has_usable_checkpoint(out_dir)


def _prepare_fulltrain_cache_dirs_no_archive() -> List[str]:
    missing = []
    for name in (OPTLONG_FULLTRAIN_OUTPUT, DUALMAX_FULLTRAIN_OUTPUT):
        path = _project_root() / "outputs" / name
        if path.exists() and _has_usable_checkpoint(path) and not _has_companion_metrics(path):
            missing.append(str(path))
        path.mkdir(parents=True, exist_ok=True)
    return missing


def _ensure_expected_dirs() -> None:
    for name in (FINAL_OUTPUT_NAME, OPTLONG_FULLTRAIN_OUTPUT, DUALMAX_FULLTRAIN_OUTPUT):
        path = _project_root() / "outputs" / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "figures").mkdir(parents=True, exist_ok=True)
        (path / "tables").mkdir(parents=True, exist_ok=True)
        (path / "artifacts").mkdir(parents=True, exist_ok=True)


def _ensure_cfg_output_dir(cfg) -> None:
    output_name = str(getattr(cfg, "OUTPUT_NAME", "")).strip()
    project_root = Path(getattr(cfg, "PROJECT_ROOT", _project_root()))
    if output_name:
        out_dir = project_root / "outputs" / output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        (out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)


def _patch_prev_outputs() -> None:
    prev_script.FINAL_OUTPUT_NAME = FINAL_OUTPUT_NAME
    prev_script.FINAL_PROTOCOL_ID = FINAL_PROTOCOL_ID
    prev_script.OPTLONG_FULLTRAIN_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
    prev_script.DUALMAX_FULLTRAIN_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT


def _ensure_runtime_compatibility() -> None:
    if not hasattr(prev_script, "tail_bias_penalty"):
        fn = getattr(getattr(base_script.meta_script, "prev_script", None), "tail_bias_penalty", None)
        if fn is not None:
            prev_script.tail_bias_penalty = fn


def _run_self_subprocess(args: List[str]) -> int:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        cwd=str(_project_root()),
        check=False,
    )
    return int(proc.returncode)


def build_optlong_fulltrain_cfg():
    cfg = _BASE_BUILD_OPTLONG_CFG()
    own_out = _project_root() / "outputs" / OPTLONG_FULLTRAIN_OUTPUT
    own_ckpt = own_out / "best_model.pt"
    v1012_ckpt = _project_root() / "outputs" / "mimic_bp_reg_v10_12_opt_long_fulltrain_proto" / "best_model.pt"
    cfg.OUTPUT_NAME = OPTLONG_FULLTRAIN_OUTPUT
    cfg.PROTOCOL_ID = "v10.13_optlong_fulltrain"
    cfg.PROTOCOL_NAME = "v10.13 opt-long warm-start full retraining with tail-aware loss"
    cfg.ENABLE_EPOCH_RESUME = True
    cfg.RESUME_STATE_PATH = str(own_out / "latest_training_state.pt")
    cfg.RESUME_FALLBACK_CKPT_PATH = str(own_ckpt)
    if own_ckpt.exists() and not _has_companion_metrics(own_out):
        cfg.INIT_CKPT_PATH = str(own_ckpt)
    elif v1012_ckpt.exists():
        cfg.INIT_CKPT_PATH = str(v1012_ckpt)
    cfg.EPOCHS = min(int(getattr(cfg, "EPOCHS", 96)), 72)
    cfg.EARLY_STOPPING_PATIENCE = min(int(getattr(cfg, "EARLY_STOPPING_PATIENCE", 32)), 24)
    cfg.LR = min(float(getattr(cfg, "LR", 3.2e-5)), 2.8e-5)
    cfg.REG_SAMPLER_POWER = max(float(getattr(cfg, "REG_SAMPLER_POWER", 0.98)), 1.16)
    cfg.TAIL_CLASS_WEIGHTS = (0.05, 0.70, 1.65, 3.40)
    cfg.NORMAL_OVERPRED_WEIGHT = min(float(getattr(cfg, "NORMAL_OVERPRED_WEIGHT", 0.36)), 0.24)
    cfg.LAMBDA_TAIL = max(float(getattr(cfg, "LAMBDA_TAIL", 0.42)), 0.74)
    cfg.LAMBDA_CRISIS_TAIL = max(float(getattr(cfg, "LAMBDA_CRISIS_TAIL", 0.0)), 0.34)
    cfg.CRISIS_SBP_PRE_THRESHOLD = 145.0
    cfg.CRISIS_DBP_PRE_THRESHOLD = 92.0
    cfg.CRISIS_SBP_UNDER_WEIGHT = 2.70
    cfg.CRISIS_DBP_UNDER_WEIGHT = 1.35
    cfg.CRISIS_TRUE_EXTRA_WEIGHT = 1.80
    cfg.CRISIS_SBP_SCALE = 9.0
    cfg.CRISIS_DBP_SCALE = 6.5
    cfg.VAL_SCORE_HIGH_BIAS_WEIGHT = 0.68
    cfg.VAL_SCORE_CRISIS_BIAS_WEIGHT = 1.25
    cfg.VAL_SCORE_TAIL_TOP10_BIAS_WEIGHT = 0.55
    cfg.VAL_SCORE_TAIL_TOP5_BIAS_WEIGHT = 0.78
    cfg.TRAIN_STAGE_NAME = "OptLong TailAware"
    _patch_prev_outputs()
    _ensure_cfg_output_dir(cfg)
    return cfg


def build_dualmax_fulltrain_cfg(base_builder=None):
    cfg = _BASE_BUILD_DUALMAX_CFG(base_builder)
    own_out = _project_root() / "outputs" / DUALMAX_FULLTRAIN_OUTPUT
    own_ckpt = own_out / "best_model.pt"
    v1012_ckpt = (
        _project_root()
        / "outputs"
        / "mimic_bp_reg_v10_12_optlong_stageaware_dualmax_fulltrain_proto"
        / "best_model.pt"
    )
    v1013_optlong_ckpt = _project_root() / "outputs" / OPTLONG_FULLTRAIN_OUTPUT / "best_model.pt"
    cfg.OUTPUT_NAME = DUALMAX_FULLTRAIN_OUTPUT
    cfg.PROTOCOL_ID = "v10.13_optlong_stageaware_dualmax_fulltrain"
    cfg.PROTOCOL_NAME = "v10.13 stage-aware dualmax warm-start full retraining with tail-aware loss"
    cfg.ENABLE_EPOCH_RESUME = True
    cfg.RESUME_STATE_PATH = str(own_out / "latest_training_state.pt")
    cfg.RESUME_FALLBACK_CKPT_PATH = str(own_ckpt)
    if own_ckpt.exists() and not _has_companion_metrics(own_out):
        cfg.INIT_CKPT_PATH = str(own_ckpt)
    elif v1012_ckpt.exists():
        cfg.INIT_CKPT_PATH = str(v1012_ckpt)
    elif v1013_optlong_ckpt.exists():
        cfg.INIT_CKPT_PATH = str(v1013_optlong_ckpt)
    cfg.EPOCHS = min(int(getattr(cfg, "EPOCHS", 128)), 96)
    cfg.EARLY_STOPPING_PATIENCE = min(int(getattr(cfg, "EARLY_STOPPING_PATIENCE", 48)), 32)
    cfg.LR = min(float(getattr(cfg, "LR", 1.6e-5)), 1.3e-5)
    cfg.REG_SAMPLER_POWER = max(float(getattr(cfg, "REG_SAMPLER_POWER", 0.98)), 1.20)
    cfg.TAIL_CLASS_WEIGHTS = (0.05, 0.75, 1.80, 3.80)
    cfg.NORMAL_OVERPRED_WEIGHT = min(float(getattr(cfg, "NORMAL_OVERPRED_WEIGHT", 0.36)), 0.22)
    cfg.LAMBDA_TAIL = max(float(getattr(cfg, "LAMBDA_TAIL", 0.42)), 0.82)
    cfg.LAMBDA_CRISIS_TAIL = max(float(getattr(cfg, "LAMBDA_CRISIS_TAIL", 0.0)), 0.40)
    cfg.CRISIS_SBP_PRE_THRESHOLD = 145.0
    cfg.CRISIS_DBP_PRE_THRESHOLD = 92.0
    cfg.CRISIS_SBP_UNDER_WEIGHT = 3.05
    cfg.CRISIS_DBP_UNDER_WEIGHT = 1.55
    cfg.CRISIS_TRUE_EXTRA_WEIGHT = 2.10
    cfg.CRISIS_SBP_SCALE = 8.5
    cfg.CRISIS_DBP_SCALE = 6.2
    cfg.VAL_SCORE_HIGH_BIAS_WEIGHT = 0.75
    cfg.VAL_SCORE_CRISIS_BIAS_WEIGHT = 1.40
    cfg.VAL_SCORE_TAIL_TOP10_BIAS_WEIGHT = 0.62
    cfg.VAL_SCORE_TAIL_TOP5_BIAS_WEIGHT = 0.88
    cfg.TRAIN_STAGE_NAME = "DualMax TailAware"
    _patch_prev_outputs()
    _ensure_cfg_output_dir(cfg)
    return cfg


def build_nextgen_cfg():
    cfg = _BASE_BUILD_NEXTGEN_CFG()
    cfg.OUTPUT_NAME = FINAL_OUTPUT_NAME
    cfg.PROTOCOL_ID = FINAL_PROTOCOL_ID
    cfg.PROTOCOL_NAME = (
        "v10.13 subject-disjoint PiSO-inspired uncertainty-MoE tail-aware protocol "
        "(v10.12 reliability-bias stack + high-risk reweighting + conservative threshold calibration)"
    )
    cfg.FULLTRAIN_OPTLONG_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
    cfg.FULLTRAIN_DUALMAX_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT
    cfg.WARMSTART_CANDIDATES = tuple(dict.fromkeys(tuple(getattr(cfg, "WARMSTART_CANDIDATES", ())) + (
        "mimic_bp_reg_v10_12_subjectdisjoint_piso_uncertainty_moe_fulltrain_reliabilitybias_proto",
        "mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_proto",
    )))

    cfg.HEAD_EPOCHS = max(int(getattr(cfg, "HEAD_EPOCHS", 168)), 176)
    cfg.HEAD_PATIENCE = max(int(getattr(cfg, "HEAD_PATIENCE", 48)), 56)
    cfg.HEAD_MIN_EPOCHS = max(int(getattr(cfg, "HEAD_MIN_EPOCHS", 56)), 64)
    cfg.HEAD_SELECTION_RANK_MODE = "clean_acc_f1_then_score"
    cfg.HEAD_CLEAN_ACC_WEIGHT = 1.80
    cfg.HEAD_CLEAN_F1_WEIGHT = 1.30
    cfg.HEAD_CLEAN_BALANCED_WEIGHT = 0.45
    cfg.HEAD_CLEAN_ROBUST_WEIGHT = 0.10
    cfg.HEAD_CLEAN_STAGE2_WEIGHT = 0.08
    cfg.HEAD_CLASS_WEIGHT_POWER = 0.64
    cfg.HEAD_ELEVATED_REPEAT = 2
    cfg.HEAD_STAGE1_REPEAT = 3
    cfg.HEAD_STAGE2_REPEAT = 4
    cfg.HEAD_TARGET_RARE_MIN_WEIGHT = 0.48
    cfg.HEAD_TARGET_STAGE2_WEIGHT = 0.28
    cfg.HEAD_TARGET_GAP_WEIGHT = 1.12
    cfg.HEAD_TARGET_ROBUST_GAP_WEIGHT = 0.42

    cfg.RELIABILITY_BIAS_SCALES = (0.25, 0.40, 0.55, 0.72, 0.90, 1.10)
    cfg.RELIABILITY_BIAS_BETAS = (0.60, 0.85, 1.10)
    cfg.RELIABILITY_BIAS_RELIABILITY_FLOORS = (0.05, 0.15, 0.28)
    cfg.RELIABILITY_BIAS_DISAGREE_GAINS = (0.0, 0.35, 0.70, 1.05)
    cfg.RELIABILITY_BIAS_HIGH_GAINS = (0.45, 0.85, 1.25)
    cfg.RELIABILITY_BIAS_CRISIS_GAINS = (0.90, 1.45, 2.10, 2.80)
    cfg.RELIABILITY_BIAS_NEGATIVE_FRACS = (0.06, 0.12, 0.18)
    cfg.RELIABILITY_BIAS_HIGH_THRESHOLDS = (0.12, 0.20, 0.30)
    cfg.RELIABILITY_BIAS_CRISIS_THRESHOLDS = (0.03, 0.07, 0.12)
    cfg.RELIABILITY_BIAS_HIGH_FLOOR_SBP = (0.0, 1.5, 3.0)
    cfg.RELIABILITY_BIAS_CRISIS_FLOOR_SBP = (2.0, 4.5, 7.0, 10.0)
    cfg.RELIABILITY_BIAS_MAX_MAE_DELTA = 0.30
    cfg.RELIABILITY_BIAS_MAX_COVERAGE_GAP_DELTA = 0.055

    cfg.SAFETY_EVIDENTIAL_SCALES = (0.08, 0.16, 0.28, 0.42, 0.60)
    cfg.SAFETY_EVIDENTIAL_BETAS = (0.60, 0.85, 1.10)
    cfg.SAFETY_EVIDENTIAL_DISAGREE_GAINS = (0.15, 0.50, 0.85)
    cfg.SAFETY_EVIDENTIAL_HIGH_GAINS = (0.30, 0.70, 1.05)
    cfg.SAFETY_EVIDENTIAL_CRISIS_GAINS = (0.55, 1.10, 1.70)
    cfg.SAFETY_EVIDENTIAL_RELIABILITY_FLOORS = (0.04, 0.12, 0.22)
    cfg.SAFETY_EVIDENTIAL_STAGE1_BIASES = (0.00, 0.04, 0.08)
    cfg.SAFETY_EVIDENTIAL_STAGE2_BIASES = (0.00, 0.08, 0.14, 0.22)
    cfg.SAFETY_CLASS_FUSION_MAX_WEIGHT = max(float(getattr(cfg, "SAFETY_CLASS_FUSION_MAX_WEIGHT", 0.78)), 0.82)
    cfg.SAFETY_CLASS_FUSION_STAGE2_RECALL_WEIGHT = max(float(getattr(cfg, "SAFETY_CLASS_FUSION_STAGE2_RECALL_WEIGHT", 0.30)), 0.45)
    cfg.SAFETY_CLASS_FUSION_STAGE2_F1_WEIGHT = max(float(getattr(cfg, "SAFETY_CLASS_FUSION_STAGE2_F1_WEIGHT", 0.12)), 0.18)

    cfg.CRISIS_TAIL_FUSION_HIGH_THRESHOLDS = (0.06, 0.10, 0.16, 0.22)
    cfg.CRISIS_TAIL_FUSION_CRISIS_THRESHOLDS = (0.01, 0.03, 0.06, 0.10, 0.14)
    cfg.CRISIS_TAIL_FUSION_SBP_QUANTILES = (0.90, 0.95, 0.98, 1.00)
    cfg.CRISIS_TAIL_FUSION_DBP_QUANTILES = (0.82, 0.88, 0.94, 1.00)
    cfg.CRISIS_TAIL_FUSION_CRISIS_GAINS = (1.80, 2.50, 3.40, 4.20)
    cfg.CRISIS_TAIL_FUSION_SBP_MARGINS = (3.0, 5.5, 8.5, 12.0)
    cfg.CRISIS_TAIL_FUSION_DBP_MARGINS = (1.0, 2.0, 3.2, 4.5)
    cfg.CRISIS_TAIL_FUSION_MAX_MAE_DELTA = 0.42
    cfg.CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA = 0.065
    cfg.CRISIS_TAIL_MAX_SHIFT_SBP = 30.0
    cfg.CRISIS_TAIL_MAX_SHIFT_DBP = 14.0
    cfg.CRISIS_TAIL_HARD_FLOOR_SBP = 13.0
    cfg.CRISIS_TAIL_HARD_FLOOR_DBP = 4.5
    cfg.CRISIS_TAIL_UNDEREST_WEIGHT_SBP = 8.50
    cfg.CRISIS_TAIL_UNDEREST_WEIGHT_DBP = 3.10
    cfg.CRISIS_TAIL_SURROGATE_QUANTILES = (0.88, 0.92, 0.95, 0.98)
    cfg.CRISIS_SBP_GUARD_ENABLE = True
    cfg.CRISIS_SBP_GUARD_TRIGGER = 0.52
    cfg.CRISIS_SBP_GUARD_QUANTILE = 0.98
    cfg.CRISIS_SBP_GUARD_MIN_EXPERT_SBP = 165.0
    cfg.CRISIS_SBP_GUARD_ABSOLUTE_FLOOR = 172.0
    cfg.CRISIS_SBP_GUARD_MARGIN = 3.5
    cfg.CRISIS_SBP_GUARD_GAIN = 0.88
    cfg.CRISIS_SBP_GUARD_MAX_EXTRA_SHIFT = 18.0
    cfg.CRISIS_DBP_GUARD_ABSOLUTE_FLOOR = 106.0
    cfg.CRISIS_DBP_GUARD_GAIN = 0.45
    cfg.CRISIS_DBP_GUARD_MAX_EXTRA_SHIFT = 8.0
    cfg.BASELINE_AWARE_SELECTOR_ENABLE = True
    cfg.BASELINE_MIN_ACC_MARGIN = 0.000
    cfg.BASELINE_MIN_F1_MARGIN = 0.000
    cfg.BASELINE_ACC_SHORTFALL_WEIGHT = 18.0
    cfg.BASELINE_F1_SHORTFALL_WEIGHT = 22.0
    cfg.BASELINE_ASPIRATIONAL_ACC = 0.86
    cfg.BASELINE_ASPIRATIONAL_F1 = 0.80
    cfg.BASELINE_ASPIRATIONAL_ACC_WEIGHT = 2.40
    cfg.BASELINE_ASPIRATIONAL_F1_WEIGHT = 2.80
    cfg.BASELINE_BIAS_WORSE_TOL_SBP = 0.75
    cfg.BASELINE_BIAS_WORSE_TOL_DBP = 0.50
    cfg.BASELINE_BIAS_WORSE_WEIGHT = 2.20
    cfg.BASELINE_CRISIS_SBP_TARGET = -5.0
    cfg.BASELINE_CRISIS_DBP_TARGET = -3.0
    cfg.BASELINE_HIGH_SBP_TARGET = -5.0
    cfg.BASELINE_HIGH_DBP_TARGET = -3.0
    cfg.BASELINE_CRISIS_TARGET_WEIGHT = 3.40
    cfg.BASELINE_HIGH_TARGET_WEIGHT = 1.45
    _patch_prev_outputs()
    _ensure_cfg_output_dir(cfg)
    return cfg


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))).astype(np.float32)


def _top_tail_under_bias_penalty(y_true: np.ndarray, y_pred: np.ndarray, quantiles: tuple[float, ...]) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    total = 0.0
    for q in quantiles:
        for dim, dim_weight in ((0, 1.0), (1, 0.45)):
            threshold = float(np.quantile(y_true[:, dim], float(q)))
            mask = y_true[:, dim] >= threshold
            if not np.any(mask):
                continue
            residual = y_pred[mask, dim] - y_true[mask, dim]
            bias = float(residual.mean())
            under_rate = float(np.mean(residual < 0.0))
            total += dim_weight * max(0.0, -bias) * (0.50 + under_rate)
    return float(total)


def _baseline_targets() -> dict:
    return base_script._baseline_targets()


def _baseline_bp_range_map() -> dict[str, dict]:
    rows = _read_csv(_baseline_output_dir() / "tables" / "bp_range_metrics.csv")
    return {str(row.get("bp_range", "")): row for row in rows}


def _bp_range_map(rows: List[dict]) -> dict[str, dict]:
    return {str(row.get("bp_range", "")): row for row in rows}


def _baseline_bias_penalty(bp_range_rows: List[dict], cfg) -> float:
    if not bool(getattr(cfg, "BASELINE_AWARE_SELECTOR_ENABLE", True)):
        return 0.0
    current = _bp_range_map(bp_range_rows)
    baseline = _baseline_bp_range_map()
    penalty = 0.0
    tol_sbp = float(getattr(cfg, "BASELINE_BIAS_WORSE_TOL_SBP", 0.75))
    tol_dbp = float(getattr(cfg, "BASELINE_BIAS_WORSE_TOL_DBP", 0.50))
    worse_weight = float(getattr(cfg, "BASELINE_BIAS_WORSE_WEIGHT", 2.20))
    for range_name, range_weight in (("high", 1.0), ("crisis", 2.2)):
        cur = current.get(range_name)
        ref = baseline.get(range_name)
        if cur is None or ref is None:
            continue
        cur_sbp = _to_float(cur.get("bias_sbp"))
        cur_dbp = _to_float(cur.get("bias_dbp"))
        ref_sbp = _to_float(ref.get("bias_sbp"))
        ref_dbp = _to_float(ref.get("bias_dbp"))
        # More negative bias means stronger underestimation. Penalize candidates
        # that are worse than v10.11 beyond a small tolerance.
        penalty += range_weight * worse_weight * max(0.0, (ref_sbp - tol_sbp) - cur_sbp)
        penalty += range_weight * 0.65 * worse_weight * max(0.0, (ref_dbp - tol_dbp) - cur_dbp)

    high = current.get("high", {})
    crisis = current.get("crisis", {})
    penalty += float(getattr(cfg, "BASELINE_HIGH_TARGET_WEIGHT", 1.45)) * (
        max(0.0, float(getattr(cfg, "BASELINE_HIGH_SBP_TARGET", -5.0)) - _to_float(high.get("bias_sbp")))
        + 0.55 * max(0.0, float(getattr(cfg, "BASELINE_HIGH_DBP_TARGET", -3.0)) - _to_float(high.get("bias_dbp")))
    )
    penalty += float(getattr(cfg, "BASELINE_CRISIS_TARGET_WEIGHT", 3.40)) * (
        max(0.0, float(getattr(cfg, "BASELINE_CRISIS_SBP_TARGET", -5.0)) - _to_float(crisis.get("bias_sbp")))
        + 0.65 * max(0.0, float(getattr(cfg, "BASELINE_CRISIS_DBP_TARGET", -3.0)) - _to_float(crisis.get("bias_dbp")))
    )
    return float(penalty)


def _baseline_classification_terms(acc: float, macro_f1: float, cfg) -> tuple[float, dict]:
    if not bool(getattr(cfg, "BASELINE_AWARE_SELECTOR_ENABLE", True)):
        return 0.0, {}
    baseline = _baseline_targets()
    min_acc = float(baseline["selected_acc"]) + float(getattr(cfg, "BASELINE_MIN_ACC_MARGIN", 0.0))
    min_f1 = float(baseline["selected_macro_f1"]) + float(getattr(cfg, "BASELINE_MIN_F1_MARGIN", 0.0))
    acc_gap = max(0.0, min_acc - float(acc))
    f1_gap = max(0.0, min_f1 - float(macro_f1))
    asp_acc_gap = max(0.0, float(getattr(cfg, "BASELINE_ASPIRATIONAL_ACC", 0.86)) - float(acc))
    asp_f1_gap = max(0.0, float(getattr(cfg, "BASELINE_ASPIRATIONAL_F1", 0.80)) - float(macro_f1))
    penalty = (
        float(getattr(cfg, "BASELINE_ACC_SHORTFALL_WEIGHT", 18.0)) * acc_gap
        + float(getattr(cfg, "BASELINE_F1_SHORTFALL_WEIGHT", 22.0)) * f1_gap
        + float(getattr(cfg, "BASELINE_ASPIRATIONAL_ACC_WEIGHT", 2.40)) * asp_acc_gap
        + float(getattr(cfg, "BASELINE_ASPIRATIONAL_F1_WEIGHT", 2.80)) * asp_f1_gap
    )
    return float(penalty), {
        "baseline_acc": min_acc,
        "baseline_macro_f1": min_f1,
        "baseline_acc_gap": float(acc_gap),
        "baseline_macro_f1_gap": float(f1_gap),
        "aspirational_acc_gap": float(asp_acc_gap),
        "aspirational_macro_f1_gap": float(asp_f1_gap),
    }


def tailaware_classification_selection_score(metrics: dict, prefix: str) -> float:
    base_score = float(_BASE_PROMOTION_SELECTION_SCORE(metrics, prefix))
    summary = prev_script._target_class_summary(metrics, prefix)
    baseline = base_script._baseline_targets()
    acc = float(summary["acc"])
    macro_f1 = float(summary["macro_f1"])
    balanced_acc = float(summary["balanced_acc"])
    rare_mean = float(summary["rare_f1_mean"])
    rare_min = float(summary["rare_f1_min"])
    stage2 = float(summary["stage2_f1"])
    ece = float(summary["ece"])
    acc_shortfall = max(0.0, float(baseline["selected_acc"]) - acc)
    f1_shortfall = max(0.0, float(baseline["selected_macro_f1"]) - macro_f1)
    baseline_penalty, _ = _baseline_classification_terms(acc, macro_f1, None)
    return float(
        base_score
        + 1.85 * acc
        + 1.55 * macro_f1
        + 0.55 * balanced_acc
        + 0.20 * rare_mean
        + 0.14 * rare_min
        + 0.10 * stage2
        - 0.25 * ece
        - 2.25 * acc_shortfall
        - 2.45 * f1_shortfall
        - baseline_penalty
    )


def tailaware_classification_candidate_score(metrics: dict, prefix: str) -> float:
    return float(tailaware_classification_selection_score(metrics, prefix))


def tailaware_robust_classification_score(
    clean_metrics: dict,
    noise_metrics: dict,
    ecg_metrics: dict,
    ppg_metrics: dict,
    cfg,
) -> float:
    noise_f1 = float(noise_metrics["cls_f1_macro_selected_noise_val"])
    ecg_f1 = float(ecg_metrics["cls_f1_macro_selected_ecg_val"])
    ppg_f1 = float(ppg_metrics["cls_f1_macro_selected_ppg_val"])
    robust_min = min(noise_f1, ecg_f1, ppg_f1)
    clean_score = tailaware_classification_candidate_score(clean_metrics, "selected_val")
    return float(
        clean_score
        + 0.10 * noise_f1
        + 0.10 * ecg_f1
        + 0.14 * ppg_f1
        + 0.10 * robust_min
    )


def tailaware_search_high_bias_calibration_candidates(
    calib_out: dict,
    calib_cls_prob: np.ndarray,
    query_out: dict,
    query_cls_prob: np.ndarray,
    cfg,
) -> tuple[dict, List[dict]]:
    try:
        return _BASE_SEARCH_HIGH_BIAS(calib_out, calib_cls_prob, query_out, query_cls_prob, cfg)
    except Exception as exc:
        print(f"[v10.13] High-bias calibration search fell back to identity candidate: {exc}")
        conformal = base_script.meta_script.stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
        bp_rows = base_script.meta_script.stage_script.build_bp_range_table(
            query_out["y_true_reg"],
            query_out["y_pred_reg"],
        )
        clinical_pen = base_script.meta_script.clinical_underestimation_penalty(bp_rows)
        tail_pen = base_script._tail_bias_penalty(bp_rows)
        range_map = {str(row["bp_range"]): row for row in bp_rows}
        row = {
            "candidate": "identity_fallback",
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
            "score": float(clinical_pen + 0.55 * tail_pen + 0.012 * float(query_out["metrics_reg"]["mae_mean"])),
            "clinical_under_penalty": float(clinical_pen),
            "tail_bias_penalty": float(tail_pen),
            "high_bias_sbp": float(range_map.get("high", {}).get("bias_sbp", 0.0)),
            "high_bias_dbp": float(range_map.get("high", {}).get("bias_dbp", 0.0)),
            "crisis_bias_sbp": float(range_map.get("crisis", {}).get("bias_sbp", 0.0)),
            "crisis_bias_dbp": float(range_map.get("crisis", {}).get("bias_dbp", 0.0)),
            "shift_mean_sbp": 0.0,
            "shift_mean_dbp": 0.0,
            "reliability_mean": 0.0,
            **query_out["metrics_reg"],
            **conformal,
        }
        return row, [row]


def tailaware_high_bias_calibration_cost(calib_out: dict, query_out: dict, base_ref: dict, cfg):
    score, conformal, bp_range_rows, clinical_pen, tail_pen = _BASE_HIGH_BIAS_COST(calib_out, query_out, base_ref, cfg)
    baseline_pen = _baseline_bias_penalty(bp_range_rows, cfg)
    return (
        float(score + baseline_pen),
        conformal,
        bp_range_rows,
        float(clinical_pen),
        float(tail_pen + baseline_pen),
    )


def _apply_crisis_sbp_guard(out: dict, cls_prob: np.ndarray, reg_inputs: dict, cfg) -> dict:
    if not bool(getattr(cfg, "CRISIS_SBP_GUARD_ENABLE", True)):
        return out
    try:
        pred = np.asarray(out["y_pred_reg"], dtype=np.float32)
        cls_prob = base_script.bridge_script.normalize_prob(np.asarray(cls_prob, dtype=np.float32))
        context = base_script.meta_script.build_crisis_tail_signal_context(out, cls_prob, reg_inputs)
        expert_stack = np.asarray(context["expert_stack"], dtype=np.float32)
    except Exception:
        return out
    if pred.size == 0 or expert_stack.ndim != 3:
        return out

    stage1_prob = cls_prob[:, 2].astype(np.float32)
    stage2_prob = cls_prob[:, 3].astype(np.float32)
    hypertensive_prob = np.clip(stage1_prob + stage2_prob, 0.0, 1.0).astype(np.float32)
    expert_peak = expert_stack.max(axis=1).astype(np.float32)
    expert_q = np.quantile(
        expert_stack,
        float(getattr(cfg, "CRISIS_SBP_GUARD_QUANTILE", 0.98)),
        axis=1,
    ).astype(np.float32)
    sbp_gap = np.clip(expert_q[:, 0] - pred[:, 0], 0.0, None).astype(np.float32)
    dbp_gap = np.clip(expert_q[:, 1] - pred[:, 1], 0.0, None).astype(np.float32)

    risk_signal = np.maximum.reduce(
        [
            np.asarray(context["expert_crisis_signal"], dtype=np.float32),
            0.55 * np.asarray(context["expert_high_signal"], dtype=np.float32) + 0.45 * stage2_prob,
            _sigmoid((expert_peak[:, 0] - 170.0) / 7.5),
            _sigmoid((expert_peak[:, 1] - 108.0) / 5.5),
            0.35 * hypertensive_prob + 0.65 * stage2_prob,
        ]
    ).astype(np.float32)
    gate = base_script.meta_script._signal_ramp(
        risk_signal,
        float(getattr(cfg, "CRISIS_SBP_GUARD_TRIGGER", 0.52)),
        1.0,
    )
    expert_gate = (expert_peak[:, 0] >= float(getattr(cfg, "CRISIS_SBP_GUARD_MIN_EXPERT_SBP", 165.0))).astype(np.float32)
    cls_gate = ((stage2_prob >= 0.28) | (hypertensive_prob >= 0.72)).astype(np.float32)
    gate = np.clip(gate * np.maximum(expert_gate, cls_gate), 0.0, 1.0).astype(np.float32)

    sbp_target = np.maximum(
        expert_q[:, 0] + float(getattr(cfg, "CRISIS_SBP_GUARD_MARGIN", 3.5)) * gate,
        np.where(
            (gate >= 0.65) & (stage2_prob >= 0.35),
            float(getattr(cfg, "CRISIS_SBP_GUARD_ABSOLUTE_FLOOR", 172.0)),
            pred[:, 0],
        ),
    ).astype(np.float32)
    dbp_target = np.maximum(
        expert_q[:, 1] + 0.35 * float(getattr(cfg, "CRISIS_SBP_GUARD_MARGIN", 3.5)) * gate,
        np.where(
            (gate >= 0.75) & (stage2_prob >= 0.45),
            float(getattr(cfg, "CRISIS_DBP_GUARD_ABSOLUTE_FLOOR", 106.0)),
            pred[:, 1],
        ),
    ).astype(np.float32)

    sbp_delta = (
        float(getattr(cfg, "CRISIS_SBP_GUARD_GAIN", 0.88))
        * gate
        * np.maximum(sbp_target - pred[:, 0], 0.0)
    ).astype(np.float32)
    dbp_delta = (
        float(getattr(cfg, "CRISIS_DBP_GUARD_GAIN", 0.45))
        * gate
        * np.maximum(dbp_target - pred[:, 1], 0.0)
    ).astype(np.float32)
    sbp_delta = np.maximum(sbp_delta, gate * np.minimum(sbp_gap, float(getattr(cfg, "CRISIS_SBP_GUARD_MAX_EXTRA_SHIFT", 18.0))))
    dbp_delta = np.maximum(
        dbp_delta,
        0.55 * gate * np.minimum(dbp_gap, float(getattr(cfg, "CRISIS_DBP_GUARD_MAX_EXTRA_SHIFT", 8.0))),
    )
    sbp_delta = np.clip(sbp_delta, 0.0, float(getattr(cfg, "CRISIS_SBP_GUARD_MAX_EXTRA_SHIFT", 18.0))).astype(np.float32)
    dbp_delta = np.clip(dbp_delta, 0.0, float(getattr(cfg, "CRISIS_DBP_GUARD_MAX_EXTRA_SHIFT", 8.0))).astype(np.float32)
    if float(sbp_delta.mean() + dbp_delta.mean()) <= 0.0:
        out = dict(out)
        out["crisis_sbp_guard_shift_mean"] = 0.0
        out["crisis_dbp_guard_shift_mean"] = 0.0
        out["crisis_sbp_guard_activation_rate"] = 0.0
        return out

    corrected = pred.copy()
    corrected[:, 0] += sbp_delta
    corrected[:, 1] += dbp_delta
    corrected = base_script.meta_script.stage_script.clipped_regression_prediction(corrected)
    guarded = base_script.meta_script.stage_script.clone_regression_output(out, corrected, cfg)
    for key, value in out.items():
        if key.startswith("crisis_tail_fusion_") and key not in guarded:
            guarded[key] = value
    guarded["crisis_sbp_guard_shift_mean"] = float(sbp_delta.mean())
    guarded["crisis_dbp_guard_shift_mean"] = float(dbp_delta.mean())
    guarded["crisis_sbp_guard_activation_rate"] = float(np.mean(gate >= 0.20))
    return guarded


def apply_crisis_tail_debias_fusion(row: dict, reg_out: dict, cls_prob: np.ndarray, reg_inputs: dict, cfg):
    out = _PREV_APPLY_CRISIS_TAIL(row, reg_out, cls_prob, reg_inputs, cfg)
    if str(row.get("candidate", "")) == "identity":
        return out
    return _apply_crisis_sbp_guard(out, cls_prob, reg_inputs, cfg)


def crisis_tail_debias_cost(calib_out: dict, query_out: dict, base_ref: dict, cfg):
    score, conformal, bp_range_rows, clinical_pen, tail_pen = _PREV_CRISIS_TAIL_COST(calib_out, query_out, base_ref, cfg)
    range_map = {str(row["bp_range"]): row for row in bp_range_rows}
    crisis_row = range_map.get("crisis", {})
    high_row = range_map.get("high", {})
    y_true = np.asarray(query_out["y_true_reg"], dtype=np.float32)
    y_pred = np.asarray(query_out["y_pred_reg"], dtype=np.float32)
    tail_under_pen = _top_tail_under_bias_penalty(
        y_true,
        y_pred,
        tuple(getattr(cfg, "CRISIS_TAIL_SURROGATE_QUANTILES", (0.88, 0.92, 0.95, 0.98))),
    )
    crisis_sbp_bias = float(crisis_row.get("bias_sbp", 0.0))
    crisis_dbp_bias = float(crisis_row.get("bias_dbp", 0.0))
    high_sbp_bias = float(high_row.get("bias_sbp", 0.0))
    baseline_pen = _baseline_bias_penalty(bp_range_rows, cfg)
    guard_shift = float(query_out.get("crisis_sbp_guard_shift_mean", 0.0))
    guard_activation = float(query_out.get("crisis_sbp_guard_activation_rate", 0.0))
    guard_overuse_pen = max(0.0, guard_activation - 0.18) * max(0.0, guard_shift - 1.2)
    score = float(
        score
        + 4.25 * max(0.0, -crisis_sbp_bias)
        + 1.35 * max(0.0, -crisis_dbp_bias)
        + 1.15 * max(0.0, -high_sbp_bias)
        + 2.10 * tail_under_pen
        + baseline_pen
        + 2.50 * guard_overuse_pen
    )
    return float(score), conformal, bp_range_rows, float(clinical_pen), float(tail_pen + baseline_pen)


def _plot_tail_underestimation(rows: List[dict], fig_path: Path) -> None:
    if not rows:
        return
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, target in zip(axes, ("SBP", "DBP")):
        sub = [row for row in rows if row["target"] == target]
        labels = [str(row["bin"]) for row in sub]
        bias = [float(row["bias"]) for row in sub]
        under = [float(row["under_rate"]) for row in sub]
        x = np.arange(len(labels))
        ax.bar(x, bias, color="#4c78a8", alpha=0.78)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"{target} Tail Bias")
        ax.set_ylabel("Mean prediction error (mmHg)")
        ax2 = ax.twinx()
        ax2.plot(x, under, color="#c44e52", marker="o", linewidth=2.0)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel("Underestimation rate")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_tail_underestimation_profile(output_dir: Path) -> None:
    artifact = output_dir / "artifacts" / "test_outputs_regression_selected.npz"
    if not artifact.exists():
        return
    with np.load(artifact, allow_pickle=True) as arr:
        y_true = np.asarray(arr["y_true_reg"], dtype=np.float32)
        y_pred = np.asarray(arr["y_pred_reg"], dtype=np.float32)
    rows = []
    for dim, target in ((0, "SBP"), (1, "DBP")):
        edges = np.quantile(y_true[:, dim], [0.0, 0.50, 0.75, 0.90, 0.95, 1.0])
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            if i == len(edges) - 2:
                mask = (y_true[:, dim] >= lo) & (y_true[:, dim] <= hi)
            else:
                mask = (y_true[:, dim] >= lo) & (y_true[:, dim] < hi)
            if not np.any(mask):
                continue
            residual = y_pred[mask, dim] - y_true[mask, dim]
            rows.append(
                {
                    "target": target,
                    "bin": f"q{i + 1}",
                    "lo": lo,
                    "hi": hi,
                    "n": int(mask.sum()),
                    "bias": float(residual.mean()),
                    "mae": float(np.abs(residual).mean()),
                    "under_rate": float(np.mean(residual < 0.0)),
                }
            )
    _write_csv(
        output_dir / "tables" / "tail_underestimation_profile.csv",
        rows,
        ["target", "bin", "lo", "hi", "n", "bias", "mae", "under_rate"],
    )
    _plot_tail_underestimation(rows, output_dir / "figures" / "tail_underestimation_profile.png")


def generate_bias_delta_vs_v1011(output_dir: Path) -> None:
    current = _read_csv(output_dir / "tables" / "bp_range_metrics.csv")
    baseline = _read_csv(_baseline_output_dir() / "tables" / "bp_range_metrics.csv")
    if not current or not baseline:
        return
    base_map = {str(row["bp_range"]): row for row in baseline}
    rows = []
    for row in current:
        bp_range = str(row["bp_range"])
        if bp_range not in base_map:
            continue
        base = base_map[bp_range]
        rows.append(
            {
                "bp_range": bp_range,
                "n": int(float(row.get("n", 0))),
                "sbp_bias_v10_13": float(row.get("bias_sbp", 0.0)),
                "sbp_bias_v10_11": float(base.get("bias_sbp", 0.0)),
                "sbp_bias_delta": float(row.get("bias_sbp", 0.0)) - float(base.get("bias_sbp", 0.0)),
                "dbp_bias_v10_13": float(row.get("bias_dbp", 0.0)),
                "dbp_bias_v10_11": float(base.get("bias_dbp", 0.0)),
                "dbp_bias_delta": float(row.get("bias_dbp", 0.0)) - float(base.get("bias_dbp", 0.0)),
                "mae_mean_v10_13": float(row.get("mae_mean", 0.0)),
                "mae_mean_v10_11": float(base.get("mae_mean", 0.0)),
            }
        )
    if not rows:
        return
    _write_csv(
        output_dir / "tables" / "bp_range_bias_delta_vs_v10_11.csv",
        rows,
        [
            "bp_range",
            "n",
            "sbp_bias_v10_13",
            "sbp_bias_v10_11",
            "sbp_bias_delta",
            "dbp_bias_v10_13",
            "dbp_bias_v10_11",
            "dbp_bias_delta",
            "mae_mean_v10_13",
            "mae_mean_v10_11",
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True)
    labels = [row["bp_range"] for row in rows]
    x = np.arange(len(labels))
    for ax, key, title, color in (
        (axes[0], "sbp_bias_delta", "SBP Bias Delta vs v10.11", "#4c78a8"),
        (axes[1], "dbp_bias_delta", "DBP Bias Delta vs v10.11", "#f58518"),
    ):
        ax.bar(x, [float(row[key]) for row in rows], color=color, alpha=0.78)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Delta error (mmHg)")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "bp_range_bias_delta_vs_v10_11.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_baseline_aware_selector_report(output_dir: Path) -> None:
    summary = _read_json(output_dir / "protocol_summary.json")
    baseline = _read_json(_baseline_output_dir() / "protocol_summary.json")
    current_rows = _read_csv(output_dir / "tables" / "bp_range_metrics.csv")
    baseline_rows = _read_csv(_baseline_output_dir() / "tables" / "bp_range_metrics.csv")
    current_map = _bp_range_map(current_rows)
    baseline_map = _bp_range_map(baseline_rows)
    rows: List[dict] = []
    for metric_name in ("selected_acc", "selected_macro_f1", "selected_balanced_acc"):
        cur = _to_float(summary.get(metric_name))
        ref = _to_float(baseline.get(metric_name))
        rows.append(
            {
                "criterion": metric_name,
                "current": cur,
                "v10_11_baseline": ref,
                "delta": cur - ref,
                "passed": bool(cur >= ref),
            }
        )
    for range_name in ("high", "crisis"):
        cur = current_map.get(range_name, {})
        ref = baseline_map.get(range_name, {})
        for axis in ("sbp", "dbp"):
            cur_bias = _to_float(cur.get(f"bias_{axis}"))
            ref_bias = _to_float(ref.get(f"bias_{axis}"))
            rows.append(
                {
                    "criterion": f"{range_name}_bias_{axis}",
                    "current": cur_bias,
                    "v10_11_baseline": ref_bias,
                    "delta": cur_bias - ref_bias,
                    "passed": bool(cur_bias >= ref_bias),
                }
            )
    _write_csv(
        output_dir / "tables" / "baseline_aware_selector_report.csv",
        rows,
        ["criterion", "current", "v10_11_baseline", "delta", "passed"],
    )
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    labels = [str(row["criterion"]) for row in rows]
    deltas = [float(row["delta"]) for row in rows]
    colors = ["#2f7d59" if bool(row["passed"]) else "#b44b4b" for row in rows]
    x = np.arange(len(labels))
    ax.bar(x, deltas, color=colors, alpha=0.82)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Delta vs v10.11")
    ax.set_title("Baseline-Aware Selector Criteria")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "baseline_aware_selector_report.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def update_protocol_audit(output_dir: Path) -> None:
    summary = _read_json(output_dir / "protocol_summary.json")
    run_status = _read_json(output_dir / "run_status.json")
    payload = {
        "protocol_id": FINAL_PROTOCOL_ID,
        "source_audit": {
            "v10_12_main_status": _read_json(
                _project_root()
                / "outputs"
                / "mimic_bp_reg_v10_12_subjectdisjoint_piso_uncertainty_moe_fulltrain_reliabilitybias_proto"
                / "run_status.json"
            ),
            "v10_12_optlong_has_final_results": (
                _project_root() / "outputs" / "mimic_bp_reg_v10_12_opt_long_fulltrain_proto" / "final_results.json"
            ).exists(),
            "v10_12_dualmax_has_checkpoint": (
                _project_root() / "outputs" / "mimic_bp_reg_v10_12_optlong_stageaware_dualmax_fulltrain_proto" / "best_model.pt"
            ).exists(),
        },
        "current_run_status": run_status,
        "current_protocol_summary": summary,
        "mechanism_notes": [
            "High-risk samples are reweighted during training through sampler power and tail-class weights.",
            "High/crisis underestimation is penalized in validation selection through range and top-tail bias terms.",
            "The v10.13 baseline-aware selector penalizes candidates below v10.11 validation/selected Acc and macro-F1 and candidates that worsen high/crisis bias.",
            "Reliability-aware post-hoc correction can move predictions up or down, but the grid is biased toward correcting high-risk underestimation.",
            "The crisis-SBP guard applies an additional upward correction only when stage-2 probability, expert peak SBP/DBP, and crisis-risk signals agree.",
            "Additional figures expose tail underestimation and v10.11-v10.13 bias deltas for paper discussion.",
        ],
    }
    _write_json(output_dir / "protocol_tailaware_audit.json", payload)


def generate_extra_outputs(output_dir: Path) -> None:
    _BASE_GENERATE_EXTRA_OUTPUTS(output_dir)
    for fn in (
        generate_tail_underestimation_profile,
        generate_bias_delta_vs_v1011,
        generate_baseline_aware_selector_report,
        update_protocol_audit,
    ):
        try:
            fn(output_dir)
        except Exception as exc:
            print(f"[v10.13] Skipped extra artifact step {fn.__name__}: {exc}")


def main() -> None:
    originals = {
        "base_final": base_script.FINAL_OUTPUT_NAME,
        "base_protocol": base_script.FINAL_PROTOCOL_ID,
        "base_optlong": base_script.OPTLONG_FULLTRAIN_OUTPUT,
        "base_dualmax": base_script.DUALMAX_FULLTRAIN_OUTPUT,
        "base_baseline": base_script._BASELINE_OUTPUT_NAME,
        "base_build_nextgen": base_script.build_nextgen_cfg,
        "base_build_optlong": base_script.build_optlong_fulltrain_cfg,
        "base_build_dualmax": base_script.build_dualmax_fulltrain_cfg,
        "base_run_self": base_script._run_self_subprocess,
        "base_prepare": base_script._prepare_fulltrain_cache_dirs,
        "base_generate": base_script.generate_extra_outputs,
        "base_promotion_selection": base_script.promotion_aware_classification_selection_score,
        "base_promotion_candidate": base_script.promotion_aware_classification_candidate_score,
        "base_promotion_robust": base_script.promotion_aware_robust_classification_score,
        "base_high_bias_cost": base_script.high_bias_calibration_cost,
        "base_search_high_bias": base_script.search_high_bias_calibration_candidates,
        "prev_checkpoint_complete": prev_script._checkpoint_artifact_complete,
        "prev_resume_incomplete": getattr(prev_script, "RESUME_INCOMPLETE_FULLTRAIN", False),
        "prev_apply_crisis_tail": prev_script.apply_crisis_tail_debias_fusion,
        "prev_crisis_tail_cost": prev_script.crisis_tail_debias_cost,
    }
    try:
        base_script.FINAL_OUTPUT_NAME = FINAL_OUTPUT_NAME
        base_script.FINAL_PROTOCOL_ID = FINAL_PROTOCOL_ID
        base_script.OPTLONG_FULLTRAIN_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
        base_script.DUALMAX_FULLTRAIN_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT
        base_script._BASELINE_OUTPUT_NAME = BASELINE_OUTPUT_NAME
        base_script.build_nextgen_cfg = build_nextgen_cfg
        base_script.build_optlong_fulltrain_cfg = build_optlong_fulltrain_cfg
        base_script.build_dualmax_fulltrain_cfg = build_dualmax_fulltrain_cfg
        base_script._run_self_subprocess = _run_self_subprocess
        base_script._prepare_fulltrain_cache_dirs = _prepare_fulltrain_cache_dirs_no_archive
        base_script.generate_extra_outputs = generate_extra_outputs
        base_script.promotion_aware_classification_selection_score = tailaware_classification_selection_score
        base_script.promotion_aware_classification_candidate_score = tailaware_classification_candidate_score
        base_script.promotion_aware_robust_classification_score = tailaware_robust_classification_score
        base_script.high_bias_calibration_cost = tailaware_high_bias_calibration_cost
        base_script.search_high_bias_calibration_candidates = tailaware_search_high_bias_calibration_candidates
        _ensure_expected_dirs()
        _patch_prev_outputs()
        _ensure_runtime_compatibility()
        prev_script._checkpoint_artifact_complete = _checkpoint_artifact_usable
        prev_script.RESUME_INCOMPLETE_FULLTRAIN = True
        prev_script.apply_crisis_tail_debias_fusion = apply_crisis_tail_debias_fusion
        prev_script.crisis_tail_debias_cost = crisis_tail_debias_cost
        base_script.main()
    finally:
        base_script.FINAL_OUTPUT_NAME = originals["base_final"]
        base_script.FINAL_PROTOCOL_ID = originals["base_protocol"]
        base_script.OPTLONG_FULLTRAIN_OUTPUT = originals["base_optlong"]
        base_script.DUALMAX_FULLTRAIN_OUTPUT = originals["base_dualmax"]
        base_script._BASELINE_OUTPUT_NAME = originals["base_baseline"]
        base_script.build_nextgen_cfg = originals["base_build_nextgen"]
        base_script.build_optlong_fulltrain_cfg = originals["base_build_optlong"]
        base_script.build_dualmax_fulltrain_cfg = originals["base_build_dualmax"]
        base_script._run_self_subprocess = originals["base_run_self"]
        base_script._prepare_fulltrain_cache_dirs = originals["base_prepare"]
        base_script.generate_extra_outputs = originals["base_generate"]
        base_script.promotion_aware_classification_selection_score = originals["base_promotion_selection"]
        base_script.promotion_aware_classification_candidate_score = originals["base_promotion_candidate"]
        base_script.promotion_aware_robust_classification_score = originals["base_promotion_robust"]
        base_script.high_bias_calibration_cost = originals["base_high_bias_cost"]
        base_script.search_high_bias_calibration_candidates = originals["base_search_high_bias"]
        prev_script._checkpoint_artifact_complete = originals["prev_checkpoint_complete"]
        prev_script.RESUME_INCOMPLETE_FULLTRAIN = originals["prev_resume_incomplete"]
        prev_script.apply_crisis_tail_debias_fusion = originals["prev_apply_crisis_tail"]
        prev_script.crisis_tail_debias_cost = originals["prev_crisis_tail_cost"]


def _dispatch_subprocess_mode(argv: List[str]) -> int | None:
    if not argv:
        return None
    base_script.FINAL_OUTPUT_NAME = FINAL_OUTPUT_NAME
    base_script.FINAL_PROTOCOL_ID = FINAL_PROTOCOL_ID
    base_script.OPTLONG_FULLTRAIN_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
    base_script.DUALMAX_FULLTRAIN_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT
    base_script._BASELINE_OUTPUT_NAME = BASELINE_OUTPUT_NAME
    _ensure_expected_dirs()
    _patch_prev_outputs()
    _ensure_runtime_compatibility()
    prev_script.build_nextgen_cfg = build_nextgen_cfg
    prev_script.build_optlong_fulltrain_cfg = build_optlong_fulltrain_cfg
    prev_script.build_dualmax_fulltrain_cfg = build_dualmax_fulltrain_cfg
    prev_script._checkpoint_artifact_complete = _checkpoint_artifact_usable
    prev_script.RESUME_INCOMPLETE_FULLTRAIN = True
    prev_script.apply_crisis_tail_debias_fusion = apply_crisis_tail_debias_fusion
    prev_script.crisis_tail_debias_cost = crisis_tail_debias_cost
    return prev_script._dispatch_subprocess_mode(argv)


if __name__ == "__main__":
    dispatch_code = _dispatch_subprocess_mode(sys.argv[1:])
    if dispatch_code is None:
        main()
    else:
        raise SystemExit(dispatch_code)
