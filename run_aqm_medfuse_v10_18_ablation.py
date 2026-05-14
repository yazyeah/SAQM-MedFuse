from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

import train_aqm_medfuse_mimic_bp_reg_v10_18_subjectdisjoint_baselineguard_crisisrepair_protocol as opt_script


VARIANTS = (
    "full",
    "no_tail_reweighting",
    "no_reliability_bias_calibration",
    "no_crisis_tail_debias",
    "no_safety_evidential_fusion",
    "no_accf1_targeted_head",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent


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


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _main_output_dir() -> Path:
    return _project_root() / "outputs" / opt_script.FINAL_OUTPUT_NAME


def _variant_output_name(variant: str) -> str:
    if variant == "full":
        return opt_script.FINAL_OUTPUT_NAME
    return f"mimic_bp_ablation_v10_18_{variant}_proto"


def _variant_output_dir(variant: str) -> Path:
    return _project_root() / "outputs" / _variant_output_name(variant)


def _bp_range_value(output: Path, label: str, key: str) -> float | str:
    rows = _read_csv_rows(output / "tables" / "bp_range_metrics.csv")
    for row in rows:
        if str(row.get("bp_range", "")).lower() == label:
            try:
                return float(row.get(key, ""))
            except (TypeError, ValueError):
                return row.get(key, "")
    return ""


def _identity_high_bias_search(calib_out, calib_cls_prob, query_out, query_cls_prob, cfg):
    stage = opt_script.v17.v16.meta_script.stage_script
    conformal = stage.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    bp_rows = stage.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"])
    range_map = {str(row["bp_range"]): row for row in bp_rows}
    row = {
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
        "score": 0.0,
        "clinical_under_penalty": 0.0,
        "tail_bias_penalty": 0.0,
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


def _identity_crisis_tail_search(calib_out, calib_cls_prob, calib_reg_inputs, query_out, query_cls_prob, query_reg_inputs, cfg):
    stage = opt_script.v17.v16.meta_script.stage_script
    conformal = stage.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    bp_rows = stage.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"])
    range_map = {str(row["bp_range"]): row for row in bp_rows}
    row = {
        "candidate": "identity",
        "bundle_key": "identity",
        "high_threshold": 0.0,
        "crisis_threshold": 0.0,
        "gamma": 1.0,
        "sbp_quantile": 0.0,
        "dbp_quantile": 0.0,
        "crisis_gain": 1.0,
        "sbp_margin": 0.0,
        "dbp_margin": 0.0,
        "uncertainty_gain": 0.0,
        "model_scale": 0.0,
        "expert_gain": 0.0,
        "score": 0.0,
        "clinical_under_penalty": 0.0,
        "tail_bias_penalty": 0.0,
        "high_bias_sbp": float(range_map.get("high", {}).get("bias_sbp", 0.0)),
        "high_bias_dbp": float(range_map.get("high", {}).get("bias_dbp", 0.0)),
        "crisis_bias_sbp": float(range_map.get("crisis", {}).get("bias_sbp", 0.0)),
        "crisis_bias_dbp": float(range_map.get("crisis", {}).get("bias_dbp", 0.0)),
        "fusion_gate_mean": 0.0,
        "shift_mean_sbp": 0.0,
        "shift_mean_dbp": 0.0,
        "activation_rate": 0.0,
        **query_out["metrics_reg"],
        **conformal,
    }
    return row, [row]


def _identity_safety_fusion_search(query_cls_prob, query_reg_out, cfg):
    stage = opt_script.v17.v16.meta_script.stage_script
    y_true = np.asarray(query_reg_out["y_true_cls"], dtype=np.int64)
    prob = np.asarray(query_cls_prob, dtype=np.float32)
    metrics = stage.risk_classification_metrics(
        y_true,
        prob.argmax(axis=1).astype(np.int64),
        prob,
        cfg,
        prefix="selected_val",
    )
    row = {
        "candidate": "identity",
        "fusion_scale": 0.0,
        "fusion_beta": 0.0,
        "fusion_disagree_gain": 0.0,
        "fusion_high_gain": 0.0,
        "fusion_crisis_gain": 0.0,
        "reliability_floor": 0.0,
        "stage1_bias": 0.0,
        "stage2_bias": 0.0,
        "score": 0.0,
        "mean_weight": 0.0,
        "high_weight_mean": 0.0,
        "crisis_weight_mean": 0.0,
        "acc": float(metrics.get("selected_val_acc", 0.0)),
        "macro_f1": float(metrics.get("selected_val_macro_f1", 0.0)),
        "balanced_acc": float(metrics.get("selected_val_balanced_acc", 0.0)),
        **metrics,
    }
    return row, [row]


def _mutate_no_tail(cfg):
    cfg.REG_USE_WEIGHTED_SAMPLER = False
    cfg.REG_SAMPLER_POWER = 0.0
    cfg.TAIL_CLASS_WEIGHTS = (0.0, 0.0, 0.0, 0.0)
    cfg.LAMBDA_TAIL = 0.0
    cfg.LAMBDA_CRISIS_TAIL = 0.0
    cfg.VAL_SCORE_HIGH_BIAS_WEIGHT = 0.0
    cfg.VAL_SCORE_CRISIS_BIAS_WEIGHT = 0.0
    cfg.VAL_SCORE_TAIL_TOP10_BIAS_WEIGHT = 0.0
    cfg.VAL_SCORE_TAIL_TOP5_BIAS_WEIGHT = 0.0
    return cfg


def _mutate_no_accf1_head(cfg):
    cfg.HEAD_SELECTION_MODE = "score"
    cfg.HEAD_SELECTION_RANK_MODE = "score"
    cfg.HEAD_CLEAN_ACC_WEIGHT = 0.0
    cfg.HEAD_CLEAN_F1_WEIGHT = 0.0
    cfg.HEAD_CLEAN_BALANCED_WEIGHT = 0.0
    cfg.HEAD_CLEAN_ROBUST_WEIGHT = 0.0
    cfg.HEAD_CLEAN_STAGE2_WEIGHT = 0.0
    cfg.HEAD_ELEVATED_REPEAT = 1
    cfg.HEAD_STAGE1_REPEAT = 1
    cfg.HEAD_STAGE2_REPEAT = 1
    cfg.HEAD_TARGET_RARE_MIN_WEIGHT = 0.0
    cfg.HEAD_TARGET_STAGE2_WEIGHT = 0.0
    cfg.HEAD_TARGET_GAP_WEIGHT = 0.0
    cfg.HEAD_TARGET_ROBUST_GAP_WEIGHT = 0.0
    cfg.HEAD_ROBUST_MIN_WEIGHT = 0.0
    return cfg


def _set_variant_names(variant: str) -> dict:
    originals = {
        "final": opt_script.FINAL_OUTPUT_NAME,
        "protocol": opt_script.FINAL_PROTOCOL_ID,
        "optlong": opt_script.OPTLONG_FULLTRAIN_OUTPUT,
        "dualmax": opt_script.DUALMAX_FULLTRAIN_OUTPUT,
    }
    opt_script.FINAL_OUTPUT_NAME = _variant_output_name(variant)
    opt_script.FINAL_PROTOCOL_ID = f"v10.18_ablation_{variant}"
    return originals


def _restore_variant_names(originals: dict) -> None:
    opt_script.FINAL_OUTPUT_NAME = originals["final"]
    opt_script.FINAL_PROTOCOL_ID = originals["protocol"]
    opt_script.OPTLONG_FULLTRAIN_OUTPUT = originals["optlong"]
    opt_script.DUALMAX_FULLTRAIN_OUTPUT = originals["dualmax"]


def _seed_feature_head_cache(variant: str) -> None:
    if variant == "no_accf1_targeted_head":
        return
    src_dir = _main_output_dir()
    dst_dir = _variant_output_dir(variant)
    for name in ("feature_head_best.pt", "feature_head_cache_meta.json"):
        src = src_dir / name
        if src.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / name)


def run_variant(variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    if variant == "full" and _main_output_dir().exists():
        return _main_output_dir()

    name_originals = _set_variant_names(variant)
    original_build_nextgen: Callable = opt_script.build_nextgen_cfg
    original_high_bias = opt_script.v17.tailaware_search_high_bias_calibration_candidates_pareto
    original_crisis = opt_script.v17.search_crisis_tail_debias_candidates_pareto
    original_safety = opt_script.v17.guarded_search_safety_class_fusion_pareto

    def build_nextgen_cfg():
        cfg = original_build_nextgen()
        cfg.PROTOCOL_ID = opt_script.FINAL_PROTOCOL_ID
        cfg.OUTPUT_NAME = opt_script.FINAL_OUTPUT_NAME
        cfg.PROTOCOL_NAME = f"v10.18 ablation: {variant}"
        cfg.FINAL_OUTPUT_NAME = opt_script.FINAL_OUTPUT_NAME
        cfg.FINAL_PROTOCOL_ID = opt_script.FINAL_PROTOCOL_ID
        if variant == "no_tail_reweighting":
            cfg = _mutate_no_tail(cfg)
        if variant == "no_accf1_targeted_head":
            cfg = _mutate_no_accf1_head(cfg)
        return cfg

    try:
        _seed_feature_head_cache(variant)
        opt_script.build_nextgen_cfg = build_nextgen_cfg
        if variant == "no_reliability_bias_calibration":
            opt_script.v17.tailaware_search_high_bias_calibration_candidates_pareto = _identity_high_bias_search
        if variant == "no_crisis_tail_debias":
            opt_script.v17.search_crisis_tail_debias_candidates_pareto = _identity_crisis_tail_search
        if variant == "no_safety_evidential_fusion":
            opt_script.v17.guarded_search_safety_class_fusion_pareto = _identity_safety_fusion_search
        opt_script.main()
        return _variant_output_dir(variant)
    finally:
        opt_script.build_nextgen_cfg = original_build_nextgen
        opt_script.v17.tailaware_search_high_bias_calibration_candidates_pareto = original_high_bias
        opt_script.v17.search_crisis_tail_debias_candidates_pareto = original_crisis
        opt_script.v17.guarded_search_safety_class_fusion_pareto = original_safety
        _restore_variant_names(name_originals)


def summarize_variants(variants: List[str]) -> Path:
    rows = []
    for variant in variants:
        output = _variant_output_dir(variant)
        if variant == "full":
            output = _main_output_dir()
        summary = _read_json(output / "protocol_summary.json")
        paper = _read_json(output / "paper_metrics.json")
        aami = paper.get("aami_like", {})
        bhs = paper.get("bhs_like", {})
        rows.append(
            {
                "variant": variant,
                "module_removed": {
                    "full": "none",
                    "no_tail_reweighting": "tail-aware training loss/sampler weights",
                    "no_reliability_bias_calibration": "high-range reliability bias calibration",
                    "no_crisis_tail_debias": "crisis-tail debias/fusion search",
                    "no_safety_evidential_fusion": "safety-aware classification fusion",
                    "no_accf1_targeted_head": "accuracy/F1-targeted feature-head ranking",
                }.get(variant, ""),
                "output": str(output),
                "selected_acc": summary.get("selected_acc", ""),
                "selected_macro_f1": summary.get("selected_macro_f1", ""),
                "selected_balanced_acc": summary.get("selected_balanced_acc", ""),
                "selected_mae_mean": summary.get("selected_mae_mean", ""),
                "sbp_mean_error": aami.get("sbp_mean_error", ""),
                "sbp_sd_error": aami.get("sbp_sd_error", ""),
                "dbp_mean_error": aami.get("dbp_mean_error", ""),
                "dbp_sd_error": aami.get("dbp_sd_error", ""),
                "sbp_bhs_grade": bhs.get("sbp_grade", ""),
                "dbp_bhs_grade": bhs.get("dbp_grade", ""),
                "high_bias_sbp": _bp_range_value(output, "high", "bias_sbp"),
                "high_bias_dbp": _bp_range_value(output, "high", "bias_dbp"),
                "crisis_n": _bp_range_value(output, "crisis", "n"),
                "crisis_bias_sbp": _bp_range_value(output, "crisis", "bias_sbp"),
                "crisis_bias_dbp": _bp_range_value(output, "crisis", "bias_dbp"),
                "selected_regression_candidate": summary.get("selected_regression_candidate", ""),
                "selected_classification_candidate": summary.get("selected_classification_candidate", ""),
                "selected_tail_correction_candidate": summary.get("selected_tail_correction_candidate", ""),
            }
        )
    path = _project_root() / "outputs" / "mimic_bp_ablation_v10_18_summary.csv"
    _write_csv(path, rows)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v10.18 AQM-MedFuse ablation protocols.")
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    for variant in variants:
        print(f"[v10.18 ablation] Running {variant}")
        run_variant(variant)
    summary_path = summarize_variants(variants)
    print(f"[v10.18 ablation] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
