from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np

import train_aqm_medfuse_mimic_bp_reg_v10_18_subjectdisjoint_baselineguard_crisisrepair_protocol as opt_script
import run_aqm_medfuse_v10_18_ablation as strict_ablation


CORE_VARIANTS = (
    "full",
    "no_tail_reweighting",
    "no_accf1_targeted_head",
    "no_reliability_bias_calibration",
    "no_safety_evidential_fusion",
)

PAPER_VARIANTS = (
    "full",
    "no_tail_reweighting",
    "no_accf1_targeted_head",
    "no_guided_policy_search",
    "no_classification_arbiter",
    "no_baseline_guarded_selector",
    "no_reliability_bias_calibration",
    "no_crisis_guard_floor",
    "no_crisis_tail_debias",
    "no_safety_evidential_fusion",
)

MAJOR_VARIANTS = (
    "major_full",
    "no_quality_sparse_fusion",
    "no_uncertainty_credibility",
    "no_anchor_decision_refinement",
    "no_tail_safety_calibration",
    "no_personalized_conformal",
)

ALL_VARIANTS = tuple(dict.fromkeys(PAPER_VARIANTS + MAJOR_VARIANTS))

VARIANT_TRAINED_HEADS = {
    "full",
    "major_full",
    "no_tail_reweighting",
    "no_accf1_targeted_head",
    "no_quality_sparse_fusion",
}

REQUIRED_OUTPUT_FILES = (
    "protocol_summary.json",
    "final_results.json",
    "paper_metrics.json",
    "selected_strategy.json",
    "tables/bp_range_metrics.csv",
    "runtime_metrics.json",
)

MODULE_REMOVED = {
    "full": "none",
    "major_full": "none (major-module equal-budget reference)",
    "no_tail_reweighting": "tail-aware train/head reweighting",
    "no_accf1_targeted_head": "accuracy/F1-targeted feature-head ranking",
    "no_guided_policy_search": "guided class-prior policy search",
    "no_classification_arbiter": "classification arbiter search",
    "no_baseline_guarded_selector": "v10.18 baseline-aware selection objectives",
    "no_reliability_bias_calibration": "high-range reliability bias calibration",
    "no_crisis_guard_floor": "crisis hard-floor and absolute guard",
    "no_crisis_tail_debias": "crisis-tail debias/fusion search",
    "no_safety_evidential_fusion": "safety-aware classification fusion",
    "no_quality_sparse_fusion": "QESF: quality-aware encoding and sparse fusion robustness pathway",
    "no_uncertainty_credibility": "URC: uncertainty-aware regression router and credibility aggregation",
    "no_anchor_decision_refinement": "ADSF: anchor-guided decision refinement",
    "no_tail_safety_calibration": "STC: safety-aware tail calibration and crisis repair",
    "no_personalized_conformal": "SAP+Conformal: subject-adaptive reliability constraints and conformal selection",
}

RUN_TAG = ""


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _summary_dir() -> Path:
    return _project_root() / "outputs" / "v10_18_paper_summary"


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


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
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


def _variant_output_name(variant: str) -> str:
    tag = f"_{RUN_TAG}" if RUN_TAG else ""
    return f"mimic_bp_ablation_v10_18_litetrain_{variant}{tag}_proto"


def _variant_output_dir(variant: str) -> Path:
    return _project_root() / "outputs" / _variant_output_name(variant)


def _is_variant_complete(output: Path) -> bool:
    return output.exists() and all((output / relative).exists() for relative in REQUIRED_OUTPUT_FILES)


def _range_row(output: Path, label: str) -> dict:
    for row in _read_csv_rows(output / "tables" / "bp_range_metrics.csv"):
        if str(row.get("bp_range", "")).lower() == label:
            return row
    return {}


def _to_float(value, default=""):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _csv_metric_value(output: Path, relative: str, criteria: dict, value_key: str):
    rows = _read_csv_rows(output / relative)
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items()):
            return _to_float(row.get(value_key, ""))
    return ""


def _csv_mean(output: Path, relative: str, key: str):
    values = []
    for row in _read_csv_rows(output / relative):
        value = _to_float(row.get(key, ""), default=None)
        if value is not None:
            values.append(value)
    if not values:
        return ""
    return sum(values) / len(values)


def _metric_from_results(results: dict, *keys: str):
    for section in ("test_selected", "test", "validation_selected", "validation"):
        metrics = results.get(section, {})
        if not isinstance(metrics, dict):
            continue
        for key in keys:
            if key in metrics:
                return metrics.get(key)
    return ""


def _candidate_totals(cfg) -> dict:
    totals = {}
    try:
        totals["classification_arbiter_candidates"] = int(opt_script.v17.v16._classification_arbiter_total(cfg))
    except Exception:
        totals["classification_arbiter_candidates"] = ""
    try:
        totals["high_bias_candidates"] = int(opt_script.v17.v16._high_bias_total(cfg))
    except Exception:
        totals["high_bias_candidates"] = ""
    try:
        totals["crisis_tail_candidates"] = int(opt_script.v17.v16._crisis_total(cfg))
    except Exception:
        totals["crisis_tail_candidates"] = ""
    return totals


def _apply_lite_search_profile(cfg, args: argparse.Namespace):
    cfg.PROTOCOL_NAME = f"{getattr(cfg, 'PROTOCOL_NAME', 'v10.18')} lite-train ablation"
    cfg.HEAD_EPOCHS = int(args.head_epochs)
    cfg.HEAD_PATIENCE = int(args.head_patience)
    cfg.HEAD_MIN_EPOCHS = int(args.head_min_epochs)

    # Keep each trainable ablation faithful to the final pipeline, but cap the
    # combinatorial selectors so the run is paper-usable without week-long search.
    cfg.GUIDED_POLICY_GAMMAS = (0.90, 1.00, 1.10)
    cfg.GUIDED_POLICY_ELEVATED_WEIGHTS = (0.96, 1.00, 1.16)
    cfg.GUIDED_POLICY_STAGE1_WEIGHTS = (0.98, 1.00, 1.18)
    cfg.GUIDED_POLICY_STAGE2_WEIGHTS = (0.96, 1.00, 1.25)
    cfg.GUIDED_POLICY_BATCH_SIZE = 4096

    cfg.CLS_ARBITER_SCALES = (0.0, 0.45, 0.85)
    cfg.CLS_ARBITER_BETAS = (1.0, 1.8)
    cfg.CLS_ARBITER_FLOORS = (0.0, 0.06)
    cfg.CLS_ARBITER_AGREE_SHRINKS = (0.0, 0.12)
    cfg.CLS_ARBITER_STAGE1_BIASES = (0.0, 0.04)
    cfg.CLS_ARBITER_STAGE2_BIASES = (0.0, 0.08)

    cfg.SAFETY_EVIDENTIAL_SCALES = (0.0, 0.04, 0.12)
    cfg.SAFETY_EVIDENTIAL_BETAS = (0.7, 1.1)
    cfg.SAFETY_EVIDENTIAL_DISAGREE_GAINS = (0.0, 0.7)
    cfg.SAFETY_EVIDENTIAL_HIGH_GAINS = (0.0, 0.8)
    cfg.SAFETY_EVIDENTIAL_CRISIS_GAINS = (0.0, 0.7)
    cfg.SAFETY_EVIDENTIAL_RELIABILITY_FLOORS = (0.0, 0.08)
    cfg.SAFETY_EVIDENTIAL_STAGE1_BIASES = (0.0, 0.04)
    cfg.SAFETY_EVIDENTIAL_STAGE2_BIASES = (0.0, 0.08)

    cfg.RELIABILITY_BIAS_SCALES = (0.0, 0.16, 0.38)
    cfg.RELIABILITY_BIAS_BETAS = (1.0, 1.55)
    cfg.RELIABILITY_BIAS_RELIABILITY_FLOORS = (0.0, 0.10)
    cfg.RELIABILITY_BIAS_DISAGREE_GAINS = (0.0, 0.75)
    cfg.RELIABILITY_BIAS_HIGH_GAINS = (0.0, 0.55)
    cfg.RELIABILITY_BIAS_CRISIS_GAINS = (0.0, 1.10)
    cfg.RELIABILITY_BIAS_NEGATIVE_FRACS = (0.0, 0.12)
    cfg.RELIABILITY_BIAS_HIGH_THRESHOLDS = (0.14, 0.22)
    cfg.RELIABILITY_BIAS_CRISIS_THRESHOLDS = (0.10, 0.18)
    cfg.RELIABILITY_BIAS_HIGH_FLOOR_SBP = (0.0, 4.0)
    cfg.RELIABILITY_BIAS_CRISIS_FLOOR_SBP = (0.0, 6.0)

    cfg.CRISIS_TAIL_FUSION_HIGH_THRESHOLDS = (0.14, 0.38)
    cfg.CRISIS_TAIL_FUSION_CRISIS_THRESHOLDS = (0.10, 0.30)
    cfg.CRISIS_TAIL_FUSION_GAMMAS = (1.0,)
    cfg.CRISIS_TAIL_FUSION_SBP_QUANTILES = (0.92, 0.96)
    cfg.CRISIS_TAIL_FUSION_DBP_QUANTILES = (0.84, 0.92)
    cfg.CRISIS_TAIL_FUSION_CRISIS_GAINS = (0.55, 1.10)
    cfg.CRISIS_TAIL_FUSION_SBP_MARGINS = (0.0, 6.0)
    cfg.CRISIS_TAIL_FUSION_DBP_MARGINS = (0.0, 3.0)
    cfg.CRISIS_TAIL_FUSION_UNCERTAINTY_GAINS = (0.0,)
    cfg.CRISIS_TAIL_FUSION_MODEL_SCALES = (0.85,)
    cfg.CRISIS_TAIL_FUSION_EXPERT_GAINS = (0.70,)

    cfg.VECTORFAST_EXACT_EXHAUSTIVE = False
    cfg.VECTORFAST_HIGH_BIAS_KEEP_ROWS = 512
    cfg.VECTORFAST_CRISIS_KEEP_ROWS = 512
    cfg.VECTORFAST_HIGH_BIAS_BATCH_SIZE = 1024
    cfg.VECTORFAST_CRISIS_BATCH_SIZE = 1024
    cfg.VECTORFAST_CACHE_FLUSH_EVERY = 128
    return cfg


def _mutate_no_guided_policy_search(cfg):
    cfg.GUIDED_POLICY_GAMMAS = (1.0,)
    cfg.GUIDED_POLICY_ELEVATED_WEIGHTS = (1.0,)
    cfg.GUIDED_POLICY_STAGE1_WEIGHTS = (1.0,)
    cfg.GUIDED_POLICY_STAGE2_WEIGHTS = (1.0,)
    return cfg


def _mutate_no_classification_arbiter(cfg):
    cfg.CLS_ARBITER_SCALES = (0.0,)
    cfg.CLS_ARBITER_BETAS = (1.0,)
    cfg.CLS_ARBITER_FLOORS = (0.0,)
    cfg.CLS_ARBITER_AGREE_SHRINKS = (0.0,)
    cfg.CLS_ARBITER_STAGE1_BIASES = (0.0,)
    cfg.CLS_ARBITER_STAGE2_BIASES = (0.0,)
    return cfg


def _mutate_no_crisis_guard_floor(cfg):
    cfg.CRISIS_TAIL_HARD_FLOOR_SBP = 0.0
    cfg.CRISIS_TAIL_HARD_FLOOR_DBP = 0.0
    cfg.CRISIS_SBP_GUARD_TRIGGER = 1.0e9
    cfg.CRISIS_SBP_GUARD_MIN_EXPERT_SBP = 1.0e9
    cfg.CRISIS_SBP_GUARD_ABSOLUTE_FLOOR = 0.0
    cfg.CRISIS_DBP_GUARD_ABSOLUTE_FLOOR = 0.0
    cfg.CRISIS_SBP_GUARD_GAIN = 0.0
    cfg.CRISIS_DBP_GUARD_GAIN = 0.0
    cfg.BASELINE_CRISIS_SBP_TARGET = 1.0e9
    cfg.BASELINE_CRISIS_DBP_TARGET = 1.0e9
    return cfg


def _mutate_no_tail_lite_head(cfg):
    """Remove tail emphasis that is active when the backbone is frozen."""
    cfg = strict_ablation._mutate_no_tail(cfg)
    cfg.HEAD_ELEVATED_REPEAT = 1
    cfg.HEAD_STAGE1_REPEAT = 1
    cfg.HEAD_STAGE2_REPEAT = 1
    cfg.HEAD_TARGET_RARE_MIN_WEIGHT = 0.0
    cfg.HEAD_TARGET_STAGE2_WEIGHT = 0.0
    cfg.HEAD_TARGET_GAP_WEIGHT = 0.0
    cfg.HEAD_TARGET_ROBUST_GAP_WEIGHT = 0.0
    cfg.HEAD_ROBUST_MIN_WEIGHT = 0.0
    cfg.HEAD_CLEAN_STAGE2_WEIGHT = 0.0
    return cfg


def _mutate_no_baseline_guarded_selector(cfg):
    cfg.V10_18_BASELINE_GUARDED_SELECTOR_ENABLED = False
    return cfg


def _mutate_no_quality_sparse_fusion(cfg):
    """Remove the trainable robustness pathway used as the frozen-backbone QESF proxy."""
    cfg.HEAD_MISSING_ECG = 0.0
    cfg.HEAD_MISSING_PPG = 0.0
    cfg.HEAD_NOISE_STD = 0.0
    cfg.HEAD_ROBUST_NOISE_WEIGHT = 0.0
    cfg.HEAD_ROBUST_ECG_WEIGHT = 0.0
    cfg.HEAD_ROBUST_PPG_WEIGHT = 0.0
    cfg.HEAD_ROBUST_MIN_WEIGHT = 0.0
    cfg.HEAD_CLEAN_ROBUST_WEIGHT = 0.0
    cfg.CLS_SCORE_NOISE_WEIGHT = 0.0
    cfg.CLS_SCORE_ECG_WEIGHT = 0.0
    cfg.CLS_SCORE_PPG_WEIGHT = 0.0
    cfg.CLS_SCORE_MINROBUST_WEIGHT = 0.0
    return cfg


def _mutate_no_uncertainty_credibility(cfg):
    cfg.ENABLE_REGRESSION_ROUTER = False
    cfg.REG_ROUTER_BLEND_SCALES = (0.0,)
    cfg.REG_ROUTER_TEMPS = (1.0,)
    cfg.REG_ROUTER_GAMMAS = (1.0,)
    cfg.REG_ROUTER_FLOORS = (0.0,)
    cfg.RELIABILITY_BIAS_RELIABILITY_FLOORS = (0.0,)
    cfg.RELIABILITY_BIAS_DISAGREE_GAINS = (0.0,)
    cfg.RELIABILITY_BIAS_CRISIS_GAINS = (0.0,)
    cfg.CRISIS_TAIL_FUSION_UNCERTAINTY_GAINS = (0.0,)
    return cfg


def _mutate_no_anchor_decision_refinement(cfg):
    cfg.ENABLE_CLASSIFICATION_ARBITER = False
    cfg = _mutate_no_guided_policy_search(cfg)
    cfg = _mutate_no_classification_arbiter(cfg)
    cfg = _mutate_no_baseline_guarded_selector(cfg)
    return cfg


def _mutate_no_tail_safety_calibration(cfg):
    cfg = _mutate_no_crisis_guard_floor(cfg)
    cfg.ENABLE_RISK_GUARD = False
    cfg.ENABLE_HIGH_BIAS_CALIBRATOR = False
    cfg.ENABLE_CRISIS_TAIL_FUSION = False
    cfg.ENABLE_SAFETY_CLASS_FUSION = False
    return cfg


def _mutate_no_personalized_conformal(cfg):
    cfg.USE_SUBJECT_CALIBRATION = False
    cfg.MODEL_SELECTION_USE_CALIBRATED_VAL = False
    cfg.SUBJECT_CALIBRATION_MODE = "identity"
    cfg.CALIBRATION_SHRINKAGE = 1.0e9
    cfg.REGRESSION_PARETO_COVERAGE_GAP_MAX = 1.0e9
    cfg.RISK_GUARD_MAX_COVERAGE_GAP_DELTA = 1.0e9
    cfg.HIGH_BIAS_CAL_MAX_COVERAGE_GAP_DELTA = 1.0e9
    cfg.CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA = 1.0e9
    cfg.RELIABILITY_BIAS_MAX_COVERAGE_GAP_DELTA = 1.0e9
    cfg.CRISIS_REPAIR_MIN_ACTIVATION_RATE = 0.0
    return cfg


def _identity_subject_calibration_state(calib_out: dict, cfg, n_shots: int | None = None) -> dict:
    n_rows = int(np.asarray(calib_out.get("y_true_reg", [])).shape[0])
    return {
        "mode": "identity_disabled_for_ablation",
        "global_scale": np.ones(2, dtype=np.float32),
        "global_offset": np.zeros(2, dtype=np.float32),
        "subject_scale": {},
        "subject_offset": {},
        "n_subjects": 0,
        "shrinkage": 0.0,
        "n_shots": None if n_shots is None else int(n_shots),
        "n_rows_used": n_rows,
    }


def _apply_identity_subject_calibration(out: dict, calib_state: dict, cfg) -> dict:
    y_pred = np.asarray(out["y_pred_reg"], dtype=np.float32)
    out_identity = dict(out)
    out_identity["calibration_scale"] = np.ones_like(y_pred, dtype=np.float32)
    out_identity["calibration_offset"] = np.zeros_like(y_pred, dtype=np.float32)
    return out_identity


def _set_variant_names(variant: str) -> dict:
    originals = {
        "final": opt_script.FINAL_OUTPUT_NAME,
        "protocol": opt_script.FINAL_PROTOCOL_ID,
        "optlong": opt_script.OPTLONG_FULLTRAIN_OUTPUT,
        "dualmax": opt_script.DUALMAX_FULLTRAIN_OUTPUT,
    }
    opt_script.FINAL_OUTPUT_NAME = _variant_output_name(variant)
    opt_script.FINAL_PROTOCOL_ID = f"v10.18_lite_train_ablation_{variant}"
    return originals


def _restore_variant_names(originals: dict) -> None:
    opt_script.FINAL_OUTPUT_NAME = originals["final"]
    opt_script.FINAL_PROTOCOL_ID = originals["protocol"]
    opt_script.OPTLONG_FULLTRAIN_OUTPUT = originals["optlong"]
    opt_script.DUALMAX_FULLTRAIN_OUTPUT = originals["dualmax"]


def _train_feature_head_no_cache(
    resume_path: Path,
    train_banks: list[dict],
    val_clean_bank: dict,
    val_noise_bank: dict,
    val_ecg_bank: dict,
    val_ppg_bank: dict,
    cfg,
):
    model, state, metrics, rows = opt_script._ORIG_RUN_FEATURE_HEAD_RESUME(
        resume_path,
        train_banks,
        val_clean_bank,
        val_noise_bank,
        val_ecg_bank,
        val_ppg_bank,
        cfg,
    )
    cached_head = _project_root() / "outputs" / opt_script.FINAL_OUTPUT_NAME / "feature_head_best.pt"
    cached_meta = cached_head.with_name("feature_head_cache_meta.json")
    try:
        cached_meta.parent.mkdir(parents=True, exist_ok=True)
        cached_meta.write_text(
            json.dumps(
                {
                    "score_version": opt_script.V10_18_SCORE_VERSION,
                    "head_selection_rank_mode": str(getattr(cfg, "HEAD_SELECTION_RANK_MODE", "")),
                    "selected_epoch": int(rows[-1].get("epoch", 0)) if rows else 0,
                    "source": "v10.18_lite_train_ablation_no_cache",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass
    return model, state, metrics, rows


def _uses_shared_full_head(variant: str, args: argparse.Namespace) -> bool:
    return variant not in VARIANT_TRAINED_HEADS and not bool(getattr(args, "force_train_each_variant", False))


def _head_reference_variant(variant: str) -> str:
    return "major_full" if variant in MAJOR_VARIANTS else "full"


def _make_feature_head_runner(variant: str, args: argparse.Namespace) -> Callable:
    def run_feature_head(
        resume_path: Path,
        train_banks: list[dict],
        val_clean_bank: dict,
        val_noise_bank: dict,
        val_ecg_bank: dict,
        val_ppg_bank: dict,
        cfg,
    ):
        if _uses_shared_full_head(variant, args):
            reference_variant = _head_reference_variant(variant)
            source_head = _variant_output_dir(reference_variant) / "feature_head_best.pt"
            dest_head = _variant_output_dir(variant) / "feature_head_best.pt"
            if source_head.exists():
                loader = opt_script.v17.v16.meta_script.prev_script.load_feature_head_checkpoint
                in_dim = int(train_banks[0]["x"].shape[1])
                try:
                    model, state = loader(source_head, in_dim, cfg)
                    dest_head.parent.mkdir(parents=True, exist_ok=True)
                    if source_head.resolve() != dest_head.resolve():
                        shutil.copy2(source_head, dest_head)
                    dest_head.with_name("feature_head_cache_meta.json").write_text(
                        json.dumps(
                            {
                                "score_version": opt_script.V10_18_SCORE_VERSION,
                                "head_selection_rank_mode": str(getattr(cfg, "HEAD_SELECTION_RANK_MODE", "")),
                                "source": "v10.18_lite_train_ablation_shared_full_head",
                                "source_output": _variant_output_name(reference_variant),
                                "variant": variant,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(
                        f"[v10.18 lite ablation] Reusing {reference_variant} lite-trained feature head for {variant}: {source_head}",
                        flush=True,
                    )
                    metrics = {"score": float("nan"), "source": "shared_full_lite_head"}
                    rows = [
                        {
                            "epoch": 0,
                            "loss": float("nan"),
                            "score": float("nan"),
                            "source": "shared_full_lite_head",
                        }
                    ]
                    return model, state, metrics, rows
                except (OSError, RuntimeError, ValueError) as exc:
                    print(
                        f"[v10.18 lite ablation] Shared full head could not be loaded for {variant} ({exc}); "
                        "training a variant head.",
                        flush=True,
                    )
            else:
                print(
                    f"[v10.18 lite ablation] Reference lite head is missing for {variant}; training a variant head.",
                    flush=True,
                )
        return _train_feature_head_no_cache(
            resume_path,
            train_banks,
            val_clean_bank,
            val_noise_bank,
            val_ecg_bank,
            val_ppg_bank,
            cfg,
        )

    return run_feature_head


def _build_variant_cfg(original_build: Callable, variant: str, args: argparse.Namespace):
    cfg = original_build()
    cfg.PROTOCOL_ID = opt_script.FINAL_PROTOCOL_ID
    cfg.OUTPUT_NAME = opt_script.FINAL_OUTPUT_NAME
    cfg.PROTOCOL_NAME = f"v10.18 lite-train ablation: {variant}"
    cfg.FINAL_OUTPUT_NAME = opt_script.FINAL_OUTPUT_NAME
    cfg.FINAL_PROTOCOL_ID = opt_script.FINAL_PROTOCOL_ID
    cfg = _apply_lite_search_profile(cfg, args)
    if variant == "no_tail_reweighting":
        cfg = _mutate_no_tail_lite_head(cfg)
    if variant == "no_accf1_targeted_head":
        cfg = strict_ablation._mutate_no_accf1_head(cfg)
        cfg.HEAD_EPOCHS = int(args.head_epochs)
        cfg.HEAD_PATIENCE = int(args.head_patience)
        cfg.HEAD_MIN_EPOCHS = int(args.head_min_epochs)
    if variant == "no_guided_policy_search":
        cfg = _mutate_no_guided_policy_search(cfg)
    if variant == "no_classification_arbiter":
        cfg = _mutate_no_classification_arbiter(cfg)
    if variant == "no_baseline_guarded_selector":
        cfg = _mutate_no_baseline_guarded_selector(cfg)
    if variant == "no_crisis_guard_floor":
        cfg = _mutate_no_crisis_guard_floor(cfg)
    if variant == "no_quality_sparse_fusion":
        cfg = _mutate_no_quality_sparse_fusion(cfg)
    if variant == "no_uncertainty_credibility":
        cfg = _mutate_no_uncertainty_credibility(cfg)
    if variant == "no_anchor_decision_refinement":
        cfg = _mutate_no_anchor_decision_refinement(cfg)
    if variant == "no_tail_safety_calibration":
        cfg = _mutate_no_tail_safety_calibration(cfg)
    if variant == "no_personalized_conformal":
        cfg = _mutate_no_personalized_conformal(cfg)
    return cfg


def run_variant(variant: str, args: argparse.Namespace) -> Path:
    output = _variant_output_dir(variant)
    if _is_variant_complete(output) and not args.rerun_existing:
        print(f"[v10.18 lite ablation] Reusing completed output: {output}", flush=True)
        return output
    if output.exists() and not args.rerun_existing:
        missing = [relative for relative in REQUIRED_OUTPUT_FILES if not (output / relative).exists()]
        print(
            f"[v10.18 lite ablation] Resuming incomplete output for {variant}; missing: {missing}",
            flush=True,
        )

    name_originals = _set_variant_names(variant)
    original_build: Callable = opt_script.build_nextgen_cfg
    original_head = opt_script._cached_or_train_feature_head
    original_high_bias = opt_script.v17.tailaware_search_high_bias_calibration_candidates_pareto
    original_crisis = opt_script.v17.search_crisis_tail_debias_candidates_pareto
    original_safety = opt_script.v17.guarded_search_safety_class_fusion_pareto
    stage_script = opt_script.v17.v16.meta_script.stage_script
    original_subject_calibration = {
        "fit": stage_script.fit_subject_calibration_state,
        "apply": stage_script.apply_subject_calibration,
    }
    original_v18_hooks = {
        "class_score": opt_script._classification_pareto_score_v18,
        "class_score_batch": opt_script._classification_pareto_score_batch_v18,
        "class_gate": opt_script._classification_gate_v18,
        "high_score": opt_script._high_bias_row_score_v18,
        "crisis_score": opt_script._crisis_tail_row_score_v18,
        "select_reg": opt_script._select_regression_row_v18,
    }

    def build_nextgen_cfg():
        return _build_variant_cfg(original_build, variant, args)

    try:
        opt_script.build_nextgen_cfg = build_nextgen_cfg
        opt_script._cached_or_train_feature_head = _make_feature_head_runner(variant, args)
        if variant == "no_baseline_guarded_selector":
            opt_script._classification_pareto_score_v18 = opt_script._V17_CLASS_SCORE
            opt_script._classification_pareto_score_batch_v18 = opt_script._V17_CLASS_SCORE_BATCH
            opt_script._classification_gate_v18 = opt_script._V17_CLASS_GATE
            opt_script._high_bias_row_score_v18 = opt_script._V17_HIGH_BIAS_SCORE
            opt_script._crisis_tail_row_score_v18 = opt_script._V17_CRISIS_TAIL_SCORE
            opt_script._select_regression_row_v18 = opt_script._V17_SELECT_REGRESSION_ROW
        if variant == "no_anchor_decision_refinement":
            opt_script._classification_pareto_score_v18 = opt_script._V17_CLASS_SCORE
            opt_script._classification_pareto_score_batch_v18 = opt_script._V17_CLASS_SCORE_BATCH
            opt_script._classification_gate_v18 = opt_script._V17_CLASS_GATE
        if variant == "no_reliability_bias_calibration":
            opt_script.v17.tailaware_search_high_bias_calibration_candidates_pareto = strict_ablation._identity_high_bias_search
        if variant == "no_crisis_tail_debias":
            opt_script.v17.search_crisis_tail_debias_candidates_pareto = strict_ablation._identity_crisis_tail_search
        if variant == "no_safety_evidential_fusion":
            opt_script.v17.guarded_search_safety_class_fusion_pareto = strict_ablation._identity_safety_fusion_search
        if variant == "no_tail_safety_calibration":
            opt_script.v17.tailaware_search_high_bias_calibration_candidates_pareto = strict_ablation._identity_high_bias_search
            opt_script.v17.search_crisis_tail_debias_candidates_pareto = strict_ablation._identity_crisis_tail_search
            opt_script.v17.guarded_search_safety_class_fusion_pareto = strict_ablation._identity_safety_fusion_search
        if variant == "no_personalized_conformal":
            stage_script.fit_subject_calibration_state = _identity_subject_calibration_state
            stage_script.apply_subject_calibration = _apply_identity_subject_calibration
        opt_script.main()
        return output
    finally:
        opt_script.build_nextgen_cfg = original_build
        opt_script._cached_or_train_feature_head = original_head
        opt_script.v17.tailaware_search_high_bias_calibration_candidates_pareto = original_high_bias
        opt_script.v17.search_crisis_tail_debias_candidates_pareto = original_crisis
        opt_script.v17.guarded_search_safety_class_fusion_pareto = original_safety
        stage_script.fit_subject_calibration_state = original_subject_calibration["fit"]
        stage_script.apply_subject_calibration = original_subject_calibration["apply"]
        opt_script._classification_pareto_score_v18 = original_v18_hooks["class_score"]
        opt_script._classification_pareto_score_batch_v18 = original_v18_hooks["class_score_batch"]
        opt_script._classification_gate_v18 = original_v18_hooks["class_gate"]
        opt_script._high_bias_row_score_v18 = original_v18_hooks["high_score"]
        opt_script._crisis_tail_row_score_v18 = original_v18_hooks["crisis_score"]
        opt_script._select_regression_row_v18 = original_v18_hooks["select_reg"]
        _restore_variant_names(name_originals)


def _summary_row(variant: str, args: argparse.Namespace, dry_run_info: dict | None = None) -> dict:
    output = _variant_output_dir(variant)
    shared_head = _uses_shared_full_head(variant, args)
    head_reference = _head_reference_variant(variant)
    complete = _is_variant_complete(output)
    summary = _read_json(output / "protocol_summary.json")
    results = _read_json(output / "final_results.json")
    paper = _read_json(output / "paper_metrics.json")
    runtime = _read_json(output / "runtime_metrics.json")
    selected = _read_json(output / "selected_strategy.json")
    aami = paper.get("aami_like", {})
    bhs = paper.get("bhs_like", {})
    high = _range_row(output, "high")
    crisis = _range_row(output, "crisis")
    row = {
        "name": variant,
        "kind": "lite_train_ablation",
        "module_removed": MODULE_REMOVED.get(variant, ""),
        "mode": "trained_short_head_reduced_search",
        "head_policy": "shared_full_lite_head" if shared_head else "variant_lite_trained",
        "head_source_output": _variant_output_name(head_reference) if shared_head else "",
        "output": _variant_output_name(variant),
        "exists": output.exists(),
        "completed": complete,
        "has_protocol_summary": (output / "protocol_summary.json").exists(),
        "has_final_results": (output / "final_results.json").exists(),
        "has_paper_metrics": (output / "paper_metrics.json").exists(),
        "has_selected_strategy": (output / "selected_strategy.json").exists(),
        "has_bp_range_metrics": (output / "tables" / "bp_range_metrics.csv").exists(),
        "has_runtime_metrics": (output / "runtime_metrics.json").exists(),
        "missing_outputs": ";".join(relative for relative in REQUIRED_OUTPUT_FILES if not (output / relative).exists()),
        "search_profile": args.search_profile,
        "head_epochs": args.head_epochs,
        "head_patience": args.head_patience,
        "head_min_epochs": args.head_min_epochs,
        "run_tag": RUN_TAG,
        "acc": summary.get("selected_acc", _metric_from_results(results, "cls_acc_from_reg", "acc")),
        "macro_f1": summary.get("selected_macro_f1", _metric_from_results(results, "cls_f1_macro_from_reg", "macro_f1")),
        "balanced_acc": summary.get(
            "selected_balanced_acc",
            _metric_from_results(results, "cls_balanced_acc_from_reg", "balanced_acc"),
        ),
        "mae_mean": summary.get("selected_mae_mean", _metric_from_results(results, "mae_mean")),
        "mae_sbp": _metric_from_results(results, "mae_sbp"),
        "mae_dbp": _metric_from_results(results, "mae_dbp"),
        "bias_sbp": _metric_from_results(results, "bias_sbp"),
        "bias_dbp": _metric_from_results(results, "bias_dbp"),
        "sbp_mean_error": aami.get("sbp_mean_error", ""),
        "sbp_sd_error": aami.get("sbp_sd_error", ""),
        "dbp_mean_error": aami.get("dbp_mean_error", ""),
        "dbp_sd_error": aami.get("dbp_sd_error", ""),
        "sbp_within_5": bhs.get("sbp_within_5", ""),
        "sbp_within_10": bhs.get("sbp_within_10", ""),
        "sbp_within_15": bhs.get("sbp_within_15", ""),
        "sbp_grade": bhs.get("sbp_grade", ""),
        "dbp_within_5": bhs.get("dbp_within_5", ""),
        "dbp_within_10": bhs.get("dbp_within_10", ""),
        "dbp_within_15": bhs.get("dbp_within_15", ""),
        "dbp_grade": bhs.get("dbp_grade", ""),
        "high_n": _to_float(high.get("n", "")),
        "high_bias_sbp": _to_float(high.get("bias_sbp", "")),
        "high_bias_dbp": _to_float(high.get("bias_dbp", "")),
        "crisis_n": _to_float(crisis.get("n", "")),
        "crisis_bias_sbp": _to_float(crisis.get("bias_sbp", "")),
        "crisis_bias_dbp": _to_float(crisis.get("bias_dbp", "")),
        "noise_macro_f1_auc": _csv_metric_value(
            output,
            "tables/robustness_summary.csv",
            {"group": "noise", "metric": "macro_f1_auc"},
            "value",
        ),
        "noise_accuracy_auc": _csv_metric_value(
            output,
            "tables/robustness_summary.csv",
            {"group": "noise", "metric": "accuracy_auc"},
            "value",
        ),
        "missing_ppg_macro_f1_auc": _csv_metric_value(
            output,
            "tables/robustness_summary.csv",
            {"group": "missing", "metric": "ppg_macro_f1_auc"},
            "value",
        ),
        "missing_ecg_macro_f1_auc": _csv_metric_value(
            output,
            "tables/robustness_summary.csv",
            {"group": "missing", "metric": "ecg_macro_f1_auc"},
            "value",
        ),
        "conformal_coverage_sbp_a10": _csv_metric_value(
            output,
            "conformal_sweep.csv",
            {"alpha": "0.1"},
            "coverage_sbp",
        ),
        "conformal_coverage_dbp_a10": _csv_metric_value(
            output,
            "conformal_sweep.csv",
            {"alpha": "0.1"},
            "coverage_dbp",
        ),
        "conformal_miw_sbp_a10": _csv_metric_value(
            output,
            "conformal_sweep.csv",
            {"alpha": "0.1"},
            "miw_sbp",
        ),
        "conformal_miw_dbp_a10": _csv_metric_value(
            output,
            "conformal_sweep.csv",
            {"alpha": "0.1"},
            "miw_dbp",
        ),
        "subject_gain_optlong_mean": _csv_mean(output, "tables/subject_gain_optlong.csv", "mae_mean_gain"),
        "subject_gain_dualmax_mean": _csv_mean(output, "tables/subject_gain_dualmax.csv", "mae_mean_gain"),
        "selected_regression_candidate": summary.get("selected_regression_candidate", ""),
        "selected_classification_candidate": summary.get("selected_classification_candidate", ""),
        "selected_tail_correction_candidate": summary.get("selected_tail_correction_candidate", ""),
        "crisis_tail_fusion_candidate": summary.get(
            "crisis_tail_fusion_candidate",
            selected.get("crisis_tail_fusion_candidate", ""),
        ),
        "runtime_seconds": runtime.get("runtime_seconds", ""),
    }
    if dry_run_info:
        row.update(dry_run_info)
    return row


DELTA_METRICS = (
    "mae_mean",
    "mae_sbp",
    "mae_dbp",
    "acc",
    "macro_f1",
    "balanced_acc",
    "high_bias_sbp",
    "high_bias_dbp",
    "crisis_bias_sbp",
    "crisis_bias_dbp",
    "noise_macro_f1_auc",
    "missing_ppg_macro_f1_auc",
    "missing_ecg_macro_f1_auc",
    "subject_gain_optlong_mean",
    "subject_gain_dualmax_mean",
)


def _float_or_none(value):
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _fmt_delta(value) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _annotate_ablation_rows(rows: List[dict]) -> List[dict]:
    full = next((row for row in rows if row.get("name") == "major_full"), None)
    if full is None:
        full = next((row for row in rows if row.get("name") == "full"), None)
    if full is None:
        return rows
    reference_name = str(full.get("name", "full"))
    full_values = {metric: _float_or_none(full.get(metric)) for metric in DELTA_METRICS}
    for row in rows:
        deltas = {}
        for metric in DELTA_METRICS:
            value = _float_or_none(row.get(metric))
            base = full_values.get(metric)
            delta = None if value is None or base is None else value - base
            deltas[metric] = delta
            row[f"delta_{metric}_vs_full"] = _fmt_delta(delta)
        row["ablation_reference"] = reference_name

        d_mae = deltas.get("mae_mean")
        d_f1 = deltas.get("macro_f1")
        d_acc = deltas.get("acc")
        d_high_bias = deltas.get("high_bias_sbp")
        d_crisis_bias = deltas.get("crisis_bias_sbp")
        d_noise_auc = deltas.get("noise_macro_f1_auc")
        row["effect_summary"] = (
            f"dMAE={_fmt_delta(d_mae)}; dAcc={_fmt_delta(d_acc)}; "
            f"dMacroF1={_fmt_delta(d_f1)}; dHighSBPBias={_fmt_delta(d_high_bias)}; "
            f"dNoiseF1AUC={_fmt_delta(d_noise_auc)}"
        )

        name = str(row.get("name", ""))
        complete = _boolish(row.get("completed"))
        crisis_n = _float_or_none(row.get("crisis_n")) or 0.0
        crisis_candidate = str(row.get("crisis_tail_fusion_candidate", "")).lower()
        meaningful = any(
            abs(value) >= threshold
            for value, threshold in (
                (d_mae, 0.01),
                (d_acc, 0.002),
                (d_f1, 0.002),
                (d_high_bias, 0.10),
                (d_crisis_bias, 0.50),
                (d_noise_auc, 0.003),
            )
            if value is not None
        )

        validity = "paper_usable"
        use_main = True
        note = ""
        if name == reference_name:
            validity = "reference"
            note = f"{reference_name} lite-train v10.18 reference for ablation deltas."
        elif not complete:
            validity = "incomplete"
            use_main = False
            note = "Run is incomplete; do not report as an ablation result."
        elif name == "no_personalized_conformal" and not meaningful:
            validity = "no_observed_effect_after_disabling_personalization"
            use_main = False
            note = (
                "Subject-level calibration and conformal selection constraints did not change the "
                "reported operating point under this reduced protocol; do not use as a main-table claim."
            )
        elif name == "no_tail_safety_calibration" and d_mae is not None and d_mae < 0 and d_high_bias is not None:
            validity = "safety_tradeoff_result"
            note = (
                "Removing the safety/tail calibration improves aggregate MAE but worsens signed high-range "
                "bias; frame this row as a safety tradeoff rather than an accuracy-only gain."
            )
        elif name in MAJOR_VARIANTS:
            validity = "paper_usable_major_module"
            note = "Equal-budget frozen-backbone major-module ablation; report in the main ablation table."
        elif name == "no_tail_reweighting" and not meaningful:
            validity = "tail_head_ablation_no_observed_effect"
            use_main = False
            note = (
                "This row now removes tail emphasis in the trainable lite head stage. If the effect "
                "remains small, report it as evidence that aggregate metrics are insensitive to "
                "tail reweighting under the frozen-backbone protocol."
            )
        elif name == "no_crisis_tail_debias" and ("identity" in crisis_candidate or not meaningful):
            validity = "inactive_identity_or_no_observed_effect"
            use_main = False
            note = (
                "The crisis-tail selector chose identity/no-op under this split, so this is not "
                "informative as a main ablation."
            )
        elif name == "no_crisis_guard_floor" and crisis_n < 10:
            validity = "supplement_only_low_crisis_support"
            use_main = False
            note = (
                "Crisis support is too small for a stable main-table claim; report only as a "
                "supplementary tail-stability check."
            )
        elif not meaningful:
            validity = "no_observed_effect_under_lite_protocol"
            use_main = False
            note = "Effect is below the reporting threshold for the lite-train protocol."
        elif name == "no_reliability_bias_calibration" and d_mae is not None and d_mae < 0:
            validity = "tradeoff_result"
            note = (
                "Removing this module improves aggregate MAE but weakens bias control; frame this "
                "as an accuracy-bias tradeoff, not a pure accuracy gain."
            )

        row["paper_ablation_validity"] = validity
        row["use_in_main_ablation_table"] = str(bool(use_main))
        row["paper_note"] = note
    return rows


def _is_major_run(variants: Iterable[str], args: argparse.Namespace) -> bool:
    variants = list(variants)
    return (
        str(getattr(args, "variant_set", "")) == "major"
        or str(getattr(args, "variant", "")) in MAJOR_VARIANTS
        or (bool(variants) and all(variant in MAJOR_VARIANTS for variant in variants))
    )


def _summary_filename(variants: Iterable[str], args: argparse.Namespace) -> str:
    return "major_module_ablation_summary.csv" if _is_major_run(variants, args) else "lite_train_ablation_summary.csv"


def _manifest_filename(variants: Iterable[str], args: argparse.Namespace) -> str:
    return "major_module_ablation_manifest.json" if _is_major_run(variants, args) else "lite_train_ablation_manifest.json"


def summarize_variants(variants: Iterable[str], args: argparse.Namespace, dry_run_infos: dict | None = None) -> Path:
    dry_run_infos = dry_run_infos or {}
    variants = list(variants)
    rows = _annotate_ablation_rows([_summary_row(variant, args, dry_run_infos.get(variant)) for variant in variants])
    out = _summary_dir() / _summary_filename(variants, args)
    _write_csv(out, rows)
    return out


def write_manifest(variants: Iterable[str], args: argparse.Namespace, dry_run_infos: dict | None = None) -> Path:
    variants = list(variants)
    major_run = _is_major_run(variants, args)
    manifest = {
        "mode": (
            "major_module_equal_budget_frozen_backbone_reduced_search"
            if major_run
            else "stagewise_equalized_lite_train_reduced_search"
        ),
        "variant_set": args.variant_set,
        "variants": variants,
        "search_profile": args.search_profile,
        "head_epochs": args.head_epochs,
        "head_patience": args.head_patience,
        "head_min_epochs": args.head_min_epochs,
        "run_tag": RUN_TAG,
        "force_train_each_variant": bool(getattr(args, "force_train_each_variant", False)),
        "protocol_note": (
            "Major-module suite reports Full + five paper-facing module ablations. All variants use "
            "the same frozen backbone/anchor checkpoints, subject-disjoint splits, and reduced "
            "vectorized selector grids. Only modules that directly change the guided feature head "
            "(the full reference and the quality/sparse-fusion proxy) retrain that head by default; "
            "downstream decision, uncertainty, safety, and personalization ablations share the "
            "major_full head and rerun their affected search/evaluation stages. The SAP/conformal "
            "ablation disables subject-level affine calibration through the stage-aware stack."
            if major_run
            else (
                "Paper suite uses stage-wise equalized lite feature-head training plus reduced candidate "
                "grids. Training/head-selection ablations train their own head; downstream selector "
                "ablations reuse the full lite-trained head and rerun only the affected search/evaluation. "
                "The no_tail_reweighting variant also disables tail emphasis in the trainable head stage, "
                "because full backbone tail-loss retraining is intentionally outside the 24h protocol."
            )
        ),
        "dry_run_candidate_totals": dry_run_infos or {},
        "outputs": {variant: _variant_output_name(variant) for variant in variants},
        "head_policy": {
            variant: "shared_full_lite_head" if _uses_shared_full_head(variant, args) else "variant_lite_trained"
            for variant in variants
        },
        "head_source_output": {
            variant: _variant_output_name(_head_reference_variant(variant)) if _uses_shared_full_head(variant, args) else ""
            for variant in variants
        },
    }
    out = _summary_dir() / _manifest_filename(variants, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def _select_variants(args: argparse.Namespace) -> list[str]:
    if args.variant:
        return [args.variant]
    if args.variant_set == "core":
        return list(CORE_VARIANTS)
    if args.variant_set == "paper":
        return list(PAPER_VARIANTS)
    if args.variant_set == "major":
        return list(MAJOR_VARIANTS)
    return list(ALL_VARIANTS)


def _dry_run_variants(variants: Iterable[str], args: argparse.Namespace) -> dict:
    infos = {}
    original_build: Callable = opt_script.build_nextgen_cfg
    for variant in variants:
        name_originals = _set_variant_names(variant)
        try:
            cfg = _build_variant_cfg(original_build, variant, args)
            infos[variant] = {
                **_candidate_totals(cfg),
                "output": _variant_output_name(variant),
            }
            print(f"[v10.18 lite ablation] Dry-run {variant}: {infos[variant]}", flush=True)
        finally:
            _restore_variant_names(name_originals)
    return infos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-usable lite-training v10.18 ablations.")
    parser.add_argument("--variant", choices=ALL_VARIANTS, default=None)
    parser.add_argument("--variant-set", choices=("core", "paper", "major", "all"), default="paper")
    parser.add_argument("--search-profile", choices=("tiny",), default="tiny")
    parser.add_argument("--head-epochs", type=int, default=32)
    parser.add_argument("--head-patience", type=int, default=8)
    parser.add_argument("--head-min-epochs", type=int, default=8)
    parser.add_argument("--force-train-each-variant", action="store_true")
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional suffix inserted before _proto to avoid mixing old and corrected ablation outputs.",
    )
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    global RUN_TAG
    args = parse_args()
    RUN_TAG = str(args.run_tag).strip().replace(" ", "_")
    variants = _select_variants(args)
    start = time.time()
    dry_run_infos = _dry_run_variants(variants, args) if args.dry_run else {}
    manifest_path = write_manifest(variants, args, dry_run_infos)
    if args.summarize_only:
        summary_path = summarize_variants(variants, args)
        print(f"[v10.18 lite ablation] Summary refreshed from existing outputs: {summary_path}")
        print(f"[v10.18 lite ablation] Manifest saved to: {manifest_path}")
        print(f"[v10.18 lite ablation] Elapsed seconds: {time.time() - start:.1f}")
        return
    if args.dry_run:
        print(f"[v10.18 lite ablation] Dry-run manifest saved to: {manifest_path}")
        print(f"[v10.18 lite ablation] Elapsed seconds: {time.time() - start:.1f}")
        return
    summary_path = summarize_variants(variants, args)
    print(f"[v10.18 lite ablation] Initial resumable summary saved to: {summary_path}", flush=True)
    try:
        for variant in variants:
            print(f"[v10.18 lite ablation] Running {variant}", flush=True)
            run_variant(variant, args)
            summary_path = summarize_variants(variants, args)
            print(f"[v10.18 lite ablation] Checkpoint summary saved to: {summary_path}", flush=True)
    except KeyboardInterrupt:
        summary_path = summarize_variants(variants, args)
        print(f"[v10.18 lite ablation] Interrupted; partial summary saved to: {summary_path}", flush=True)
        raise
    except Exception:
        summary_path = summarize_variants(variants, args)
        print(f"[v10.18 lite ablation] Failed; partial summary saved to: {summary_path}", flush=True)
        raise
    print(f"[v10.18 lite ablation] Summary saved to: {summary_path}")
    print(f"[v10.18 lite ablation] Manifest saved to: {manifest_path}")
    print(f"[v10.18 lite ablation] Elapsed seconds: {time.time() - start:.1f}")


if __name__ == "__main__":
    main()
