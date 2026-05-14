from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

import train_aqm_medfuse_mimic_bp_reg_v10_17_subjectdisjoint_paretoordinal_crisiscal_protocol as v17


FINAL_OUTPUT_NAME = "mimic_bp_reg_v10_18_subjectdisjoint_baselineguard_crisisrepair_proto"
FINAL_PROTOCOL_ID = "v10.18_subjectdisjoint_baselineguard_crisisrepair"
V10_18_SCORE_VERSION = "v10.18_baselineguard_crisisrepair_20260501_accheadrank"

# v10.17 moved the full-train anchors forward, but the attached audit shows the
# v10.11/v10.2 anchors are still the stronger subject-disjoint operating point.
OPTLONG_FULLTRAIN_OUTPUT = "mimic_bp_reg_v10_2_opt_long_proto"
DUALMAX_FULLTRAIN_OUTPUT = "mimic_bp_reg_v10_2_optlong_stageaware_dualmax_proto"

_V17_BUILD_NEXTGEN_CFG = v17.build_nextgen_cfg
_V17_CLASS_SCORE = v17._classification_pareto_score
_V17_CLASS_SCORE_BATCH = v17._classification_pareto_score_batch
_V17_CLASS_GATE = v17._classification_gate
_V17_HIGH_BIAS_SCORE = v17._high_bias_row_score
_V17_CRISIS_TAIL_SCORE = v17._crisis_tail_row_score
_V17_SELECT_REGRESSION_ROW = v17._select_regression_row
_V17_GENERATE_EXTRA_OUTPUTS = v17.generate_extra_outputs
_ORIG_RUN_FEATURE_HEAD_RESUME = v17.v16.meta_script.prev_script.run_feature_head_resume
_ORIG_V16_ADAPTIVE_BATCH_SIZE = v17.v16._adaptive_batch_size
_ORIG_V16_MERGE_TOP_ROWS = v17.v16._merge_top_rows
_ORIG_V16_VECTOR_HIGH_BIAS_ROWS = v17.v16._vectorized_high_bias_rows
_ORIG_V16_VECTOR_CRISIS_TAIL_ROWS = v17.v16._vectorized_crisis_tail_rows

_CRISIS_TAIL_SCORE_CUTOFF: float | None = None
_CRISIS_TAIL_MASK_CACHE: dict[tuple, tuple[list, list]] = {}


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=v17._json_default)


def _target(cfg, name: str, default: float) -> float:
    try:
        return float(getattr(cfg, name, default))
    except (TypeError, ValueError):
        return float(default)


def _vectorfast_batch_size_v18(total_samples: int, stage: str) -> int:
    cfg = getattr(v17, "_ACTIVE_CFG", None)
    if cfg is not None:
        if stage == "crisis_tail":
            override = int(getattr(cfg, "VECTORFAST_CRISIS_BATCH_SIZE", 0) or 0)
            if override > 0:
                return override
        if stage == "high_bias":
            override = int(getattr(cfg, "VECTORFAST_HIGH_BIAS_BATCH_SIZE", 0) or 0)
            if override > 0:
                return override
    return _ORIG_V16_ADAPTIVE_BATCH_SIZE(total_samples, stage)


def _merge_top_rows_v18(existing: List[dict], incoming: Iterable[dict], keep: int) -> List[dict]:
    global _CRISIS_TAIL_SCORE_CUTOFF
    incoming_rows = list(incoming)
    if not incoming_rows:
        rows = existing[:keep]
        if len(rows) >= keep and rows and "sbp_quantile" in rows[0]:
            _CRISIS_TAIL_SCORE_CUTOFF = float(rows[keep - 1].get("score", 0.0))
        return rows
    if keep <= 0:
        return []
    if len(existing) >= keep:
        # Existing rows are already score-sorted by the search loop. Most later
        # batches cannot enter the retained frontier, so avoid re-sorting 30k+
        # rows unless the new batch actually improves the current cutoff.
        cutoff = float(existing[keep - 1].get("score", 0.0))
        incoming_rows = [row for row in incoming_rows if float(row.get("score", 0.0)) < cutoff]
        if not incoming_rows:
            rows = existing[:keep]
            if len(rows) >= keep and rows and "sbp_quantile" in rows[0]:
                _CRISIS_TAIL_SCORE_CUTOFF = float(rows[keep - 1].get("score", 0.0))
            return rows

    merged = list(existing) + incoming_rows
    merged.sort(key=lambda item: float(item.get("score", 0.0)))
    dedup: dict[str, dict] = {}
    for row in merged:
        key = str(row.get("candidate", ""))
        if key not in dedup:
            dedup[key] = row
            if len(dedup) >= keep:
                break
    rows = list(dedup.values())
    if len(rows) >= keep and any("sbp_quantile" in row for row in incoming_rows + existing[:1]):
        _CRISIS_TAIL_SCORE_CUTOFF = float(rows[keep - 1].get("score", 0.0))
    return rows


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "unable to allocate" in text or "cuda out of memory" in text


def _safe_vectorized_rows_v18(vector_fn, batch_rows: List[dict], *args, stage: str) -> List[dict]:
    try:
        return vector_fn(batch_rows, *args)
    except Exception as exc:
        if not _is_oom_error(exc) or len(batch_rows) <= 1:
            raise
        mid = len(batch_rows) // 2
        print(
            f"[v10.18] {stage} batch OOM at {len(batch_rows)} candidates; "
            f"retrying as {mid}+{len(batch_rows) - mid} without dropping candidates."
        )
        left = _safe_vectorized_rows_v18(vector_fn, batch_rows[:mid], *args, stage=stage)
        right = _safe_vectorized_rows_v18(vector_fn, batch_rows[mid:], *args, stage=stage)
        return left + right


def _vectorized_high_bias_rows_v18(batch_rows: List[dict], calib_pre: dict, query_pre: dict, base_ref: dict, cfg) -> List[dict]:
    return _safe_vectorized_rows_v18(
        _ORIG_V16_VECTOR_HIGH_BIAS_ROWS,
        batch_rows,
        calib_pre,
        query_pre,
        base_ref,
        cfg,
        stage="high-bias",
    )


def _crisis_tail_masks_v18(query_pre: dict, cfg) -> tuple[list, list]:
    y_true = query_pre["y_true"]
    tail_q = tuple(float(x) for x in getattr(cfg, "CRISIS_TAIL_SURROGATE_QUANTILES", (0.90, 0.95, 0.98)))
    top_q = tuple(float(x) for x in getattr(cfg, "CRISIS_TAIL_SURROGATE_QUANTILES", (0.88, 0.92, 0.95, 0.98)))
    key = (id(y_true), y_true.shape, tail_q, top_q)
    cached = _CRISIS_TAIL_MASK_CACHE.get(key)
    if cached is None:
        cached = (
            v17.v16._tail_under_masks(y_true, tail_q),
            v17.v16._top_tail_masks(y_true, top_q),
        )
        _CRISIS_TAIL_MASK_CACHE[key] = cached
    return cached


def _vectorized_crisis_tail_rows_lazy_v18(
    batch_rows: List[dict],
    calib_pre: dict,
    query_pre: dict,
    base_ref: dict,
    cfg,
) -> List[dict]:
    if not batch_rows:
        return []

    cutoff = _CRISIS_TAIL_SCORE_CUTOFF
    if cutoff is None or not np.isfinite(cutoff):
        return _ORIG_V16_VECTOR_CRISIS_TAIL_ROWS(batch_rows, calib_pre, query_pre, base_ref, cfg)

    calib_pred_pre_guard, _ = v17.v16._build_base_crisis_tail_batch(batch_rows, calib_pre, cfg)
    query_pred_pre_guard, diag = v17.v16._build_base_crisis_tail_batch(batch_rows, query_pre, cfg)
    calib_pred, _ = v17.v16._apply_guard_batch(calib_pred_pre_guard, calib_pre, cfg)
    query_pred, guard_diag = v17.v16._apply_guard_batch(query_pred_pre_guard, query_pre, cfg)

    reg = v17.v16._batch_regression_summary(query_pre["y_true"], query_pred)
    range_bias = v17.v16._batch_bp_bias_summary(reg["err_sbp"], reg["err_dbp"], query_pre["masks"])
    clinical_pen = v17.v16._batch_clinical_penalty(range_bias)
    tail_pen = v17.v16._batch_tail_bias_penalty(range_bias)
    tail_masks, top_tail_masks = _crisis_tail_masks_v18(query_pre, cfg)
    surrogate_pen = v17.v16._batch_tail_under_penalty(reg["err_sbp"], reg["err_dbp"], tail_masks)
    top_tail_pen = v17.v16._batch_top_tail_under_penalty(reg["err_sbp"], reg["err_dbp"], top_tail_masks)
    baseline_pen = v17.v16._batch_baseline_bias_penalty(range_bias, cfg)

    crisis_under_pen = (
        float(getattr(cfg, "CRISIS_TAIL_UNDEREST_WEIGHT_SBP", 8.50)) * np.maximum(0.0, -range_bias["crisis_bias_sbp"])
        + float(getattr(cfg, "CRISIS_TAIL_UNDEREST_WEIGHT_DBP", 3.10)) * np.maximum(0.0, -range_bias["crisis_bias_dbp"])
    ).astype(np.float32)
    crisis_abs_pen = (1.35 * np.abs(range_bias["crisis_bias_sbp"]) + 0.70 * np.abs(range_bias["crisis_bias_dbp"])).astype(np.float32)
    high_under_pen = (1.55 * np.maximum(0.0, -range_bias["high_bias_sbp"]) + 0.82 * np.maximum(0.0, -range_bias["high_bias_dbp"])).astype(np.float32)
    mae_excess = np.maximum(0.0, reg["mae_mean"] - float(base_ref["mae_mean"]) - float(cfg.CRISIS_TAIL_FUSION_MAX_MAE_DELTA))

    lower_score = (
        1.90 * surrogate_pen
        + crisis_under_pen
        + 0.55 * crisis_abs_pen
        + 0.75 * high_under_pen
        + 0.42 * clinical_pen
        + 0.20 * tail_pen
        + 18.0 * mae_excess
        + 0.008 * reg["mae_mean"]
        + 4.25 * np.maximum(0.0, -range_bias["crisis_bias_sbp"])
        + 1.35 * np.maximum(0.0, -range_bias["crisis_bias_dbp"])
        + 1.15 * np.maximum(0.0, -range_bias["high_bias_sbp"])
        + 2.10 * top_tail_pen
        + baseline_pen
        + 2.50
        * (
            np.maximum(0.0, guard_diag["guard_activation_rate"] - 0.18)
            * np.maximum(0.0, guard_diag["guard_shift_mean_sbp"] - 1.2)
        )
    ).astype(np.float32)

    survivor_idx = np.flatnonzero(lower_score <= float(cutoff))
    if survivor_idx.size == 0:
        return []

    conformal = v17.v16._batch_conformal_summary(
        calib_pre["y_true"],
        calib_pred[survivor_idx],
        calib_pre["unc_scale"],
        query_pre["y_true"],
        query_pred[survivor_idx],
        query_pre["unc_scale"],
        alpha=float(cfg.CONFORMAL_ALPHA),
    )
    cov_excess = np.maximum(
        0.0,
        conformal["coverage_gap"] - float(base_ref["coverage_gap"]) - float(cfg.CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA),
    )
    score = (lower_score[survivor_idx] + 8.0 * cov_excess).astype(np.float32)

    rows: List[dict] = []
    for local_idx, idx in enumerate(survivor_idx.tolist()):
        if float(score[local_idx]) > float(cutoff):
            continue
        row = batch_rows[idx]
        rows.append(
            {
                **row,
                "score": float(score[local_idx]),
                "clinical_under_penalty": float(clinical_pen[idx]),
                "tail_bias_penalty": float(tail_pen[idx] + baseline_pen[idx]),
                "high_bias_sbp": float(range_bias["high_bias_sbp"][idx]),
                "high_bias_dbp": float(range_bias["high_bias_dbp"][idx]),
                "crisis_bias_sbp": float(range_bias["crisis_bias_sbp"][idx]),
                "crisis_bias_dbp": float(range_bias["crisis_bias_dbp"][idx]),
                "fusion_gate_mean": float(diag["fusion_gate_mean"][idx]),
                "shift_mean_sbp": float(diag["shift_mean_sbp"][idx]),
                "shift_mean_dbp": float(diag["shift_mean_dbp"][idx]),
                "activation_rate": float(diag["activation_rate"][idx]),
                "guard_shift_mean_sbp": float(guard_diag["guard_shift_mean_sbp"][idx]),
                "guard_shift_mean_dbp": float(guard_diag["guard_shift_mean_dbp"][idx]),
                "guard_activation_rate": float(guard_diag["guard_activation_rate"][idx]),
                "mae_sbp": float(reg["mae_sbp"][idx]),
                "mae_dbp": float(reg["mae_dbp"][idx]),
                "mae_mean": float(reg["mae_mean"][idx]),
                "bias_sbp": float(reg["bias_sbp"][idx]),
                "bias_dbp": float(reg["bias_dbp"][idx]),
                "coverage_sbp": float(conformal["coverage_sbp"][local_idx]),
                "coverage_dbp": float(conformal["coverage_dbp"][local_idx]),
                "miw_sbp": float(conformal["miw_sbp"][local_idx]),
                "miw_dbp": float(conformal["miw_dbp"][local_idx]),
                "coverage_gap": float(conformal["coverage_gap"][local_idx]),
                "miw_mean": float(conformal["miw_mean"][local_idx]),
            }
        )
    return rows


def _vectorized_crisis_tail_rows_v18(batch_rows: List[dict], calib_pre: dict, query_pre: dict, base_ref: dict, cfg) -> List[dict]:
    return _safe_vectorized_rows_v18(
        _vectorized_crisis_tail_rows_lazy_v18,
        batch_rows,
        calib_pre,
        query_pre,
        base_ref,
        cfg,
        stage="crisis-tail",
    )


def _baseline_metric(cfg, name: str, default: float) -> float:
    try:
        return float(getattr(cfg, name))
    except (AttributeError, TypeError, ValueError):
        return v17._baseline_metric(name, default)


def _v1011_output_dir() -> Path:
    return _project_root() / "outputs" / "mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_proto"


def _load_v1011_targets() -> dict:
    metrics = _read_json(_v1011_output_dir() / "final_results.json").get("test_selected", {})
    return {
        "acc": float(metrics.get("cls_acc_from_reg", metrics.get("acc", 0.8122807017543859))),
        "macro_f1": float(metrics.get("cls_f1_macro_from_reg", metrics.get("macro_f1", 0.7334048956487922))),
        "balanced_acc": float(metrics.get("cls_balanced_acc_from_reg", metrics.get("balanced_acc", 0.751738213088013))),
        "mae_mean": float(metrics.get("mae_mean", 6.086612939834595)),
    }


def _objective_gap(value: float, target: float) -> float:
    return max(0.0, float(target) - float(value))


def _classification_pareto_score_v18(summary: dict, cfg, row: dict | None = None) -> float:
    score = _V17_CLASS_SCORE(summary, cfg, row)
    acc = float(summary.get("acc", 0.0))
    macro_f1 = float(summary.get("macro_f1", 0.0))
    balanced = float(summary.get("balanced_acc", 0.0))

    base_acc = _baseline_metric(cfg, "CLASSIFICATION_BASELINE_ACC", 0.81228)
    base_f1 = _baseline_metric(cfg, "CLASSIFICATION_BASELINE_MACRO_F1", 0.73340)
    base_bal = _baseline_metric(cfg, "CLASSIFICATION_BASELINE_BALANCED_ACC", 0.75174)
    target_acc = float(getattr(cfg, "CLASSIFICATION_ASPIRATIONAL_ACC", 0.86))
    target_f1 = float(getattr(cfg, "CLASSIFICATION_ASPIRATIONAL_MACRO_F1", 0.80))

    # Make v10.11 dominance the primary selection pressure. The aspirational
    # terms still guide the search toward the paper target when candidates exist.
    baseline_penalty = (
        520.0 * _objective_gap(acc, base_acc)
        + 620.0 * _objective_gap(macro_f1, base_f1)
        + 190.0 * _objective_gap(balanced, base_bal)
    )
    aspirational_penalty = 72.0 * _objective_gap(acc, target_acc) + 82.0 * _objective_gap(macro_f1, target_f1)
    baseline_bonus = 45.0 * max(0.0, acc - base_acc) + 58.0 * max(0.0, macro_f1 - base_f1)
    return score - baseline_penalty - aspirational_penalty + baseline_bonus


def _classification_pareto_score_batch_v18(
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
    score = _V17_CLASS_SCORE_BATCH(
        acc,
        macro_f1,
        balanced,
        ece,
        rare_min,
        rare_mean,
        stage2,
        robust_min,
        cfg,
    )
    base_acc = _baseline_metric(cfg, "CLASSIFICATION_BASELINE_ACC", 0.81228)
    base_f1 = _baseline_metric(cfg, "CLASSIFICATION_BASELINE_MACRO_F1", 0.73340)
    base_bal = _baseline_metric(cfg, "CLASSIFICATION_BASELINE_BALANCED_ACC", 0.75174)
    target_acc = float(getattr(cfg, "CLASSIFICATION_ASPIRATIONAL_ACC", 0.86))
    target_f1 = float(getattr(cfg, "CLASSIFICATION_ASPIRATIONAL_MACRO_F1", 0.80))
    baseline_penalty = (
        520.0 * np.maximum(0.0, base_acc - acc)
        + 620.0 * np.maximum(0.0, base_f1 - macro_f1)
        + 190.0 * np.maximum(0.0, base_bal - balanced)
    )
    aspirational_penalty = 72.0 * np.maximum(0.0, target_acc - acc) + 82.0 * np.maximum(0.0, target_f1 - macro_f1)
    baseline_bonus = 45.0 * np.maximum(0.0, acc - base_acc) + 58.0 * np.maximum(0.0, macro_f1 - base_f1)
    return score - baseline_penalty - aspirational_penalty + baseline_bonus


def _classification_gate_v18(summary: dict, cfg) -> dict:
    base_gate = _V17_CLASS_GATE(summary, cfg)
    acc = float(summary.get("acc", 0.0))
    macro_f1 = float(summary.get("macro_f1", 0.0))
    balanced = float(summary.get("balanced_acc", 0.0))
    base_acc = _baseline_metric(cfg, "CLASSIFICATION_BASELINE_ACC", 0.81228)
    base_f1 = _baseline_metric(cfg, "CLASSIFICATION_BASELINE_MACRO_F1", 0.73340)
    base_bal = _baseline_metric(cfg, "CLASSIFICATION_BASELINE_BALANCED_ACC", 0.75174)
    min_acc = base_acc + _target(cfg, "BASELINE_MIN_ACC_MARGIN", 0.0)
    min_f1 = base_f1 + _target(cfg, "BASELINE_MIN_F1_MARGIN", 0.0)
    min_bal = base_bal + _target(cfg, "BASELINE_MIN_BAL_MARGIN", 0.0)
    gate = dict(base_gate) if isinstance(base_gate, dict) else {}
    gate.update(
        {
            "classification_gate_pass": bool(acc >= min_acc and macro_f1 >= min_f1 and balanced >= min_bal),
            "baseline_acc": base_acc,
            "baseline_macro_f1": base_f1,
            "baseline_balanced_acc": base_bal,
            "baseline_min_acc": min_acc,
            "baseline_min_macro_f1": min_f1,
            "baseline_min_balanced_acc": min_bal,
            "baseline_acc_gap": max(0.0, min_acc - acc),
            "baseline_macro_f1_gap": max(0.0, min_f1 - macro_f1),
            "baseline_balanced_acc_gap": max(0.0, min_bal - balanced),
            "baseline_acc_delta": acc - base_acc,
            "baseline_macro_f1_delta": macro_f1 - base_f1,
            "baseline_balanced_acc_delta": balanced - base_bal,
        }
    )
    return gate


def _cached_or_train_feature_head(
    resume_path: Path,
    train_banks: list[dict],
    val_clean_bank: dict,
    val_noise_bank: dict,
    val_ecg_bank: dict,
    val_ppg_bank: dict,
    cfg,
):
    cached_head = _project_root() / "outputs" / FINAL_OUTPUT_NAME / "feature_head_best.pt"
    cached_meta = cached_head.with_name("feature_head_cache_meta.json")
    if cached_head.exists():
        meta = _read_json(cached_meta) if cached_meta.exists() else {}
        cache_ok = (
            str(meta.get("score_version", "")) == V10_18_SCORE_VERSION
            and str(meta.get("head_selection_rank_mode", "")).lower()
            == str(getattr(cfg, "HEAD_SELECTION_RANK_MODE", "")).lower()
        )
        if not cache_ok:
            print(
                "[v10.18] Existing feature head cache was trained with an older "
                "selection policy; resuming head training.",
                flush=True,
            )
        else:
            loader = v17.v16.meta_script.prev_script.load_feature_head_checkpoint
            in_dim = int(train_banks[0]["x"].shape[1])
            try:
                model, state = loader(cached_head, in_dim, cfg)
                print(f"[v10.18] Reusing cached feature head: {cached_head}")
                metrics = {"score": float("nan"), "source": "cached_feature_head"}
                rows = [{"epoch": 0, "loss": float("nan"), "score": float("nan"), "source": "cached_feature_head"}]
                return model, state, metrics, rows
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"[v10.18] Cached feature head could not be loaded ({exc}); resuming head training.", flush=True)
    model, state, metrics, rows = _ORIG_RUN_FEATURE_HEAD_RESUME(
        resume_path,
        train_banks,
        val_clean_bank,
        val_noise_bank,
        val_ecg_bank,
        val_ppg_bank,
        cfg,
    )
    try:
        cached_meta.write_text(
            json.dumps(
                {
                    "score_version": V10_18_SCORE_VERSION,
                    "head_selection_rank_mode": str(getattr(cfg, "HEAD_SELECTION_RANK_MODE", "")),
                    "selected_epoch": int(rows[-1].get("epoch", 0)) if rows else 0,
                    "source": "v10.18_feature_head_rank",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass
    return model, state, metrics, rows


def _tail_bias_score(row: dict, cfg, base_score: float, *, high_stage: bool) -> float:
    mae = v17._row_float(row, "mae_mean")
    high_sbp = v17._row_float(row, "high_bias_sbp")
    high_dbp = v17._row_float(row, "high_bias_dbp")
    crisis_sbp = v17._row_float(row, "crisis_bias_sbp")
    crisis_dbp = v17._row_float(row, "crisis_bias_dbp")
    mae_target = _target(cfg, "REGRESSION_PARETO_MAE_TARGET", 5.55)
    high_sbp_target = _target(cfg, "HIGH_BIAS_TARGET_ABS_SBP", 3.5)
    high_dbp_target = _target(cfg, "HIGH_BIAS_TARGET_ABS_DBP", 3.0)
    crisis_sbp_target = _target(cfg, "CRISIS_BIAS_TARGET_ABS_SBP", 4.0)
    crisis_dbp_target = _target(cfg, "CRISIS_BIAS_TARGET_ABS_DBP", 3.5)
    activation = max(v17._row_float(row, "activation_rate"), v17._row_float(row, "guard_activation_rate"))
    shift_sbp = max(v17._row_float(row, "shift_mean_sbp"), v17._row_float(row, "guard_shift_mean_sbp"))
    sparse_min = _target(cfg, "CRISIS_REPAIR_MIN_ACTIVATION_RATE", 0.002)
    sparse_max = _target(cfg, "CRISIS_REPAIR_MAX_ACTIVATION_RATE", 0.24)

    score = base_score
    score += 60.0 * max(0.0, mae - mae_target)
    score += 12.0 * max(0.0, abs(high_sbp) - high_sbp_target)
    score += 10.0 * max(0.0, abs(high_dbp) - high_dbp_target)
    score += 30.0 * max(0.0, abs(crisis_sbp) - crisis_sbp_target)
    score += 18.0 * max(0.0, abs(crisis_dbp) - crisis_dbp_target)

    # The observed failure mode is SBP underestimation in crisis. Penalize the
    # signed direction more than a symmetric miss, while keeping DBP controlled.
    score += 18.0 * max(0.0, -crisis_sbp)
    score += 8.0 * max(0.0, -crisis_dbp)
    score += 9.0 * max(0.0, sparse_min - activation)
    score += 16.0 * max(0.0, activation - sparse_max)
    score += 5.5 * max(0.0, 0.12 - shift_sbp)
    if high_stage:
        score += 6.0 * max(0.0, -high_sbp)
    return float(score)


def _identity_row(rows: List[dict]) -> dict:
    return next((row for row in rows if str(row.get("candidate", "")) == "identity"), rows[0])


def _high_bias_row_score_v18(row: dict, cfg) -> float:
    return _tail_bias_score(row, cfg, _V17_HIGH_BIAS_SCORE(row, cfg), high_stage=True)


def _crisis_tail_row_score_v18(row: dict, cfg) -> float:
    return _tail_bias_score(row, cfg, _V17_CRISIS_TAIL_SCORE(row, cfg), high_stage=False)


def _sort_key_for_crisis(row: dict, cfg) -> tuple:
    mae = v17._row_float(row, "mae_mean")
    coverage = v17._row_float(row, "coverage_gap")
    high_sbp = v17._row_float(row, "high_bias_sbp")
    high_dbp = v17._row_float(row, "high_bias_dbp")
    crisis_sbp = v17._row_float(row, "crisis_bias_sbp")
    crisis_dbp = v17._row_float(row, "crisis_bias_dbp")
    clinical_under = v17._row_float(row, "clinical_under_penalty")
    tail_bias_pen = v17._row_float(row, "tail_bias_penalty")
    activation = max(v17._row_float(row, "activation_rate"), v17._row_float(row, "guard_activation_rate"))
    shift_sbp = max(v17._row_float(row, "shift_mean_sbp"), v17._row_float(row, "guard_shift_mean_sbp"))
    score = v17._row_float(row, "score")
    mae_target = _target(cfg, "REGRESSION_PARETO_MAE_TARGET", 5.55)
    coverage_target = _target(cfg, "REGRESSION_PARETO_COVERAGE_GAP_MAX", 0.14)
    crisis_sbp_target = _target(cfg, "CRISIS_BIAS_TARGET_ABS_SBP", 4.0)
    crisis_dbp_target = _target(cfg, "CRISIS_BIAS_TARGET_ABS_DBP", 3.5)
    high_sbp_target = _target(cfg, "HIGH_BIAS_TARGET_ABS_SBP", 3.5)
    high_dbp_target = _target(cfg, "HIGH_BIAS_TARGET_ABS_DBP", 3.0)
    sparse_min = _target(cfg, "CRISIS_REPAIR_MIN_ACTIVATION_RATE", 0.002)
    sparse_max = _target(cfg, "CRISIS_REPAIR_MAX_ACTIVATION_RATE", 0.24)
    shift_min = _target(cfg, "CRISIS_REPAIR_MIN_SBP_SHIFT", 0.12)
    feasible = (
        mae <= mae_target + 0.22
        and coverage <= coverage_target + 0.03
        and abs(crisis_sbp) <= crisis_sbp_target
        and abs(crisis_dbp) <= crisis_dbp_target + 0.75
        and abs(high_sbp) <= high_sbp_target + 1.25
        and abs(high_dbp) <= high_dbp_target + 1.25
        and activation <= sparse_max
    )
    return (
        0 if feasible else 1,
        max(0.0, sparse_min - activation),
        max(0.0, shift_min - shift_sbp),
        max(0.0, mae - mae_target),
        max(0.0, abs(crisis_sbp) - crisis_sbp_target),
        max(0.0, abs(crisis_dbp) - crisis_dbp_target),
        max(0.0, -crisis_sbp),
        abs(crisis_sbp) + 0.55 * abs(crisis_dbp),
        max(0.0, -high_sbp),
        abs(high_sbp) + 0.5 * abs(high_dbp),
        clinical_under,
        tail_bias_pen,
        coverage,
        score,
    )


def _select_regression_row_v18(rows: List[dict], cfg, prefer_crisis: bool = False) -> dict:
    if not rows:
        raise RuntimeError("No regression candidate rows were available.")
    if not prefer_crisis:
        return _V17_SELECT_REGRESSION_ROW(rows, cfg, prefer_crisis=False)
    selected = min(rows, key=lambda row: _sort_key_for_crisis(row, cfg))
    selected = dict(selected)
    identity = _identity_row(rows)
    selected["v10_18_identity_mae_mean"] = v17._row_float(identity, "mae_mean")
    selected["v10_18_identity_clinical_under_penalty"] = v17._row_float(identity, "clinical_under_penalty")
    selected["v10_18_selection_rule"] = "sparse_crisis_repair_with_v1011_regression_guard"
    return selected


def _prepend_unique(values: Iterable[str], existing: Iterable[str]) -> tuple[str, ...]:
    seen = set()
    result = []
    for value in list(values) + list(existing):
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


def build_nextgen_cfg():
    cfg = _V17_BUILD_NEXTGEN_CFG()
    v1011 = _load_v1011_targets()
    cfg.OUTPUT_NAME = FINAL_OUTPUT_NAME
    cfg.PROTOCOL_ID = FINAL_PROTOCOL_ID
    cfg.PROTOCOL_NAME = (
        "v10.18 subject-disjoint baseline-guarded crisis-repair protocol "
        "(v10.11 anchors + baseline classification gate + sparse exact crisis repair search)"
    )
    cfg.OPTLONG_FULLTRAIN_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
    cfg.DUALMAX_FULLTRAIN_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT
    cfg.FINAL_OUTPUT_NAME = FINAL_OUTPUT_NAME
    cfg.FINAL_PROTOCOL_ID = FINAL_PROTOCOL_ID

    cfg.WARMSTART_CANDIDATES = _prepend_unique(
        (
            "mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_proto",
            "mimic_bp_reg_v10_2_optlong_dualanchor_meta_stack_proto",
            "mimic_bp_reg_v10_17_subjectdisjoint_paretoordinal_crisiscal_proto",
        ),
        tuple(getattr(cfg, "WARMSTART_CANDIDATES", ())),
    )

    cfg.CLASSIFICATION_BASELINE_OUTPUT = "mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_proto"
    cfg.CLASSIFICATION_BASELINE_ACC = v1011["acc"]
    cfg.CLASSIFICATION_BASELINE_MACRO_F1 = v1011["macro_f1"]
    cfg.CLASSIFICATION_BASELINE_BALANCED_ACC = v1011["balanced_acc"]
    cfg.CLASSIFICATION_ASPIRATIONAL_ACC = 0.86
    cfg.CLASSIFICATION_ASPIRATIONAL_MACRO_F1 = 0.80
    cfg.CLASSIFICATION_BASELINE_MARGIN = 0.0
    cfg.BASELINE_ASPIRATIONAL_ACC = 0.86
    cfg.BASELINE_ASPIRATIONAL_F1 = 0.80
    cfg.BASELINE_MIN_ACC_MARGIN = 0.0
    cfg.BASELINE_MIN_F1_MARGIN = 0.0
    cfg.BASELINE_MIN_BAL_MARGIN = 0.0
    cfg.HEAD_SELECTION_RANK_MODE = "clean_acc_f1_then_score"
    cfg.HEAD_CLEAN_ACC_WEIGHT = max(float(getattr(cfg, "HEAD_CLEAN_ACC_WEIGHT", 0.0)), 22.0)
    cfg.HEAD_CLEAN_F1_WEIGHT = max(float(getattr(cfg, "HEAD_CLEAN_F1_WEIGHT", 0.0)), 6.0)
    cfg.HEAD_CLEAN_BALANCED_WEIGHT = max(float(getattr(cfg, "HEAD_CLEAN_BALANCED_WEIGHT", 0.0)), 1.2)
    cfg.HEAD_CLEAN_ROBUST_WEIGHT = max(float(getattr(cfg, "HEAD_CLEAN_ROBUST_WEIGHT", 0.0)), 0.15)
    cfg.HEAD_CLEAN_STAGE2_WEIGHT = max(float(getattr(cfg, "HEAD_CLEAN_STAGE2_WEIGHT", 0.0)), 0.15)

    cfg.REGRESSION_PARETO_MAE_TARGET = min(5.55, v1011["mae_mean"] - 0.35)
    cfg.REGRESSION_PARETO_COVERAGE_GAP_MAX = 0.14
    cfg.HIGH_BIAS_TARGET_ABS_SBP = 3.5
    cfg.HIGH_BIAS_TARGET_ABS_DBP = 3.0
    cfg.CRISIS_BIAS_TARGET_ABS_SBP = 4.0
    cfg.CRISIS_BIAS_TARGET_ABS_DBP = 3.5
    cfg.CRISIS_REPAIR_MIN_ACTIVATION_RATE = 0.002
    cfg.CRISIS_REPAIR_MAX_ACTIVATION_RATE = 0.22
    cfg.CRISIS_REPAIR_MIN_SBP_SHIFT = 0.12
    cfg.BASELINE_CRISIS_SBP_TARGET = 0.0
    cfg.BASELINE_CRISIS_DBP_TARGET = 0.0

    cfg.CRISIS_TAIL_FUSION_MAX_MAE_DELTA = 0.12
    cfg.CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA = 0.035
    cfg.CRISIS_TAIL_FUSION_HIGH_THRESHOLDS = (0.14, 0.22, 0.38, 0.55, 0.72)
    cfg.CRISIS_TAIL_FUSION_CRISIS_THRESHOLDS = (0.10, 0.18, 0.30, 0.45, 0.62, 0.78)
    cfg.CRISIS_TAIL_FUSION_GAMMAS = (0.85, 1.00, 1.25, 1.55)
    cfg.CRISIS_TAIL_FUSION_SBP_QUANTILES = (0.88, 0.92, 0.96, 0.985)
    cfg.CRISIS_TAIL_FUSION_DBP_QUANTILES = (0.84, 0.92, 0.975)
    cfg.CRISIS_TAIL_FUSION_CRISIS_GAINS = (0.35, 0.55, 0.80, 1.10, 1.55, 2.10)
    cfg.CRISIS_TAIL_FUSION_SBP_MARGINS = (0.0, 2.0, 4.0, 6.0, 9.0, 12.0, 15.0)
    cfg.CRISIS_TAIL_FUSION_DBP_MARGINS = (0.0, 1.5, 3.0, 4.5, 6.0)
    cfg.CRISIS_TAIL_FUSION_UNCERTAINTY_GAINS = (0.0, 0.15, 0.35, 0.60)
    cfg.CRISIS_TAIL_FUSION_MODEL_SCALES = (0.45, 0.65, 0.85, 1.05, 1.30)
    cfg.CRISIS_TAIL_FUSION_EXPERT_GAINS = (0.25, 0.45, 0.70, 1.00, 1.35)
    cfg.CRISIS_TAIL_HARD_FLOOR_SBP = 10.0
    cfg.CRISIS_TAIL_HARD_FLOOR_DBP = 3.5
    cfg.CRISIS_TAIL_MAX_SHIFT_SBP = 16.0
    cfg.CRISIS_TAIL_MAX_SHIFT_DBP = 7.0
    cfg.CRISIS_SBP_GUARD_TRIGGER = 0.68
    cfg.CRISIS_SBP_GUARD_MIN_EXPERT_SBP = 170.0
    cfg.CRISIS_SBP_GUARD_ABSOLUTE_FLOOR = 176.0
    cfg.CRISIS_DBP_GUARD_ABSOLUTE_FLOOR = 108.0
    cfg.CRISIS_SBP_GUARD_GAIN = 0.74
    cfg.CRISIS_DBP_GUARD_GAIN = 0.38

    cfg.VECTORFAST_EXACT_EXHAUSTIVE = True
    cfg.VECTORFAST_HIGH_BIAS_KEEP_ROWS = 24576
    cfg.VECTORFAST_CRISIS_KEEP_ROWS = 32768
    cfg.VECTORFAST_HIGH_BIAS_BATCH_SIZE = 4096
    cfg.VECTORFAST_CRISIS_BATCH_SIZE = 8192
    cfg.VECTORFAST_CACHE_FLUSH_EVERY = 64
    cfg.VECTORFAST_LAZY_EXACT_TOPK = True
    cfg.VECTORFAST_MASK_CACHE = True
    cfg.VECTORFAST_OOM_SAFE_SPLIT = True
    cfg.V10_17_SCORE_VERSION = V10_18_SCORE_VERSION
    cfg.V10_18_SCORE_VERSION = V10_18_SCORE_VERSION
    v17._ACTIVE_CFG = cfg
    out = _project_root() / "outputs" / FINAL_OUTPUT_NAME
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    return cfg


def _find_range(rows: List[dict], label: str) -> dict:
    label = label.lower()
    return next((row for row in rows if str(row.get("bp_range", row.get("class", ""))).lower() == label), {})


def generate_v1018_audit(output_dir: Path) -> None:
    root = _project_root() / "outputs"
    v1011_targets = _load_v1011_targets()
    final_results = _read_json(output_dir / "final_results.json")
    selected = _read_json(output_dir / "selected_strategy.json")
    bp_rows = _read_csv_rows(output_dir / "tables" / "bp_range_metrics.csv")
    v1011_rows = _read_csv_rows(
        root / "mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_proto" / "tables" / "bp_range_metrics.csv"
    )
    v1017_rows = _read_csv_rows(
        root / "mimic_bp_reg_v10_17_subjectdisjoint_paretoordinal_crisiscal_proto" / "tables" / "bp_range_metrics.csv"
    )
    crisis = _find_range(bp_rows, "crisis")
    high = _find_range(bp_rows, "high")
    crisis_1011 = _find_range(v1011_rows, "crisis")
    crisis_1017 = _find_range(v1017_rows, "crisis")
    metrics = final_results.get("test_selected", {})
    selected_acc = float(metrics.get("cls_acc_from_reg", metrics.get("acc", 0.0)))
    selected_f1 = float(metrics.get("cls_f1_macro_from_reg", metrics.get("macro_f1", 0.0)))
    selected_bal = float(metrics.get("cls_balanced_acc_from_reg", metrics.get("balanced_acc", 0.0)))
    audit = {
        "protocol_id": FINAL_PROTOCOL_ID,
        "score_version": V10_18_SCORE_VERSION,
        "selection_strategy": selected,
        "baseline_requirements": {
            "v10_11_acc": v1011_targets["acc"],
            "v10_11_macro_f1": v1011_targets["macro_f1"],
            "v10_11_balanced_acc": v1011_targets["balanced_acc"],
            "v10_11_mae_mean": v1011_targets["mae_mean"],
            "paper_target_acc": 0.86,
            "paper_target_macro_f1": 0.80,
            "crisis_abs_sbp_target": 4.0,
            "crisis_abs_dbp_target": 3.5,
        },
        "dominance_vs_v10_11": {
            "acc_delta": selected_acc - v1011_targets["acc"],
            "macro_f1_delta": selected_f1 - v1011_targets["macro_f1"],
            "balanced_acc_delta": selected_bal - v1011_targets["balanced_acc"],
            "mae_mean_delta": float(metrics.get("mae_mean", 0.0)) - v1011_targets["mae_mean"],
        },
        "selected_metrics": {
            "acc": selected_acc,
            "macro_f1": selected_f1,
            "balanced_acc": selected_bal,
            "mae_mean": float(metrics.get("mae_mean", 0.0)),
            "mae_sbp": float(metrics.get("mae_sbp", 0.0)),
            "mae_dbp": float(metrics.get("mae_dbp", 0.0)),
        },
        "tail_metrics": {
            "high": high,
            "crisis": crisis,
            "v10_11_crisis": crisis_1011,
            "v10_17_crisis": crisis_1017,
            "crisis_sample_count": int(float(crisis.get("n", 0) or 0)),
        },
        "paper_readiness_gate": bool(
            selected_acc >= v1011_targets["acc"]
            and selected_f1 >= v1011_targets["macro_f1"]
            and selected_bal >= v1011_targets["balanced_acc"] - 0.003
            and abs(v17._row_float(crisis, "bias_sbp")) <= 4.0
            and abs(v17._row_float(crisis, "bias_dbp")) <= 5.0
        ),
        "search_strategy": {
            "mode": "exact exhaustive traversal over the declared v10.18 candidate grid with vectorized scoring and cache resume; no within-grid heuristic candidate pruning",
            "anchor_change": "restores the stronger v10.11/v10.2 full-train checkpoints before applying v10.18 baseline-guarded selection",
            "acceleration": (
                "candidate batches are vectorized with enlarged v10.18 batch sizes; "
                "crisis-tail search uses exact lazy top-K scoring, first applying a safe lower-bound score "
                "and only running conformal calibration for candidates that can still enter the retained frontier; "
                "range-tail masks are cached across crisis-tail batches; "
                "top-row merging skips full frontier sorting unless a batch can enter the retained K; "
                "search cache writes are amortized over larger intervals to reduce disk I/O; "
                "OOM-safe recursive batch splitting preserves exact scoring under memory pressure; "
                "every generated candidate in the configured grid is still scored"
            ),
        },
        "notes": [
            "Classification scoring is baseline-first: candidates below v10.11 receive large penalties before aspirational 0.86/0.80 bonuses are considered.",
            "Crisis-tail selection is lexicographic: it requires sparse high-risk activation, then controls MAE/coverage, signed crisis underestimation, and high-range drift.",
            "The crisis bin count is reported explicitly; if n remains 1, the paper should discuss crisis-bin instability and also cite high/tail surrogate strata.",
        ],
    }
    _write_json(output_dir / "v10_18_baselineguard_crisisrepair_audit.json", audit)


def generate_extra_outputs(output_dir: Path) -> None:
    v17._ORIG_V16_GENERATE_EXTRA_OUTPUTS(output_dir)
    generate_v1018_audit(output_dir)


def main() -> None:
    global _CRISIS_TAIL_SCORE_CUTOFF
    _CRISIS_TAIL_SCORE_CUTOFF = None
    _CRISIS_TAIL_MASK_CACHE.clear()
    originals = {
        "final_output": v17.FINAL_OUTPUT_NAME,
        "protocol_id": v17.FINAL_PROTOCOL_ID,
        "optlong": v17.OPTLONG_FULLTRAIN_OUTPUT,
        "dualmax": v17.DUALMAX_FULLTRAIN_OUTPUT,
        "score_version": v17.V10_17_SCORE_VERSION,
        "build": v17.build_nextgen_cfg,
        "extra": v17.generate_extra_outputs,
        "class_score": v17._classification_pareto_score,
        "class_score_batch": v17._classification_pareto_score_batch,
        "class_gate": v17._classification_gate,
        "high_score": v17._high_bias_row_score,
        "crisis_score": v17._crisis_tail_row_score,
        "select_reg": v17._select_regression_row,
        "feature_head_resume": v17.v16.meta_script.prev_script.run_feature_head_resume,
        "v16_adaptive_batch": v17.v16._adaptive_batch_size,
        "v16_merge_top_rows": v17.v16._merge_top_rows,
        "v16_vector_high": v17.v16._vectorized_high_bias_rows,
        "v16_vector_crisis": v17.v16._vectorized_crisis_tail_rows,
    }
    try:
        v17.FINAL_OUTPUT_NAME = FINAL_OUTPUT_NAME
        v17.FINAL_PROTOCOL_ID = FINAL_PROTOCOL_ID
        v17.OPTLONG_FULLTRAIN_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
        v17.DUALMAX_FULLTRAIN_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT
        v17.V10_17_SCORE_VERSION = V10_18_SCORE_VERSION
        v17.build_nextgen_cfg = build_nextgen_cfg
        v17.generate_extra_outputs = generate_extra_outputs
        v17._classification_pareto_score = _classification_pareto_score_v18
        v17._classification_pareto_score_batch = _classification_pareto_score_batch_v18
        v17._classification_gate = _classification_gate_v18
        v17._high_bias_row_score = _high_bias_row_score_v18
        v17._crisis_tail_row_score = _crisis_tail_row_score_v18
        v17._select_regression_row = _select_regression_row_v18
        v17.v16.meta_script.prev_script.run_feature_head_resume = _cached_or_train_feature_head
        v17.v16._adaptive_batch_size = _vectorfast_batch_size_v18
        v17.v16._merge_top_rows = _merge_top_rows_v18
        v17.v16._vectorized_high_bias_rows = _vectorized_high_bias_rows_v18
        v17.v16._vectorized_crisis_tail_rows = _vectorized_crisis_tail_rows_v18
        v17.main()
    finally:
        v17.FINAL_OUTPUT_NAME = originals["final_output"]
        v17.FINAL_PROTOCOL_ID = originals["protocol_id"]
        v17.OPTLONG_FULLTRAIN_OUTPUT = originals["optlong"]
        v17.DUALMAX_FULLTRAIN_OUTPUT = originals["dualmax"]
        v17.V10_17_SCORE_VERSION = originals["score_version"]
        v17.build_nextgen_cfg = originals["build"]
        v17.generate_extra_outputs = originals["extra"]
        v17._classification_pareto_score = originals["class_score"]
        v17._classification_pareto_score_batch = originals["class_score_batch"]
        v17._classification_gate = originals["class_gate"]
        v17._high_bias_row_score = originals["high_score"]
        v17._crisis_tail_row_score = originals["crisis_score"]
        v17._select_regression_row = originals["select_reg"]
        v17.v16.meta_script.prev_script.run_feature_head_resume = originals["feature_head_resume"]
        v17.v16._adaptive_batch_size = originals["v16_adaptive_batch"]
        v17.v16._merge_top_rows = originals["v16_merge_top_rows"]
        v17.v16._vectorized_high_bias_rows = originals["v16_vector_high"]
        v17.v16._vectorized_crisis_tail_rows = originals["v16_vector_crisis"]


if __name__ == "__main__":
    main()
