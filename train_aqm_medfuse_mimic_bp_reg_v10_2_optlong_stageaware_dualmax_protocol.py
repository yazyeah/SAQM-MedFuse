from __future__ import annotations

import copy
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from aqm_bp_shared_v10_2_multi_protocol import build_protocol_loaders, make_v10_2_optimized_long_cfg
from aqm_bp_shared_v10_2_protocol import (
    ModelEMA,
    QMoERegressionNet,
    build_bp_range_table,
    build_calibration_curve_table,
    build_class_weights,
    build_conditional_coverage_table,
    build_range_weights,
    build_subjectwise_error_table,
    collect_outputs_regression,
    conformal_from_outputs,
    ensure_out_dirs,
    measure_runtime,
    move_batch,
    plot_bland_altman,
    plot_calibration,
    plot_confusion,
    plot_noise_robustness,
    plot_quality_conditional_coverage,
    plot_roc_pr,
    plot_router_heatmap,
    plot_scatter_true_vs_pred,
    plot_sharpness_vs_coverage,
    plot_uncertainty_error_corr,
    regression_metrics,
    risk_classification_metrics,
    save_epoch_log,
    save_json,
    save_regression_npz,
    save_rows_csv,
    seed_everything,
    set_warmup_cosine_lr,
)
from train_aqm_medfuse_mimic_bp_reg_v10_2_common import (
    _load_regression_resume_state,
    _resume_enabled,
    _resume_state_path,
    _save_regression_resume_state,
    _torch_load_checkpoint,
    apply_subject_calibration,
    build_error_cdf_rows,
    compute_regression_loss,
    evaluate_personalized_validation,
    build_paper_metrics,
    build_split_distribution_rows,
    build_subject_gain_table,
    fit_subject_calibration_state,
    plot_bp_range_bias,
    plot_error_cdf,
    plot_few_shot_curve,
    plot_missing_modality_curve,
    plot_split_class_distribution,
    plot_subject_calibration_gain,
    plot_training_curves_reg_split,
)
from train_aqm_medfuse_mimic_bp_reg_v10_2_hybrid_protocol import (
    build_hybrid_features,
    clone_regression_output,
    predict_proba_full,
)


def clipped_regression_prediction(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32).copy()
    pred[:, 0] = np.clip(pred[:, 0], 70.0, 200.0)
    pred[:, 1] = np.clip(pred[:, 1], 35.0, 130.0)
    return pred


def pick_optlong_checkpoint(cfg) -> Path:
    candidates = [
        Path(cfg.PROJECT_ROOT) / "outputs" / "mimic_bp_reg_v10_2_opt_long_proto" / "best_model_pre_stageaware_2026-03-28.pt",
        Path(cfg.PROJECT_ROOT) / "outputs" / "mimic_bp_reg_v10_2_opt_long_proto" / "best_model_epoch050_backup_2026-03-26.pt",
        Path(cfg.PROJECT_ROOT) / "outputs" / "mimic_bp_reg_v10_2_opt_long_proto" / "best_model.pt",
        Path(cfg.PROJECT_ROOT) / "outputs" / "mimic_bp_reg_v10_2_opt_long_proto" / "mimic_bp_reg_v10_2_opt_long_proto_best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No usable opt-long checkpoint found for stage-aware dual optimization.")


def stageaware_feature_names() -> List[str]:
    base_names = [
        "raw_sbp",
        "raw_dbp",
        "corr_sbp",
        "corr_dbp",
        "delta_sbp",
        "delta_dbp",
        "scale_sbp",
        "scale_dbp",
        "offset_sbp",
        "offset_dbp",
        "map_corr",
        "pp_corr",
        "ratio_corr",
        "uncertainty",
        "quality",
        "alpha_ppg",
        "alpha_ecg",
        "alpha_joint",
        "alpha_cross",
        "cred_ppg",
        "cred_ecg",
        "cred_fused",
    ]
    prob_names = [
        "raw_p_normal",
        "raw_p_elevated",
        "raw_p_stage1",
        "raw_p_stage2",
        "corr_p_normal",
        "corr_p_elevated",
        "corr_p_stage1",
        "corr_p_stage2",
        "raw_top1",
        "raw_margin",
        "corr_top1",
        "corr_margin",
        "raw_cls_normal",
        "raw_cls_elevated",
        "raw_cls_stage1",
        "raw_cls_stage2",
        "corr_cls_normal",
        "corr_cls_elevated",
        "corr_cls_stage1",
        "corr_cls_stage2",
    ]
    return base_names + prob_names


def build_stageaware_features(raw_out: dict, corr_out: dict, cfg) -> np.ndarray:
    base = build_hybrid_features(raw_out, corr_out)
    raw_prob = np.asarray(raw_out["y_prob_cls_from_reg"], dtype=np.float32)
    corr_prob = np.asarray(corr_out["y_prob_cls_from_reg"], dtype=np.float32)

    raw_sorted = np.sort(raw_prob, axis=1)
    corr_sorted = np.sort(corr_prob, axis=1)
    raw_stats = np.stack([raw_sorted[:, -1], raw_sorted[:, -1] - raw_sorted[:, -2]], axis=1)
    corr_stats = np.stack([corr_sorted[:, -1], corr_sorted[:, -1] - corr_sorted[:, -2]], axis=1)

    raw_pred_cls = np.asarray(raw_out["y_pred_cls_from_reg"], dtype=np.int64)
    corr_pred_cls = np.asarray(corr_out["y_pred_cls_from_reg"], dtype=np.int64)
    raw_onehot = np.eye(cfg.N_CLASSES, dtype=np.float32)[raw_pred_cls]
    corr_onehot = np.eye(cfg.N_CLASSES, dtype=np.float32)[corr_pred_cls]

    return np.concatenate(
        [
            base,
            raw_prob,
            corr_prob,
            raw_stats.astype(np.float32),
            corr_stats.astype(np.float32),
            raw_onehot,
            corr_onehot,
        ],
        axis=1,
    ).astype(np.float32)


def fit_regressor_ensemble(x: np.ndarray, y_residual: np.ndarray, seed: int):
    models = [
        RandomForestRegressor(
            n_estimators=500,
            max_depth=16,
            min_samples_leaf=6,
            random_state=seed,
            n_jobs=-1,
        ),
        ExtraTreesRegressor(
            n_estimators=700,
            max_depth=18,
            min_samples_leaf=4,
            random_state=seed + 17,
            n_jobs=-1,
        ),
    ]
    for model in models:
        model.fit(x, y_residual)
    return models


def predict_regressor_ensemble(models, x: np.ndarray) -> np.ndarray:
    preds = [np.asarray(model.predict(x), dtype=np.float32) for model in models]
    return np.mean(preds, axis=0).astype(np.float32)


def fit_classifier_ensemble(x: np.ndarray, y_cls: np.ndarray, seed: int):
    models = [
        RandomForestClassifier(
            n_estimators=700,
            max_depth=16,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        ExtraTreesClassifier(
            n_estimators=900,
            max_depth=20,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed + 29,
            n_jobs=-1,
        ),
    ]
    for model in models:
        model.fit(x, y_cls)
    return models


def predict_classifier_ensemble(models, x: np.ndarray, n_classes: int) -> np.ndarray:
    probs = [predict_proba_full(model, x, n_classes) for model in models]
    prob = np.mean(probs, axis=0).astype(np.float32)
    return prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-8, None)


def feature_importance_rows(models) -> List[dict]:
    names = stageaware_feature_names()
    importances = np.mean([np.asarray(model.feature_importances_, dtype=np.float32) for model in models], axis=0)
    rows = [{"feature": name, "importance": float(score)} for name, score in zip(names, importances)]
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows


def build_reg_prob_sources(raw_out: dict, cal_out: dict, corr_out: dict) -> Dict[str, np.ndarray]:
    raw = np.asarray(raw_out["y_prob_cls_from_reg"], dtype=np.float32)
    cal = np.asarray(cal_out["y_prob_cls_from_reg"], dtype=np.float32)
    corr = np.asarray(corr_out["y_prob_cls_from_reg"], dtype=np.float32)
    return {
        "raw": raw,
        "corr": corr,
        "cal": cal,
        "raw_corr_avg": 0.5 * (raw + corr),
    }


def fit_stage_residual_centroids(out: dict, cfg, prior_count: float = 48.0):
    residual = np.asarray(out["y_true_reg"], dtype=np.float32) - np.asarray(out["y_pred_reg"], dtype=np.float32)
    y_true_cls = np.asarray(out["y_true_cls"], dtype=np.int64)
    global_mean = residual.mean(axis=0)
    centroids = []
    rows = []
    for cls_idx, cls_name in enumerate(cfg.CLASS_NAMES):
        mask = y_true_cls == cls_idx
        n = int(mask.sum())
        class_mean = residual[mask].mean(axis=0) if n > 0 else global_mean
        shrink = (n * class_mean + prior_count * global_mean) / max(n + prior_count, 1.0)
        shrink = np.asarray(shrink, dtype=np.float32)
        shrink[0] = float(np.clip(shrink[0], -12.0, 16.0))
        shrink[1] = float(np.clip(shrink[1], -8.0, 10.0))
        centroids.append(shrink)
        rows.append(
            {
                "class_name": cls_name,
                "n": n,
                "residual_mean_sbp": float(class_mean[0]),
                "residual_mean_dbp": float(class_mean[1]),
                "shrunken_shift_sbp": float(shrink[0]),
                "shrunken_shift_dbp": float(shrink[1]),
            }
        )
    return np.asarray(centroids, dtype=np.float32), rows


def apply_stageaware_shift(base_pred_reg: np.ndarray, hybrid_prob: np.ndarray, centroids: np.ndarray, shift_scale: float) -> np.ndarray:
    soft_shift = np.asarray(hybrid_prob, dtype=np.float32) @ np.asarray(centroids, dtype=np.float32)
    pred = np.asarray(base_pred_reg, dtype=np.float32) + float(shift_scale) * soft_shift
    return clipped_regression_prediction(pred)


def apply_prob_policy(
    reg_prob: np.ndarray,
    meta_prob: np.ndarray,
    meta_blend_weight: float,
    gamma: float,
    class_weights: Sequence[float],
) -> np.ndarray:
    prob = (1.0 - float(meta_blend_weight)) * np.asarray(reg_prob, dtype=np.float32) + float(meta_blend_weight) * np.asarray(meta_prob, dtype=np.float32)
    prob = np.clip(prob, 1e-8, None)
    prob = np.power(prob, float(gamma), dtype=np.float32)
    prob = prob * np.asarray(class_weights, dtype=np.float32).reshape(1, -1)
    prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-8, None)
    return prob.astype(np.float32)


def summarize_conformal_tradeoff(calib_out: dict, query_out: dict, cfg) -> Dict[str, float]:
    _, _, conformal = conformal_from_outputs(calib_out, query_out, alpha=cfg.CONFORMAL_ALPHA)
    target_coverage = 1.0 - float(cfg.CONFORMAL_ALPHA)
    coverage_gap = 0.5 * (
        abs(float(conformal["coverage_sbp"]) - target_coverage)
        + abs(float(conformal["coverage_dbp"]) - target_coverage)
    )
    miw_mean = 0.5 * (float(conformal["miw_sbp"]) + float(conformal["miw_dbp"]))
    return {
        "coverage_target": float(target_coverage),
        "coverage_sbp": float(conformal["coverage_sbp"]),
        "coverage_dbp": float(conformal["coverage_dbp"]),
        "miw_sbp": float(conformal["miw_sbp"]),
        "miw_dbp": float(conformal["miw_dbp"]),
        "coverage_gap": float(coverage_gap),
        "miw_mean": float(miw_mean),
    }


def policy_score(
    cls_metrics: Dict[str, float],
    reg_metrics: Dict[str, float],
    conformal_metrics: Dict[str, float],
    baseline_reg_metrics: Dict[str, float],
) -> float:
    macro_f1 = float(cls_metrics["cls_f1_macro_stageaware_val"])
    bal_acc = float(cls_metrics["cls_balanced_acc_stageaware_val"])
    elevated_f1 = float(cls_metrics.get("cls_f1_stageaware_val_Elevated", 0.0))
    stage1_f1 = float(cls_metrics.get("cls_f1_stageaware_val_Stage1", 0.0))
    stage2_f1 = float(cls_metrics.get("cls_f1_stageaware_val_Stage2", 0.0))
    rare_f1_mean = (elevated_f1 + stage1_f1 + stage2_f1) / 3.0
    rare_f1_min = min(elevated_f1, stage1_f1, stage2_f1)

    mae_mean = float(reg_metrics["mae_mean"])
    mae_drift = max(0.0, mae_mean - float(baseline_reg_metrics["mae_mean"]))
    sbp_drift = max(0.0, float(reg_metrics["mae_sbp"]) - float(baseline_reg_metrics["mae_sbp"]))
    dbp_drift = max(0.0, float(reg_metrics["mae_dbp"]) - float(baseline_reg_metrics["mae_dbp"]))
    bias_penalty = 0.010 * (
        abs(float(reg_metrics["bias_sbp"])) + abs(float(reg_metrics["bias_dbp"]))
    )
    ece_penalty = 0.15 * float(cls_metrics.get("cls_ece_stageaware_val", 0.0))

    # Cost is easier to reason about here: keep regression close to the corrected
    # baseline, reward stage-aware class separation, and reject policies that win
    # by collapsing conformal coverage too aggressively.
    cost = (
        mae_mean
        + 0.60 * mae_drift
        + 0.18 * (sbp_drift + dbp_drift)
        + 1.60 * float(conformal_metrics["coverage_gap"])
        + 0.018 * float(conformal_metrics["miw_mean"])
        + bias_penalty
        + ece_penalty
        - 1.00 * macro_f1
        - 0.45 * bal_acc
        - 0.20 * rare_f1_mean
        - 0.08 * rare_f1_min
    )
    return -float(cost)


def evaluate_stageaware_policy(
    query_corr: dict,
    calib_corr: dict,
    query_reg_prob_sources: Dict[str, np.ndarray],
    calib_reg_prob_sources: Dict[str, np.ndarray],
    query_meta_prob: np.ndarray,
    calib_meta_prob: np.ndarray,
    centroids: np.ndarray,
    reg_source_name: str,
    meta_blend_weight: float,
    gamma: float,
    class_weights: Sequence[float],
    shift_scale: float,
    cfg,
):
    query_hybrid_prob = apply_prob_policy(
        query_reg_prob_sources[reg_source_name],
        query_meta_prob,
        meta_blend_weight,
        gamma,
        class_weights,
    )
    query_hybrid_pred = query_hybrid_prob.argmax(axis=1).astype(np.int64)
    query_stage_pred_reg = apply_stageaware_shift(
        np.asarray(query_corr["y_pred_reg"], dtype=np.float32),
        query_hybrid_prob,
        centroids,
        shift_scale,
    )
    query_stageaware = clone_regression_output(query_corr, query_stage_pred_reg, cfg)
    reg_metrics = query_stageaware["metrics_reg"]
    cls_metrics = risk_classification_metrics(
        np.asarray(query_stageaware["y_true_cls"], dtype=np.int64),
        query_hybrid_pred,
        query_hybrid_prob,
        cfg,
        prefix="stageaware_val",
    )

    calib_hybrid_prob = apply_prob_policy(
        calib_reg_prob_sources[reg_source_name],
        calib_meta_prob,
        meta_blend_weight,
        gamma,
        class_weights,
    )
    calib_stage_pred_reg = apply_stageaware_shift(
        np.asarray(calib_corr["y_pred_reg"], dtype=np.float32),
        calib_hybrid_prob,
        centroids,
        shift_scale,
    )
    calib_stageaware = clone_regression_output(calib_corr, calib_stage_pred_reg, cfg)
    conformal_metrics = summarize_conformal_tradeoff(calib_stageaware, query_stageaware, cfg)
    score = policy_score(cls_metrics, reg_metrics, conformal_metrics, query_corr["metrics_reg"])

    elevated_f1 = float(cls_metrics.get("cls_f1_stageaware_val_Elevated", 0.0))
    stage1_f1 = float(cls_metrics.get("cls_f1_stageaware_val_Stage1", 0.0))
    stage2_f1 = float(cls_metrics.get("cls_f1_stageaware_val_Stage2", 0.0))
    rare_f1_mean = (elevated_f1 + stage1_f1 + stage2_f1) / 3.0
    mae_drift = max(0.0, float(reg_metrics["mae_mean"]) - float(query_corr["metrics_reg"]["mae_mean"]))
    row = {
        "reg_source": reg_source_name,
        "meta_blend_weight": float(meta_blend_weight),
        "gamma": float(gamma),
        "shift_scale": float(shift_scale),
        "w_normal": float(class_weights[0]),
        "w_elevated": float(class_weights[1]),
        "w_stage1": float(class_weights[2]),
        "w_stage2": float(class_weights[3]),
        "score": float(score),
        "mae_mean": float(reg_metrics["mae_mean"]),
        "mae_sbp": float(reg_metrics["mae_sbp"]),
        "mae_dbp": float(reg_metrics["mae_dbp"]),
        "mae_drift_vs_corrected": float(mae_drift),
        "bias_sbp": float(reg_metrics["bias_sbp"]),
        "bias_dbp": float(reg_metrics["bias_dbp"]),
        "rare_f1_mean": float(rare_f1_mean),
        **conformal_metrics,
        **cls_metrics,
    }
    return (
        query_hybrid_prob,
        query_hybrid_pred,
        query_stageaware,
        calib_stageaware,
        reg_metrics,
        cls_metrics,
        conformal_metrics,
        row,
    )


def tune_stageaware_policy(
    query_corr: dict,
    calib_corr: dict,
    query_reg_prob_sources: Dict[str, np.ndarray],
    calib_reg_prob_sources: Dict[str, np.ndarray],
    query_meta_prob: np.ndarray,
    calib_meta_prob: np.ndarray,
    centroids: np.ndarray,
    cfg,
):
    coarse_rows: List[dict] = []
    best_row = None
    best_state = None
    base_weights = (1.0, 1.25, 1.45, 1.75)

    for reg_source in ("raw_corr_avg", "corr", "cal", "raw"):
        for meta_blend_weight in (0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0):
            for gamma in (0.85, 1.00, 1.12, 1.25, 1.40):
                for shift_scale in (0.0, 0.15, 0.30, 0.45, 0.60):
                    state = evaluate_stageaware_policy(
                        query_corr,
                        calib_corr,
                        query_reg_prob_sources,
                        calib_reg_prob_sources,
                        query_meta_prob,
                        calib_meta_prob,
                        centroids,
                        reg_source,
                        meta_blend_weight,
                        gamma,
                        base_weights,
                        shift_scale,
                        cfg,
                    )
                    row = {"stage": "coarse", **state[-1]}
                    coarse_rows.append(row)
                    if best_row is None or row["score"] > best_row["score"]:
                        best_row = row
                        best_state = state

    assert best_row is not None and best_state is not None
    fine_rows: List[dict] = []
    best_reg_source = str(best_row["reg_source"])
    best_blend = float(best_row["meta_blend_weight"])
    best_gamma = float(best_row["gamma"])
    best_shift = float(best_row["shift_scale"])

    fine_blends = sorted({round(float(np.clip(best_blend + delta, 0.0, 1.0)), 3) for delta in (-0.10, 0.0, 0.10)})
    fine_gammas = sorted({round(max(0.75, best_gamma + delta), 3) for delta in (-0.10, 0.0, 0.10)})
    fine_shift = sorted({round(max(0.0, best_shift + delta), 3) for delta in (-0.15, 0.0, 0.15)})
    elev_weights = sorted({round(max(1.0, base_weights[1] * scale), 3) for scale in (0.90, 1.0, 1.15)})
    stage1_weights = sorted({round(max(1.0, base_weights[2] * scale), 3) for scale in (0.90, 1.0, 1.15)})
    stage2_weights = sorted({round(max(1.0, base_weights[3] * scale), 3) for scale in (0.90, 1.0, 1.15)})

    for meta_blend_weight in fine_blends:
        for gamma in fine_gammas:
            for shift_scale in fine_shift:
                for elevated_w, stage1_w, stage2_w in product(elev_weights, stage1_weights, stage2_weights):
                    weights = (1.0, elevated_w, stage1_w, stage2_w)
                    state = evaluate_stageaware_policy(
                        query_corr,
                        calib_corr,
                        query_reg_prob_sources,
                        calib_reg_prob_sources,
                        query_meta_prob,
                        calib_meta_prob,
                        centroids,
                        best_reg_source,
                        meta_blend_weight,
                        gamma,
                        weights,
                        shift_scale,
                        cfg,
                    )
                    row = {"stage": "fine", **state[-1]}
                    fine_rows.append(row)
                    if row["score"] > best_row["score"]:
                        best_row = row
                        best_state = state

    best_prob, best_pred, best_stageaware, best_calib_stageaware, best_reg_metrics, best_cls_metrics, best_conformal_metrics, _ = best_state
    policy = {
        "reg_source": str(best_row["reg_source"]),
        "meta_blend_weight": float(best_row["meta_blend_weight"]),
        "gamma": float(best_row["gamma"]),
        "shift_scale": float(best_row["shift_scale"]),
        "class_weights": [
            float(best_row["w_normal"]),
            float(best_row["w_elevated"]),
            float(best_row["w_stage1"]),
            float(best_row["w_stage2"]),
        ],
        "score": float(best_row["score"]),
        "val_reg_metrics": best_reg_metrics,
        "val_cls_metrics": best_cls_metrics,
        "val_conformal_metrics": best_conformal_metrics,
    }
    return policy, coarse_rows + fine_rows, best_prob, best_pred, best_stageaware, best_calib_stageaware


def apply_stageaware_policy_to_test(
    base_pred_reg: np.ndarray,
    reg_prob_sources: Dict[str, np.ndarray],
    meta_prob: np.ndarray,
    centroids: np.ndarray,
    policy: dict,
):
    hybrid_prob = apply_prob_policy(
        reg_prob_sources[str(policy["reg_source"])],
        meta_prob,
        float(policy["meta_blend_weight"]),
        float(policy["gamma"]),
        policy["class_weights"],
    )
    hybrid_pred = hybrid_prob.argmax(axis=1).astype(np.int64)
    stage_pred_reg = apply_stageaware_shift(base_pred_reg, hybrid_prob, centroids, float(policy["shift_scale"]))
    return hybrid_prob, hybrid_pred, stage_pred_reg


def build_stageaware_cfg():
    cfg = make_v10_2_optimized_long_cfg()
    cfg.OUTPUT_NAME = "mimic_bp_reg_v10_2_optlong_stageaware_dualmax_proto"
    cfg.PROTOCOL_ID = "v10.2_optlong_stageaware_dualmax"
    cfg.PROTOCOL_NAME = "v10.2 opt-long stage-aware dual optimization with warm-start fine-tuning"
    cfg.EPOCHS = 80
    cfg.EARLY_STOPPING_PATIENCE = max(int(cfg.EARLY_STOPPING_PATIENCE), int(cfg.EPOCHS))
    cfg.LR = min(float(cfg.LR), 2.0e-5)
    cfg.PLOT_COMPOSITE_TRAINING_CURVES = False
    cfg.SAVE_SPLIT_TRAINING_CURVES = True
    return cfg


def calibration_support_indices(out: dict, n_shots: int | None) -> np.ndarray:
    n_rows = len(out["subject_ids"])
    if n_shots is None or int(n_shots) <= 0:
        return np.arange(n_rows, dtype=np.int64)

    subject_ids = np.asarray(out["subject_ids"], dtype=object)
    seg_indices = np.asarray(out["seg_indices"], dtype=np.int64)
    rows: List[int] = []
    for sid in sorted(set(subject_ids.tolist())):
        idx = np.where(subject_ids == sid)[0]
        if idx.size == 0:
            continue
        idx = idx[np.argsort(seg_indices[idx])]
        rows.extend(idx[: min(int(n_shots), idx.size)].tolist())
    return np.asarray(rows, dtype=np.int64)


def subset_regression_output(out: dict, row_idx: np.ndarray, cfg) -> dict:
    n_rows = len(out["subject_ids"])
    idx = np.asarray(row_idx, dtype=np.int64)
    if idx.size == 0:
        idx = np.arange(n_rows, dtype=np.int64)

    subset = {}
    for key, value in out.items():
        if key in {"metrics_reg", "metrics_cls_from_reg", "metrics_cls_from_reg_hard", "uncertainty_metrics"}:
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n_rows:
            subset[key] = value[idx]
        elif isinstance(value, list) and len(value) == n_rows:
            subset[key] = [value[i] for i in idx.tolist()]
        else:
            subset[key] = value
    return clone_regression_output(subset, np.asarray(subset["y_pred_reg"], dtype=np.float32), cfg)


def train_stageaware_backbone(cfg, loaders, out_root: Path, ckpt_path: Path):
    ds_train = loaders.ds_train
    range_weights = build_range_weights(ds_train, cfg, cfg.DEVICE)
    class_weights = build_class_weights(ds_train, cfg, cfg.DEVICE)
    resume_path = _resume_state_path(out_root, cfg)
    resume_state = _load_regression_resume_state(resume_path, cfg, cfg.DEVICE) if _resume_enabled(cfg) else None

    model = QMoERegressionNet(cfg).to(cfg.DEVICE)
    fallback_resume_ckpt_text = str(getattr(cfg, "RESUME_FALLBACK_CKPT_PATH", "") or "").strip()
    if resume_state is None and fallback_resume_ckpt_text:
        fallback_resume_ckpt = Path(fallback_resume_ckpt_text)
        if fallback_resume_ckpt.exists():
            ckpt_path = fallback_resume_ckpt
    state = _torch_load_checkpoint(ckpt_path, map_location=cfg.DEVICE)
    model.load_state_dict(state, strict=False)
    print(f"Warm-started model from: {ckpt_path}")

    use_ema = bool(getattr(cfg, "USE_EMA", True))
    ema = ModelEMA(model, decay=cfg.EMA_DECAY) if use_ema else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    epoch_logs: List[dict] = []
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    start_epoch = 1
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"], strict=False)
        if ema is not None and resume_state.get("ema_state") is not None:
            ema.module.load_state_dict(resume_state["ema_state"], strict=False)
        elif ema is not None:
            ema.module.load_state_dict(model.state_dict(), strict=False)
        if resume_state.get("optimizer_state") is not None:
            optimizer.load_state_dict(resume_state["optimizer_state"])
        epoch_logs = list(resume_state.get("epoch_logs", []))
        best_score = float(resume_state.get("best_score", best_score))
        best_epoch = int(resume_state.get("best_epoch", best_epoch))
        best_state = resume_state.get("best_state", best_state)
        patience = int(resume_state.get("patience", patience))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        print(f"Resuming {cfg.PROTOCOL_NAME} from epoch {start_epoch} using: {resume_path}")

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        model.train()
        lr = set_warmup_cosine_lr(optimizer, cfg, epoch)
        train_loss_total, train_grad_norms = [], []
        loss_buckets = {
            k: []
            for k in [
                "loss_reg", "loss_nll", "loss_q", "loss_router", "loss_bal",
                "loss_ppg_aux", "loss_ecg_aux", "loss_reg_cls", "loss_cred", "loss_tail", "loss_crisis_tail",
            ]
        }

        for batch in loaders.train_loader:
            batch = move_batch(batch, cfg.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            total_loss, logs = compute_regression_loss(model, batch, range_weights, class_weights, cfg, epoch)
            total_loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            train_loss_total.append(float(total_loss.item()))
            train_grad_norms.append(float(grad_norm))
            for key, value in logs.items():
                loss_buckets.setdefault(key, []).append(value)

        eval_model = ema.module if ema is not None else model
        val_row, _ = evaluate_personalized_validation(eval_model, loaders.val_query_loader, loaders.val_calib_loader, cfg)
        row = {
            "epoch": epoch,
            "lr": float(lr),
            "train_loss": float(np.mean(train_loss_total)),
            "grad_norm": float(np.mean(train_grad_norms)),
            **{k: (float(np.mean(v)) if v else float("nan")) for k, v in loss_buckets.items()},
            **val_row,
        }
        epoch_logs.append(row)
        stage_name = str(getattr(cfg, "TRAIN_STAGE_NAME", "StageAware Backbone"))
        print(
            f"[{stage_name} Epoch {epoch:03d}] train_loss={row['train_loss']:.4f} | "
            f"chosen_mae_sbp={row['mae_sbp']:.3f} | chosen_mae_dbp={row['mae_dbp']:.3f} | "
            f"backbone_f1_from_reg(chosen)={row['cls_f1_macro_from_reg']:.3f} | "
            f"backbone_f1_from_reg(raw)={row.get('raw_cls_f1_macro_from_reg', row['cls_f1_macro_from_reg']):.3f} | "
            f"backbone_f1_from_reg(cal)={row.get('cal_cls_f1_macro_from_reg', row['cls_f1_macro_from_reg']):.3f} | "
            f"val_score(lower=better)={row['val_score']:.3f}"
        )

        if row["val_score"] < best_score:
            best_score = float(row["val_score"])
            best_epoch = epoch
            best_state = copy.deepcopy(eval_model.state_dict())
            patience = 0
            out_root.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, out_root / "best_model.pt")
            torch.save(best_state, out_root / f"{cfg.OUTPUT_NAME}_best.pt")
        else:
            patience += 1
        if _resume_enabled(cfg):
            _save_regression_resume_state(
                resume_path,
                cfg,
                epoch,
                model,
                ema,
                optimizer,
                epoch_logs,
                best_score,
                best_epoch,
                best_state,
                patience,
            )
        if patience >= cfg.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state is None:
        best_path = out_root / "best_model.pt"
        if best_path.exists():
            best_state = _torch_load_checkpoint(best_path, map_location=cfg.DEVICE)
            print(f"Loaded existing best checkpoint for finalization: {best_path}")
        else:
            raise RuntimeError("No best checkpoint was recorded during stage-aware warm-start fine-tuning.")

    model.load_state_dict(best_state)
    model.eval()
    out_root.mkdir(parents=True, exist_ok=True)
    save_epoch_log(out_root / "epoch_log.csv", epoch_logs)
    return model, epoch_logs, {
        "epochs_target": int(cfg.EPOCHS),
        "epochs_ran": int(len(epoch_logs)),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "use_ema": bool(use_ema),
    }


def fit_stageaware_bundle(
    calib_raw: dict,
    query_raw: dict,
    cfg,
    seed: int,
    prefix: str,
    policy: dict | None = None,
    tune_policy: bool = False,
    n_shots: int | None = None,
):
    calib_state = fit_subject_calibration_state(calib_raw, cfg, n_shots=n_shots)
    calib_support_idx = calibration_support_indices(calib_raw, n_shots)
    calib_cal_full = apply_subject_calibration(calib_raw, calib_state, cfg)
    query_cal = apply_subject_calibration(query_raw, calib_state, cfg)

    calib_raw_support = subset_regression_output(calib_raw, calib_support_idx, cfg)
    calib_cal_support = subset_regression_output(calib_cal_full, calib_support_idx, cfg)

    x_calib = build_stageaware_features(calib_raw_support, calib_cal_support, cfg)
    y_calib_resid = np.asarray(calib_cal_support["y_true_reg"], dtype=np.float32) - np.asarray(calib_cal_support["y_pred_reg"], dtype=np.float32)
    y_calib_cls = np.asarray(calib_cal_support["y_true_cls"], dtype=np.int64)
    reg_models = fit_regressor_ensemble(x_calib, y_calib_resid, seed)

    query_corr_pred = clipped_regression_prediction(
        np.asarray(query_cal["y_pred_reg"], dtype=np.float32)
        + predict_regressor_ensemble(reg_models, build_stageaware_features(query_raw, query_cal, cfg))
    )
    calib_corr_pred_full = clipped_regression_prediction(
        np.asarray(calib_cal_full["y_pred_reg"], dtype=np.float32)
        + predict_regressor_ensemble(reg_models, build_stageaware_features(calib_raw, calib_cal_full, cfg))
    )
    query_corr = clone_regression_output(query_cal, query_corr_pred, cfg)
    calib_corr_full = clone_regression_output(calib_cal_full, calib_corr_pred_full, cfg)
    calib_corr_support = subset_regression_output(calib_corr_full, calib_support_idx, cfg)

    clf_models = fit_classifier_ensemble(build_stageaware_features(calib_raw_support, calib_corr_support, cfg), y_calib_cls, seed)
    query_meta_prob = predict_classifier_ensemble(clf_models, build_stageaware_features(query_raw, query_corr, cfg), cfg.N_CLASSES)
    calib_meta_prob = predict_classifier_ensemble(
        clf_models,
        build_stageaware_features(calib_raw, calib_corr_full, cfg),
        cfg.N_CLASSES,
    )
    centroids, centroid_rows = fit_stage_residual_centroids(calib_corr_support, cfg)
    query_reg_prob_sources = build_reg_prob_sources(query_raw, query_cal, query_corr)
    calib_reg_prob_sources = build_reg_prob_sources(calib_raw, calib_cal_full, calib_corr_full)

    search_rows: List[dict] = []
    if tune_policy:
        policy, search_rows, _, _, _, _ = tune_stageaware_policy(
            query_corr,
            calib_corr_full,
            query_reg_prob_sources,
            calib_reg_prob_sources,
            query_meta_prob,
            calib_meta_prob,
            centroids,
            cfg,
        )
    if policy is None:
        raise ValueError("A tuned policy is required before applying the stage-aware dualmax stack.")

    _, _, calib_stage_pred_reg = apply_stageaware_policy_to_test(
        np.asarray(calib_corr_full["y_pred_reg"], dtype=np.float32),
        calib_reg_prob_sources,
        calib_meta_prob,
        centroids,
        policy,
    )
    calib_stageaware = clone_regression_output(calib_corr_full, calib_stage_pred_reg, cfg)

    hybrid_prob, hybrid_pred, stage_pred_reg = apply_stageaware_policy_to_test(
        np.asarray(query_corr["y_pred_reg"], dtype=np.float32),
        query_reg_prob_sources,
        query_meta_prob,
        centroids,
        policy,
    )
    stageaware_out = clone_regression_output(query_corr, stage_pred_reg, cfg)
    hybrid_metrics = risk_classification_metrics(
        np.asarray(stageaware_out["y_true_cls"], dtype=np.int64),
        hybrid_pred,
        hybrid_prob,
        cfg,
        prefix=prefix,
    )
    return {
        "calib_state": calib_state,
        "calib_cal_full": calib_cal_full,
        "calib_corr": calib_corr_full,
        "calib_stageaware": calib_stageaware,
        "query_cal": query_cal,
        "query_corr": query_corr,
        "stageaware": stageaware_out,
        "hybrid_prob": hybrid_prob,
        "hybrid_pred": hybrid_pred,
        "hybrid_metrics": hybrid_metrics,
        "policy": policy,
        "search_rows": search_rows,
        "centroid_rows": centroid_rows,
        "feature_rows": feature_importance_rows(clf_models),
        "reg_models": reg_models,
        "clf_models": clf_models,
        "centroids": centroids,
        "n_rows_used": int(calib_support_idx.size),
    }


def apply_stageaware_stack(
    query_raw: dict,
    calib_state: dict,
    reg_models,
    clf_models,
    centroids: np.ndarray,
    policy: dict,
    cfg,
    prefix: str,
):
    query_cal = apply_subject_calibration(query_raw, calib_state, cfg)
    query_corr_pred = clipped_regression_prediction(
        np.asarray(query_cal["y_pred_reg"], dtype=np.float32)
        + predict_regressor_ensemble(reg_models, build_stageaware_features(query_raw, query_cal, cfg))
    )
    query_corr = clone_regression_output(query_cal, query_corr_pred, cfg)
    query_meta_prob = predict_classifier_ensemble(clf_models, build_stageaware_features(query_raw, query_corr, cfg), cfg.N_CLASSES)
    reg_prob_sources = build_reg_prob_sources(query_raw, query_cal, query_corr)
    hybrid_prob, hybrid_pred, stage_pred_reg = apply_stageaware_policy_to_test(
        np.asarray(query_corr["y_pred_reg"], dtype=np.float32),
        reg_prob_sources,
        query_meta_prob,
        centroids,
        policy,
    )
    stageaware_out = clone_regression_output(query_corr, stage_pred_reg, cfg)
    hybrid_metrics = risk_classification_metrics(
        np.asarray(stageaware_out["y_true_cls"], dtype=np.int64),
        hybrid_pred,
        hybrid_prob,
        cfg,
        prefix=prefix,
    )
    return {
        "query_cal": query_cal,
        "query_corr": query_corr,
        "stageaware": stageaware_out,
        "hybrid_prob": hybrid_prob,
        "hybrid_pred": hybrid_pred,
        "hybrid_metrics": hybrid_metrics,
    }


def hybrid_metrics_for_compat(hybrid_metrics: dict, prefix: str) -> dict:
    return {
        "cls_acc_from_reg": float(hybrid_metrics[f"cls_acc_{prefix}"]),
        "cls_balanced_acc_from_reg": float(hybrid_metrics[f"cls_balanced_acc_{prefix}"]),
        "cls_f1_macro_from_reg": float(hybrid_metrics[f"cls_f1_macro_{prefix}"]),
        "cls_f1_weighted_from_reg": float(hybrid_metrics[f"cls_f1_weighted_{prefix}"]),
        "cls_precision_macro_from_reg": float(hybrid_metrics[f"cls_precision_macro_{prefix}"]),
        "cls_precision_weighted_from_reg": float(hybrid_metrics[f"cls_precision_weighted_{prefix}"]),
        "cls_recall_macro_from_reg": float(hybrid_metrics[f"cls_recall_macro_{prefix}"]),
        "cls_recall_weighted_from_reg": float(hybrid_metrics[f"cls_recall_weighted_{prefix}"]),
        "cls_kappa_from_reg": float(hybrid_metrics[f"cls_kappa_{prefix}"]),
        "cls_mcc_from_reg": float(hybrid_metrics[f"cls_mcc_{prefix}"]),
        "cls_ece_from_reg": float(hybrid_metrics[f"cls_ece_{prefix}"]),
        "cls_brier_from_reg": float(hybrid_metrics[f"cls_brier_{prefix}"]),
    }


def _legacy_main():
    cfg = make_v10_2_optimized_long_cfg()
    cfg.OUTPUT_NAME = "mimic_bp_reg_v10_2_optlong_stageaware_dualmax_proto"
    cfg.PROTOCOL_ID = "v10.2_optlong_stageaware_dualmax"
    cfg.PROTOCOL_NAME = "v10.2 opt-long stage-aware dual optimization with frozen checkpoint"

    seed_everything(cfg.SEED)
    ckpt_path = pick_optlong_checkpoint(cfg)
    out_root, fig_dir, art_dir, tbl_dir = ensure_out_dirs(cfg)
    loaders = build_protocol_loaders(cfg, task="regression")
    save_json(out_root / "protocol_manifest.json", loaders.manifest)
    print(f"Using device: {cfg.DEVICE}")
    print(f"Checkpoint: {ckpt_path}")

    model = QMoERegressionNet(cfg).to(cfg.DEVICE)
    state = torch.load(ckpt_path, map_location=cfg.DEVICE)
    model.load_state_dict(state, strict=False)

    val_query_raw = collect_outputs_regression(model, loaders.val_query_loader, cfg)
    val_calib_raw = collect_outputs_regression(model, loaders.val_calib_loader, cfg)
    test_query_raw = collect_outputs_regression(model, loaders.test_query_loader, cfg)
    test_calib_raw = collect_outputs_regression(model, loaders.test_calib_loader, cfg)

    val_cal_state = fit_subject_calibration_state(val_calib_raw, cfg)
    val_calib_cal = apply_subject_calibration(val_calib_raw, val_cal_state, cfg)
    val_query_cal = apply_subject_calibration(val_query_raw, val_cal_state, cfg)

    x_val_calib = build_stageaware_features(val_calib_raw, val_calib_cal, cfg)
    y_val_calib_resid = np.asarray(val_calib_cal["y_true_reg"], dtype=np.float32) - np.asarray(val_calib_cal["y_pred_reg"], dtype=np.float32)
    y_val_calib_cls = np.asarray(val_calib_cal["y_true_cls"], dtype=np.int64)
    reg_models_val = fit_regressor_ensemble(x_val_calib, y_val_calib_resid, cfg.SEED)

    val_query_corr_pred = clipped_regression_prediction(
        np.asarray(val_query_cal["y_pred_reg"], dtype=np.float32)
        + predict_regressor_ensemble(reg_models_val, build_stageaware_features(val_query_raw, val_query_cal, cfg))
    )
    val_calib_corr_pred = clipped_regression_prediction(
        np.asarray(val_calib_cal["y_pred_reg"], dtype=np.float32) + predict_regressor_ensemble(reg_models_val, x_val_calib)
    )
    val_query_corr = clone_regression_output(val_query_cal, val_query_corr_pred, cfg)
    val_calib_corr = clone_regression_output(val_calib_cal, val_calib_corr_pred, cfg)

    val_clf_models = fit_classifier_ensemble(build_stageaware_features(val_calib_raw, val_calib_corr, cfg), y_val_calib_cls, cfg.SEED)
    val_meta_prob = predict_classifier_ensemble(val_clf_models, build_stageaware_features(val_query_raw, val_query_corr, cfg), cfg.N_CLASSES)
    val_calib_meta_prob = predict_classifier_ensemble(
        val_clf_models,
        build_stageaware_features(val_calib_raw, val_calib_corr, cfg),
        cfg.N_CLASSES,
    )
    val_centroids, val_centroid_rows = fit_stage_residual_centroids(val_calib_corr, cfg)
    val_reg_prob_sources = build_reg_prob_sources(val_query_raw, val_query_cal, val_query_corr)
    val_calib_reg_prob_sources = build_reg_prob_sources(val_calib_raw, val_calib_cal, val_calib_corr)
    best_policy, search_rows, _, _, _, _ = tune_stageaware_policy(
        val_query_corr,
        val_calib_corr,
        val_reg_prob_sources,
        val_calib_reg_prob_sources,
        val_meta_prob,
        val_calib_meta_prob,
        val_centroids,
        cfg,
    )

    test_cal_state = fit_subject_calibration_state(test_calib_raw, cfg)
    test_calib_cal = apply_subject_calibration(test_calib_raw, test_cal_state, cfg)
    test_query_cal = apply_subject_calibration(test_query_raw, test_cal_state, cfg)

    x_test_calib = build_stageaware_features(test_calib_raw, test_calib_cal, cfg)
    y_test_calib_resid = np.asarray(test_calib_cal["y_true_reg"], dtype=np.float32) - np.asarray(test_calib_cal["y_pred_reg"], dtype=np.float32)
    y_test_calib_cls = np.asarray(test_calib_cal["y_true_cls"], dtype=np.int64)
    reg_models = fit_regressor_ensemble(x_test_calib, y_test_calib_resid, cfg.SEED)

    test_query_corr_pred = clipped_regression_prediction(
        np.asarray(test_query_cal["y_pred_reg"], dtype=np.float32)
        + predict_regressor_ensemble(reg_models, build_stageaware_features(test_query_raw, test_query_cal, cfg))
    )
    test_calib_corr_pred = clipped_regression_prediction(
        np.asarray(test_calib_cal["y_pred_reg"], dtype=np.float32) + predict_regressor_ensemble(reg_models, x_test_calib)
    )
    test_query_corr = clone_regression_output(test_query_cal, test_query_corr_pred, cfg)
    test_calib_corr = clone_regression_output(test_calib_cal, test_calib_corr_pred, cfg)

    clf_models = fit_classifier_ensemble(build_stageaware_features(test_calib_raw, test_calib_corr, cfg), y_test_calib_cls, cfg.SEED)
    test_meta_prob = predict_classifier_ensemble(clf_models, build_stageaware_features(test_query_raw, test_query_corr, cfg), cfg.N_CLASSES)
    test_centroids, test_centroid_rows = fit_stage_residual_centroids(test_calib_corr, cfg)
    test_reg_prob_sources = build_reg_prob_sources(test_query_raw, test_query_cal, test_query_corr)
    test_hybrid_prob, test_hybrid_pred, test_stage_pred_reg = apply_stageaware_policy_to_test(
        np.asarray(test_query_corr["y_pred_reg"], dtype=np.float32),
        test_reg_prob_sources,
        test_meta_prob,
        test_centroids,
        best_policy,
    )

    test_stageaware = clone_regression_output(test_query_corr, test_stage_pred_reg, cfg)
    test_hybrid_metrics = risk_classification_metrics(
        np.asarray(test_stageaware["y_true_cls"], dtype=np.int64),
        test_hybrid_pred,
        test_hybrid_prob,
        cfg,
        prefix="stageaware_dualmax",
    )

    low, high, conformal_default = conformal_from_outputs(test_calib_corr, test_stageaware, alpha=cfg.CONFORMAL_ALPHA)
    conformal_rows = []
    for alpha in cfg.CONFORMAL_ALPHAS:
        _, _, met = conformal_from_outputs(test_calib_corr, test_stageaware, alpha=alpha)
        conformal_rows.append({"alpha": alpha, **met})

    cond_rows = build_conditional_coverage_table(test_stageaware["y_true_reg"], low, high, test_stageaware["quality"], cfg)
    calib_curve_rows = build_calibration_curve_table(np.asarray(test_stageaware["y_true_cls"], dtype=np.int64), test_hybrid_prob, n_bins=cfg.ECE_BINS)
    bp_range_rows = build_bp_range_table(test_stageaware["y_true_reg"], test_stageaware["y_pred_reg"])
    subject_rows = build_subjectwise_error_table(test_stageaware["y_true_reg"], test_stageaware["y_pred_reg"], test_stageaware["subject_ids"])
    subject_gain_rows = build_subject_gain_table(test_query_raw, test_stageaware)
    error_cdf_rows = build_error_cdf_rows(test_stageaware["y_true_reg"], test_stageaware["y_pred_reg"])
    split_dist_rows = build_split_distribution_rows(loaders.split_datasets, cfg)

    runtime = measure_runtime(model, next(iter(loaders.test_query_loader)), cfg)
    final_results = {
        "device": cfg.DEVICE,
        "protocol_id": cfg.PROTOCOL_ID,
        "split_protocol": cfg.SPLIT_PROTOCOL,
        "checkpoint": str(ckpt_path),
        "hybrid_policy": best_policy,
        "paper_metrics_reg_corrected": build_paper_metrics(test_query_corr["metrics_reg"]),
        "paper_metrics_reg_stageaware": build_paper_metrics(test_stageaware["metrics_reg"]),
        "test_regression_corrected": {
            **test_query_corr["metrics_reg"],
            **test_query_corr["metrics_cls_from_reg"],
            **test_query_corr["uncertainty_metrics"],
        },
        "test_regression_stageaware": {
            **test_stageaware["metrics_reg"],
            **test_stageaware["metrics_cls_from_reg"],
            **test_stageaware["uncertainty_metrics"],
        },
        "test_hybrid_classification": test_hybrid_metrics,
        "conformal_default": conformal_default,
        "runtime": runtime,
    }

    save_json(out_root / "final_results.json", final_results)
    save_json(out_root / "paper_metrics_reg_corrected.json", build_paper_metrics(test_query_corr["metrics_reg"]))
    save_json(out_root / "paper_metrics_reg_stageaware.json", build_paper_metrics(test_stageaware["metrics_reg"]))
    save_json(out_root / "runtime_metrics.json", runtime)
    save_rows_csv(out_root / "stageaware_policy_search.csv", sorted(search_rows, key=lambda row: row["score"], reverse=True))
    save_rows_csv(out_root / "stageaware_feature_importance.csv", feature_importance_rows(clf_models))
    save_rows_csv(out_root / "stageaware_centroids_val.csv", val_centroid_rows)
    save_rows_csv(out_root / "stageaware_centroids_test.csv", test_centroid_rows)
    save_rows_csv(out_root / "conformal_sweep.csv", conformal_rows)
    save_rows_csv(out_root / "conditional_coverage.csv", cond_rows)
    save_rows_csv(out_root / "calibration_curve.csv", calib_curve_rows)
    save_rows_csv(out_root / "error_cdf.csv", error_cdf_rows)
    save_rows_csv(out_root / "split_class_distribution.csv", split_dist_rows)
    save_rows_csv(tbl_dir / "bp_range_metrics.csv", bp_range_rows)
    save_rows_csv(tbl_dir / "subjectwise_error.csv", subject_rows)
    save_rows_csv(tbl_dir / "subject_calibration_gain.csv", subject_gain_rows)
    save_regression_npz(art_dir / "test_outputs_regression_raw.npz", test_query_raw)
    save_regression_npz(art_dir / "test_outputs_regression_corrected.npz", test_query_corr)
    save_regression_npz(art_dir / "test_outputs_regression_stageaware.npz", test_stageaware)

    plot_scatter_true_vs_pred(test_stageaware["y_true_reg"], test_stageaware["y_pred_reg"], fig_dir, filename="scatter_true_vs_pred.png")
    plot_bland_altman(test_stageaware["y_true_reg"], test_stageaware["y_pred_reg"], fig_dir, filename="bland_altman.png")
    plot_confusion(
        np.asarray(test_stageaware["y_true_cls"], dtype=np.int64),
        test_hybrid_pred,
        list(cfg.CLASS_NAMES),
        fig_dir,
        "confusion_matrix_stageaware_dualmax.png",
    )
    plot_roc_pr(np.asarray(test_stageaware["y_true_cls"], dtype=np.int64), test_hybrid_prob, cfg, fig_dir, prefix="stageaware_dualmax")
    plot_calibration(calib_curve_rows, fig_dir, filename="calibration_curve.png")
    plot_quality_conditional_coverage(cond_rows, fig_dir)
    plot_sharpness_vs_coverage(conformal_rows, fig_dir)
    plot_uncertainty_error_corr(
        np.asarray(test_stageaware["uncertainty"], dtype=np.float32),
        np.abs(np.asarray(test_stageaware["y_pred_reg"], dtype=np.float32) - np.asarray(test_stageaware["y_true_reg"], dtype=np.float32)).mean(axis=1),
        fig_dir,
    )
    plot_router_heatmap(np.asarray(test_stageaware["alpha"], dtype=np.float32), test_stageaware["y_true_reg"], fig_dir, ["PPG", "ECG", "Joint", "Cross"])
    plot_error_cdf(error_cdf_rows, fig_dir)
    plot_bp_range_bias(bp_range_rows, fig_dir)
    plot_subject_calibration_gain(subject_gain_rows, fig_dir)
    plot_split_class_distribution(split_dist_rows, fig_dir, cfg)

    print(f"Done. Stage-aware dual optimization results saved to: {out_root}")


def main():
    cfg = build_stageaware_cfg()
    seed_everything(cfg.SEED)
    ckpt_path = pick_optlong_checkpoint(cfg)
    cfg.INIT_CKPT_PATH = str(ckpt_path)

    out_root, fig_dir, art_dir, tbl_dir = ensure_out_dirs(cfg)
    loaders = build_protocol_loaders(cfg, task="regression")
    save_json(out_root / "protocol_manifest.json", loaders.manifest)
    print(f"Using device: {cfg.DEVICE}")
    print(f"Protocol rank: {cfg.PROTOCOL_STRICTNESS_RANK}")
    print(f"Split protocol: {cfg.SPLIT_PROTOCOL}")
    print(f"Protocol name: {cfg.PROTOCOL_NAME}")
    print(f"Warm-start checkpoint: {ckpt_path}")
    print(f"Target epochs: {cfg.EPOCHS}")

    model, epoch_logs, training_meta = train_stageaware_backbone(cfg, loaders, out_root, ckpt_path)
    if epoch_logs:
        plot_training_curves_reg_split(epoch_logs, fig_dir)

    val_query_raw = collect_outputs_regression(model, loaders.val_query_loader, cfg)
    val_calib_raw = collect_outputs_regression(model, loaders.val_calib_loader, cfg)
    val_bundle = fit_stageaware_bundle(
        val_calib_raw,
        val_query_raw,
        cfg,
        seed=cfg.SEED,
        prefix="stageaware_val",
        tune_policy=True,
    )
    best_policy = val_bundle["policy"]
    print(
        "Best stage-aware policy: "
        f"reg_source={best_policy['reg_source']} | "
        f"blend={best_policy['meta_blend_weight']:.3f} | "
        f"gamma={best_policy['gamma']:.3f} | "
        f"shift={best_policy['shift_scale']:.3f}"
    )

    test_query_raw = collect_outputs_regression(model, loaders.test_query_loader, cfg)
    test_calib_raw = collect_outputs_regression(model, loaders.test_calib_loader, cfg)
    test_bundle = fit_stageaware_bundle(
        test_calib_raw,
        test_query_raw,
        cfg,
        seed=cfg.SEED,
        prefix="stageaware_dualmax",
        policy=best_policy,
        tune_policy=False,
    )

    test_query_corr = test_bundle["query_corr"]
    test_stageaware = test_bundle["stageaware"]
    test_hybrid_prob = test_bundle["hybrid_prob"]
    test_hybrid_pred = test_bundle["hybrid_pred"]
    test_hybrid_metrics = test_bundle["hybrid_metrics"]
    test_calib_stageaware = test_bundle["calib_stageaware"]

    low, high, conformal_default = conformal_from_outputs(test_calib_stageaware, test_stageaware, alpha=cfg.CONFORMAL_ALPHA)
    conformal_rows = []
    for alpha in cfg.CONFORMAL_ALPHAS:
        _, _, met = conformal_from_outputs(test_calib_stageaware, test_stageaware, alpha=alpha)
        conformal_rows.append({"alpha": alpha, **met})

    cond_rows = build_conditional_coverage_table(test_stageaware["y_true_reg"], low, high, test_stageaware["quality"], cfg)
    calib_curve_rows = build_calibration_curve_table(np.asarray(test_stageaware["y_true_cls"], dtype=np.int64), test_hybrid_prob, n_bins=cfg.ECE_BINS)
    bp_range_rows_raw = build_bp_range_table(test_query_raw["y_true_reg"], test_query_raw["y_pred_reg"])
    bp_range_rows = build_bp_range_table(test_stageaware["y_true_reg"], test_stageaware["y_pred_reg"])
    subject_rows_raw = build_subjectwise_error_table(test_query_raw["y_true_reg"], test_query_raw["y_pred_reg"], test_query_raw["subject_ids"])
    subject_rows = build_subjectwise_error_table(test_stageaware["y_true_reg"], test_stageaware["y_pred_reg"], test_stageaware["subject_ids"])
    subject_gain_rows = build_subject_gain_table(test_query_raw, test_stageaware)
    error_cdf_rows = build_error_cdf_rows(test_stageaware["y_true_reg"], test_stageaware["y_pred_reg"])
    split_dist_rows = build_split_distribution_rows(loaders.split_datasets, cfg)

    noise_rows = []
    for noise_std in cfg.NOISE_STDS:
        noise_raw = collect_outputs_regression(model, loaders.test_query_loader, cfg, noise_std=noise_std)
        noise_prefix = f"noise_{str(noise_std).replace('.', '_')}"
        noise_eval = apply_stageaware_stack(
            noise_raw,
            test_bundle["calib_state"],
            test_bundle["reg_models"],
            test_bundle["clf_models"],
            test_bundle["centroids"],
            best_policy,
            cfg,
            prefix=noise_prefix,
        )
        noise_rows.append(
            {
                "noise_std": float(noise_std),
                **noise_eval["stageaware"]["metrics_reg"],
                **hybrid_metrics_for_compat(noise_eval["hybrid_metrics"], noise_prefix),
                **noise_eval["stageaware"]["uncertainty_metrics"],
            }
        )

    missing_rows = []
    test_drop_ppg = None
    test_drop_ecg = None
    test_drop_ppg_metrics = {}
    test_drop_ecg_metrics = {}
    for missing_prob in cfg.MISSING_PROBS:
        drop_ppg_raw = collect_outputs_regression(model, loaders.test_query_loader, cfg, drop_modality="ppg", missing_prob=missing_prob)
        drop_ecg_raw = collect_outputs_regression(model, loaders.test_query_loader, cfg, drop_modality="ecg", missing_prob=missing_prob)
        ppg_prefix = f"drop_ppg_{str(missing_prob).replace('.', '_')}"
        ecg_prefix = f"drop_ecg_{str(missing_prob).replace('.', '_')}"
        drop_ppg_eval = apply_stageaware_stack(
            drop_ppg_raw,
            test_bundle["calib_state"],
            test_bundle["reg_models"],
            test_bundle["clf_models"],
            test_bundle["centroids"],
            best_policy,
            cfg,
            prefix=ppg_prefix,
        )
        drop_ecg_eval = apply_stageaware_stack(
            drop_ecg_raw,
            test_bundle["calib_state"],
            test_bundle["reg_models"],
            test_bundle["clf_models"],
            test_bundle["centroids"],
            best_policy,
            cfg,
            prefix=ecg_prefix,
        )
        if abs(float(missing_prob) - 1.0) < 1e-8:
            test_drop_ppg = drop_ppg_eval
            test_drop_ecg = drop_ecg_eval
            test_drop_ppg_metrics = hybrid_metrics_for_compat(drop_ppg_eval["hybrid_metrics"], ppg_prefix)
            test_drop_ecg_metrics = hybrid_metrics_for_compat(drop_ecg_eval["hybrid_metrics"], ecg_prefix)
        missing_rows.append(
            {
                "missing_prob": float(missing_prob),
                "ppg_missing_mae_sbp": float(drop_ppg_eval["stageaware"]["metrics_reg"]["mae_sbp"]),
                "ppg_missing_mae_dbp": float(drop_ppg_eval["stageaware"]["metrics_reg"]["mae_dbp"]),
                "ppg_missing_cls_f1_macro_from_reg": float(drop_ppg_eval["hybrid_metrics"][f"cls_f1_macro_{ppg_prefix}"]),
                "ecg_missing_mae_sbp": float(drop_ecg_eval["stageaware"]["metrics_reg"]["mae_sbp"]),
                "ecg_missing_mae_dbp": float(drop_ecg_eval["stageaware"]["metrics_reg"]["mae_dbp"]),
                "ecg_missing_cls_f1_macro_from_reg": float(drop_ecg_eval["hybrid_metrics"][f"cls_f1_macro_{ecg_prefix}"]),
            }
        )

    few_shot_rows = []
    for shot in sorted(set(int(x) for x in cfg.FEW_SHOT_SWEEP)):
        if shot <= 0 or not cfg.USE_SUBJECT_CALIBRATION:
            out = test_query_raw
            compat_metrics = dict(test_query_raw["metrics_cls_from_reg"])
            n_rows_used = 0
        else:
            shot_prefix = f"stageaware_shot{shot}"
            shot_bundle = fit_stageaware_bundle(
                test_calib_raw,
                test_query_raw,
                cfg,
                seed=cfg.SEED + 101 * shot,
                prefix=shot_prefix,
                policy=best_policy,
                tune_policy=False,
                n_shots=shot,
            )
            out = shot_bundle["stageaware"]
            compat_metrics = hybrid_metrics_for_compat(shot_bundle["hybrid_metrics"], shot_prefix)
            n_rows_used = int(shot_bundle["n_rows_used"])
        few_shot_rows.append(
            {
                "n_shots": int(shot),
                **out["metrics_reg"],
                **compat_metrics,
                **out["uncertainty_metrics"],
                "calibration_enabled": bool(shot > 0 and cfg.USE_SUBJECT_CALIBRATION),
                "n_rows_used": int(n_rows_used),
            }
        )

    runtime = measure_runtime(model, next(iter(loaders.test_query_loader)), cfg)
    class_weights_train = build_class_weights(loaders.ds_train, cfg, cfg.DEVICE).detach().cpu().tolist()
    paper_metrics_raw = build_paper_metrics(test_query_raw["metrics_reg"])
    paper_metrics_final = build_paper_metrics(test_stageaware["metrics_reg"])
    paper_metrics_corrected = build_paper_metrics(test_query_corr["metrics_reg"])
    subject_calibration = {
        "enabled": bool(cfg.USE_SUBJECT_CALIBRATION),
        "mode": test_bundle["calib_state"]["mode"],
        "shrinkage": float(test_bundle["calib_state"]["shrinkage"]),
        "n_subjects": int(test_bundle["calib_state"]["n_subjects"]),
        "global_scale": np.asarray(test_bundle["calib_state"]["global_scale"], dtype=np.float32).tolist(),
        "global_offset": np.asarray(test_bundle["calib_state"]["global_offset"], dtype=np.float32).tolist(),
    }
    stageaware_compat = hybrid_metrics_for_compat(test_hybrid_metrics, "stageaware_dualmax")

    final_results = {
        "device": cfg.DEVICE,
        "protocol_id": cfg.PROTOCOL_ID,
        "protocol_rank": int(cfg.PROTOCOL_STRICTNESS_RANK),
        "split_protocol": cfg.SPLIT_PROTOCOL,
        "protocol_manifest": loaders.manifest,
        "checkpoint_init": str(ckpt_path),
        "checkpoint_best": str(out_root / "best_model.pt"),
        "training": training_meta,
        "subject_calibration": subject_calibration,
        "few_shot_protocol": {
            "shots": [int(r["n_shots"]) for r in few_shot_rows],
            "note": "0-shot is raw inference; k-shot re-fits subject calibration and stage-aware adapters with the first k calibration segments per subject.",
        },
        "hybrid_policy": best_policy,
        "paper_metrics_raw": paper_metrics_raw,
        "paper_metrics": paper_metrics_final,
        "paper_metrics_reg_corrected": paper_metrics_corrected,
        "paper_metrics_reg_stageaware": paper_metrics_final,
        "class_counts_train": loaders.ds_train.class_counts,
        "class_weights_train": class_weights_train,
        "avg_credibility_test": np.asarray(test_query_raw["credibility"], dtype=np.float32).mean(axis=0).tolist(),
        "avg_router_test": np.asarray(test_query_raw["alpha"], dtype=np.float32).mean(axis=0).tolist(),
        "avg_quality_test": float(np.asarray(test_query_raw["quality"], dtype=np.float32).mean()),
        "runtime": runtime,
        "test_full_raw": {
            **test_query_raw["metrics_reg"],
            **test_query_raw["metrics_cls_from_reg"],
            **test_query_raw["metrics_cls_from_reg_hard"],
            **test_query_raw["uncertainty_metrics"],
        },
        "test_regression_corrected": {
            **test_query_corr["metrics_reg"],
            **test_query_corr["metrics_cls_from_reg"],
            **test_query_corr["uncertainty_metrics"],
        },
        "test_full": {
            **test_stageaware["metrics_reg"],
            **test_stageaware["metrics_cls_from_reg"],
            **stageaware_compat,
            **test_stageaware["uncertainty_metrics"],
            **{f"conformal_{k}": v for k, v in conformal_default.items()},
        },
        "test_regression_stageaware": {
            **test_stageaware["metrics_reg"],
            **test_stageaware["metrics_cls_from_reg"],
            **test_stageaware["uncertainty_metrics"],
        },
        "test_hybrid_classification": test_hybrid_metrics,
        "test_drop_ppg": {
            **(test_drop_ppg["stageaware"]["metrics_reg"] if test_drop_ppg is not None else {}),
            **test_drop_ppg_metrics,
            **(test_drop_ppg["stageaware"]["uncertainty_metrics"] if test_drop_ppg is not None else {}),
        },
        "test_drop_ecg": {
            **(test_drop_ecg["stageaware"]["metrics_reg"] if test_drop_ecg is not None else {}),
            **test_drop_ecg_metrics,
            **(test_drop_ecg["stageaware"]["uncertainty_metrics"] if test_drop_ecg is not None else {}),
        },
        "conformal_default": conformal_default,
    }

    save_json(out_root / "final_results.json", final_results)
    save_json(out_root / "paper_metrics_raw.json", paper_metrics_raw)
    save_json(out_root / "paper_metrics.json", paper_metrics_final)
    save_json(out_root / "paper_metrics_reg_corrected.json", paper_metrics_corrected)
    save_json(out_root / "paper_metrics_reg_stageaware.json", paper_metrics_final)
    save_json(out_root / "runtime_metrics.json", runtime)
    save_rows_csv(out_root / "stageaware_policy_search.csv", sorted(val_bundle["search_rows"], key=lambda row: row["score"], reverse=True))
    save_rows_csv(out_root / "stageaware_feature_importance.csv", test_bundle["feature_rows"])
    save_rows_csv(out_root / "stageaware_centroids_val.csv", val_bundle["centroid_rows"])
    save_rows_csv(out_root / "stageaware_centroids_test.csv", test_bundle["centroid_rows"])
    save_rows_csv(out_root / "conformal_sweep.csv", conformal_rows)
    save_rows_csv(out_root / "noise_metrics.csv", noise_rows)
    save_rows_csv(out_root / "missing_modality_metrics.csv", missing_rows)
    save_rows_csv(out_root / "conditional_coverage.csv", cond_rows)
    save_rows_csv(out_root / "calibration_curve.csv", calib_curve_rows)
    save_rows_csv(out_root / "error_cdf.csv", error_cdf_rows)
    save_rows_csv(out_root / "few_shot_sweep.csv", few_shot_rows)
    save_rows_csv(out_root / "split_class_distribution.csv", split_dist_rows)
    save_rows_csv(tbl_dir / "bp_range_metrics_raw.csv", bp_range_rows_raw)
    save_rows_csv(tbl_dir / "bp_range_metrics.csv", bp_range_rows)
    save_rows_csv(tbl_dir / "subjectwise_error_raw.csv", subject_rows_raw)
    save_rows_csv(tbl_dir / "subjectwise_error.csv", subject_rows)
    save_rows_csv(tbl_dir / "subject_calibration_gain.csv", subject_gain_rows)
    save_regression_npz(art_dir / "test_outputs_regression_raw.npz", test_query_raw)
    save_regression_npz(art_dir / "test_outputs_regression_corrected.npz", test_query_corr)
    save_regression_npz(art_dir / "test_outputs_regression_stageaware.npz", test_stageaware)
    save_regression_npz(art_dir / "test_outputs_regression.npz", test_stageaware)

    plot_scatter_true_vs_pred(test_stageaware["y_true_reg"], test_stageaware["y_pred_reg"], fig_dir, filename="scatter_true_vs_pred.png")
    plot_bland_altman(test_stageaware["y_true_reg"], test_stageaware["y_pred_reg"], fig_dir, filename="bland_altman.png")
    plot_confusion(np.asarray(test_stageaware["y_true_cls"], dtype=np.int64), test_hybrid_pred, list(cfg.CLASS_NAMES), fig_dir, "confusion_matrix_stageaware_dualmax.png")
    plot_confusion(np.asarray(test_stageaware["y_true_cls"], dtype=np.int64), test_hybrid_pred, list(cfg.CLASS_NAMES), fig_dir, "confusion_matrix_reg_to_class.png")
    plot_roc_pr(np.asarray(test_stageaware["y_true_cls"], dtype=np.int64), test_hybrid_prob, cfg, fig_dir, prefix="stageaware_dualmax")
    plot_roc_pr(np.asarray(test_stageaware["y_true_cls"], dtype=np.int64), test_hybrid_prob, cfg, fig_dir, prefix="reg_to_class")
    plot_calibration(calib_curve_rows, fig_dir, filename="calibration_curve.png")
    plot_quality_conditional_coverage(cond_rows, fig_dir)
    plot_sharpness_vs_coverage(conformal_rows, fig_dir)
    plot_uncertainty_error_corr(
        np.asarray(test_stageaware["uncertainty"], dtype=np.float32),
        np.abs(np.asarray(test_stageaware["y_pred_reg"], dtype=np.float32) - np.asarray(test_stageaware["y_true_reg"], dtype=np.float32)).mean(axis=1),
        fig_dir,
    )
    plot_router_heatmap(np.asarray(test_stageaware["alpha"], dtype=np.float32), test_stageaware["y_true_reg"], fig_dir, ["PPG", "ECG", "Joint", "Cross"])
    plot_noise_robustness(noise_rows, fig_dir)
    plot_missing_modality_curve(missing_rows, fig_dir)
    plot_error_cdf(error_cdf_rows, fig_dir)
    plot_bp_range_bias(bp_range_rows, fig_dir)
    plot_few_shot_curve(few_shot_rows, fig_dir)
    plot_subject_calibration_gain(subject_gain_rows, fig_dir)
    plot_split_class_distribution(split_dist_rows, fig_dir, cfg)

    print(f"Done. Stage-aware dual optimization results saved to: {out_root}")


if __name__ == "__main__":
    main()
