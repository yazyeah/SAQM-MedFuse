from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

import aqm_bp_shared_v9 as shared_plots
import train_aqm_medfuse_mimic_bp_reg_v10_2_dualmax_bridgeguided_refinement_protocol as guided_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_dualanchor_conservative_fusion_protocol as base_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_dualanchor_resume_tailcal_protocol as prev_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_dualanchor_stability_ensemble_protocol as stability_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_dualanchor_consensus_sparse_tail_protocol as consensus_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_dualbackbone_bridge_protocol as bridge_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_stageaware_dualmax_protocol as stage_script


def build_meta_stack_cfg():
    cfg = stability_script.build_stability_ensemble_cfg()
    cfg.OUTPUT_NAME = "mimic_bp_reg_v10_2_optlong_dualanchor_meta_stack_proto"
    cfg.PROTOCOL_ID = "v10.2_optlong_dualanchor_meta_stack"
    cfg.PROTOCOL_NAME = (
        "v10.2 opt-long dual-anchor meta-stack protocol "
        "(resume stability feature_head_best + learned meta classification blend + "
        "confidence-gated residual regression stack + extended paper diagnostics)"
    )
    cfg.HEAD_RESUME_PATH = (
        Path(cfg.PROJECT_ROOT)
        / "outputs"
        / "mimic_bp_reg_v10_2_optlong_dualanchor_stability_ensemble_proto"
        / "feature_head_best.pt"
    )
    cfg.HEAD_EPOCHS = 10
    cfg.HEAD_PATIENCE = 3
    cfg.HEAD_LR = 6.0e-5
    cfg.HEAD_MISSING_PPG = max(float(cfg.HEAD_MISSING_PPG), 0.35)
    cfg.HEAD_MISSING_ECG = max(float(cfg.HEAD_MISSING_ECG), 0.35)

    cfg.META_BLEND_WEIGHTS = (0.00, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00)
    cfg.META_RESIDUAL_SCALES = (0.00, 0.35, 0.55, 0.75, 1.00)
    cfg.META_CONF_BETAS = (0.00, 0.50, 1.00, 1.50)
    cfg.META_RESID_CLIP_SBP = 5.0
    cfg.META_RESID_CLIP_DBP = 3.5
    cfg.BOOTSTRAP_SAMPLES = 200
    cfg.ENABLE_CLASSIFICATION_ARBITER = False
    cfg.CLS_ARBITER_SCALES = (0.35, 0.50, 0.65, 0.80, 1.00)
    cfg.CLS_ARBITER_BETAS = (0.80, 1.00, 1.20, 1.50)
    cfg.CLS_ARBITER_FLOORS = (0.00, 0.02, 0.05)
    cfg.CLS_ARBITER_AGREE_SHRINKS = (0.30, 0.45, 0.60)
    cfg.CLS_ARBITER_MAX_OVERRIDE = 0.97
    cfg.ENABLE_REGRESSION_ROUTER = False
    cfg.REG_ROUTER_BLEND_SCALES = (0.25, 0.50, 0.75, 1.00)
    cfg.REG_ROUTER_TEMPS = (0.60, 0.85, 1.10, 1.35)
    cfg.REG_ROUTER_GAMMAS = (0.75, 1.00, 1.25)
    cfg.REG_ROUTER_FLOORS = (0.00, 0.02, 0.05)
    cfg.ENABLE_RISK_GUARD = False
    cfg.RISK_GUARD_SCALES = (0.0, 0.35, 0.55, 0.75, 1.00)
    cfg.RISK_GUARD_BETAS = (0.60, 0.85, 1.10)
    cfg.RISK_GUARD_HIGH_GAINS = (1.00, 1.20, 1.40)
    cfg.RISK_GUARD_CRISIS_GAINS = (1.00, 1.35, 1.70)
    cfg.RISK_GUARD_HIGH_TRUE_WEIGHT = 2.5
    cfg.RISK_GUARD_CRISIS_TRUE_WEIGHT = 4.5
    cfg.RISK_GUARD_HIGH_UNDER_WEIGHT = 1.8
    cfg.RISK_GUARD_MAX_SHIFT_SBP = 12.0
    cfg.RISK_GUARD_MAX_SHIFT_DBP = 8.0
    cfg.RISK_GUARD_MAX_NEG_FRAC = 0.12
    cfg.RISK_GUARD_MAX_MAE_DELTA = 0.08
    cfg.RISK_GUARD_MAX_COVERAGE_GAP_DELTA = 0.02
    cfg.ENABLE_SAFETY_CLASS_FUSION = False
    cfg.SAFETY_CLASS_FUSION_SCALES = (0.0, 0.15, 0.30, 0.45, 0.60)
    cfg.SAFETY_CLASS_FUSION_BETAS = (0.55, 0.80, 1.05, 1.30)
    cfg.SAFETY_CLASS_FUSION_DISAGREE_GAINS = (1.00, 1.20, 1.45, 1.70)
    cfg.SAFETY_CLASS_FUSION_HIGH_GAINS = (1.00, 1.20, 1.45)
    cfg.SAFETY_CLASS_FUSION_CRISIS_GAINS = (1.00, 1.35, 1.70)
    cfg.SAFETY_CLASS_FUSION_MAX_WEIGHT = 0.60
    cfg.SAFETY_CLASS_FUSION_STAGE1_RECALL_WEIGHT = 0.08
    cfg.SAFETY_CLASS_FUSION_STAGE2_RECALL_WEIGHT = 0.18
    cfg.SAFETY_CLASS_FUSION_STAGE2_F1_WEIGHT = 0.06
    cfg.ENABLE_HIGH_BIAS_CALIBRATOR = False
    cfg.HIGH_BIAS_CAL_HIGH_THRESHOLDS = (0.35, 0.45, 0.55)
    cfg.HIGH_BIAS_CAL_CRISIS_THRESHOLDS = (0.20, 0.30, 0.40)
    cfg.HIGH_BIAS_CAL_GAMMAS = (0.80, 1.00, 1.25)
    cfg.HIGH_BIAS_CAL_SBP_HIGH_SHIFTS = (0.0, 0.5, 1.0, 1.5, 2.0)
    cfg.HIGH_BIAS_CAL_SBP_CRISIS_SHIFTS = (0.0, 1.0, 2.0, 3.0)
    cfg.HIGH_BIAS_CAL_DBP_HIGH_SHIFTS = (0.0, 0.25, 0.5, 0.75, 1.0)
    cfg.HIGH_BIAS_CAL_DBP_CRISIS_SHIFTS = (0.0, 0.5, 1.0, 1.5)
    cfg.HIGH_BIAS_CAL_MAX_MAE_DELTA = 0.12
    cfg.HIGH_BIAS_CAL_MAX_COVERAGE_GAP_DELTA = 0.03
    cfg.ENABLE_CRISIS_TAIL_FUSION = False
    cfg.CRISIS_TAIL_FUSION_HIGH_THRESHOLDS = (0.30, 0.45)
    cfg.CRISIS_TAIL_FUSION_CRISIS_THRESHOLDS = (0.12, 0.22, 0.32)
    cfg.CRISIS_TAIL_FUSION_GAMMAS = (0.75, 1.00)
    cfg.CRISIS_TAIL_FUSION_SBP_QUANTILES = (0.75, 0.90, 1.00)
    cfg.CRISIS_TAIL_FUSION_DBP_QUANTILES = (0.70, 0.85, 1.00)
    cfg.CRISIS_TAIL_FUSION_CRISIS_GAINS = (1.00, 1.35, 1.70)
    cfg.CRISIS_TAIL_FUSION_SBP_MARGINS = (0.0, 1.5, 3.0, 4.5)
    cfg.CRISIS_TAIL_FUSION_DBP_MARGINS = (0.0, 0.75, 1.5)
    cfg.CRISIS_TAIL_FUSION_UNCERTAINTY_GAINS = (0.0, 0.25, 0.50)
    cfg.CRISIS_TAIL_FUSION_MAX_MAE_DELTA = 0.18
    cfg.CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA = 0.04
    return cfg


def fit_regressor_ensemble_safe(x: np.ndarray, y_residual: np.ndarray, seed: int):
    models = [
        stage_script.RandomForestRegressor(
            n_estimators=500,
            max_depth=16,
            min_samples_leaf=6,
            random_state=seed,
            n_jobs=1,
        ),
        stage_script.ExtraTreesRegressor(
            n_estimators=700,
            max_depth=18,
            min_samples_leaf=4,
            random_state=seed + 17,
            n_jobs=1,
        ),
    ]
    for model in models:
        model.fit(x, y_residual)
    return models


def fit_classifier_ensemble_safe(x: np.ndarray, y_cls: np.ndarray, seed: int):
    models = [
        stage_script.RandomForestClassifier(
            n_estimators=700,
            max_depth=16,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        ),
        stage_script.ExtraTreesClassifier(
            n_estimators=900,
            max_depth=20,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed + 29,
            n_jobs=1,
        ),
    ]
    for model in models:
        model.fit(x, y_cls)
    return models


def patch_stage_script_for_windows_runtime():
    # Avoid joblib multiprocessing/thread-pool startup failures in locked-down Windows environments.
    stage_script.fit_regressor_ensemble = fit_regressor_ensemble_safe
    stage_script.fit_classifier_ensemble = fit_classifier_ensemble_safe


def _entropy(prob: np.ndarray) -> np.ndarray:
    prob = bridge_script.normalize_prob(prob)
    return -np.sum(prob * np.log(np.clip(prob, 1.0e-8, 1.0)), axis=1, keepdims=True).astype(np.float32)


def _margin(prob: np.ndarray) -> np.ndarray:
    sorted_prob = np.sort(np.asarray(prob, dtype=np.float32), axis=1)
    return (sorted_prob[:, -1] - sorted_prob[:, -2]).reshape(-1, 1).astype(np.float32)


def _confidence(prob: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(prob, dtype=np.float32), axis=1, keepdims=True).astype(np.float32)


def _one_hot_from_prob(prob: np.ndarray, n_classes: int) -> np.ndarray:
    pred = np.asarray(prob, dtype=np.float32).argmax(axis=1).astype(np.int64)
    return np.eye(n_classes, dtype=np.float32)[pred]


def _agreement(a_prob: np.ndarray, b_prob: np.ndarray) -> np.ndarray:
    a = np.asarray(a_prob, dtype=np.float32).argmax(axis=1)
    b = np.asarray(b_prob, dtype=np.float32).argmax(axis=1)
    return (a == b).astype(np.float32).reshape(-1, 1)


def _apply_policy_row(prob: np.ndarray, policy_row: dict | None) -> np.ndarray:
    prob = bridge_script.normalize_prob(prob)
    if policy_row is None:
        return prob
    return guided_script.apply_policy(
        prob,
        float(policy_row["gamma"]),
        (
            float(policy_row["w_normal"]),
            float(policy_row["w_elevated"]),
            float(policy_row["w_stage1"]),
            float(policy_row["w_stage2"]),
        ),
    )


def predict_guided_bundle(
    head_model,
    head_state: dict,
    bank: dict,
    cfg,
    prefix: str,
    temperature: float,
    policy_row: dict | None,
):
    bank_norm = guided_script.normalize_bank(bank, head_state)
    loader = guided_script.build_plain_loader(bank_norm, cfg.HEAD_EVAL_BATCH_SIZE)
    reg_mu = guided_script.state_tensor_to_device(head_state["reg_mu"], cfg.DEVICE)
    reg_sigma = guided_script.state_tensor_to_device(head_state["reg_sigma"], cfg.DEVICE)

    head_model.eval()
    prob_parts: List[np.ndarray] = []
    bp_parts: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _, _, _ in loader:
            xb = xb.to(cfg.DEVICE)
            logits, _, bp_proxy = head_model(xb)
            prob = torch.softmax(logits / float(temperature), dim=1)
            bp_pred = bp_proxy * reg_sigma + reg_mu
            prob_parts.append(prob.cpu().numpy())
            bp_parts.append(bp_pred.cpu().numpy())

    prob_raw = np.concatenate(prob_parts, axis=0).astype(np.float32)
    prob = _apply_policy_row(prob_raw, policy_row)
    pred = prob.argmax(axis=1).astype(np.int64)
    bp_pred = np.concatenate(bp_parts, axis=0).astype(np.float32)
    metrics = stage_script.risk_classification_metrics(bank["y"].numpy(), pred, prob, cfg, prefix=prefix)
    proxy_metrics = guided_script.proxy_regression_metrics(bank["y_reg"].numpy(), bp_pred, prefix=prefix)
    return {
        "prob_raw": prob_raw,
        "prob": prob,
        "pred": pred,
        "bp_pred": bp_pred,
        "metrics": metrics,
        "proxy_metrics": proxy_metrics,
    }


def meta_classification_feature_names(cfg) -> List[str]:
    names: List[str] = []
    for prefix in ("opt_raw_prob", "opt_corr_prob", "dual_raw_prob", "dual_corr_prob", "dual_hybrid_prob", "guided_prob", "mean_prob"):
        for cls_name in cfg.CLASS_NAMES:
            names.append(f"{prefix}_{cls_name}")
    for prefix in ("dual_minus_opt", "guided_minus_hybrid"):
        for cls_name in cfg.CLASS_NAMES:
            names.append(f"{prefix}_{cls_name}")
    for prefix in ("opt_corr", "dual_corr", "guided_bp"):
        for target in ("sbp", "dbp"):
            names.append(f"{prefix}_{target}")
    for prefix in ("dual_minus_opt_bp", "guided_minus_dual_bp", "guided_minus_opt_bp"):
        for target in ("sbp", "dbp"):
            names.append(f"{prefix}_{target}")
    for prefix in ("opt_margin", "dual_margin", "hybrid_margin", "guided_margin", "opt_entropy", "dual_entropy", "hybrid_entropy", "guided_entropy"):
        names.append(prefix)
    for prefix in ("opt_alpha", "dual_alpha"):
        names.extend([f"{prefix}_{k}" for k in ("ppg", "ecg", "joint", "cross")])
    for prefix in ("opt_cred", "dual_cred"):
        names.extend([f"{prefix}_{k}" for k in ("low", "mid", "high")])
    names.extend(
        [
            "opt_quality",
            "dual_quality",
            "opt_uncertainty",
            "dual_uncertainty",
            "agree_opt_dual",
            "agree_dual_guided",
            "agree_hybrid_guided",
            "agree_all",
        ]
    )
    for prefix in ("opt_pred", "dual_pred", "hybrid_pred", "guided_pred"):
        for cls_name in cfg.CLASS_NAMES:
            names.append(f"{prefix}_{cls_name}")
    return names


def build_meta_classification_features(
    opt_raw: dict,
    opt_corr: dict,
    dual_raw: dict,
    dual_corr: dict,
    dual_hybrid_prob: np.ndarray,
    guided_prob: np.ndarray,
    guided_bp: np.ndarray,
    cfg,
) -> np.ndarray:
    opt_raw_prob = bridge_script.normalize_prob(np.asarray(opt_raw["y_prob_cls_from_reg"], dtype=np.float32))
    opt_corr_prob = bridge_script.normalize_prob(np.asarray(opt_corr["y_prob_cls_from_reg"], dtype=np.float32))
    dual_raw_prob = bridge_script.normalize_prob(np.asarray(dual_raw["y_prob_cls_from_reg"], dtype=np.float32))
    dual_corr_prob = bridge_script.normalize_prob(np.asarray(dual_corr["y_prob_cls_from_reg"], dtype=np.float32))
    dual_hybrid_prob = bridge_script.normalize_prob(np.asarray(dual_hybrid_prob, dtype=np.float32))
    guided_prob = bridge_script.normalize_prob(np.asarray(guided_prob, dtype=np.float32))
    mean_prob = bridge_script.normalize_prob((opt_corr_prob + dual_corr_prob + dual_hybrid_prob + guided_prob) / 4.0)

    opt_corr_pred = np.asarray(opt_corr["y_pred_reg"], dtype=np.float32)
    dual_corr_pred = np.asarray(dual_corr["y_pred_reg"], dtype=np.float32)
    guided_bp = np.asarray(guided_bp, dtype=np.float32)

    all_agree = (
        _agreement(opt_corr_prob, dual_corr_prob).reshape(-1)
        * _agreement(dual_hybrid_prob, guided_prob).reshape(-1)
        * _agreement(opt_corr_prob, guided_prob).reshape(-1)
    ).reshape(-1, 1)

    return np.concatenate(
        [
            opt_raw_prob,
            opt_corr_prob,
            dual_raw_prob,
            dual_corr_prob,
            dual_hybrid_prob,
            guided_prob,
            mean_prob,
            dual_corr_prob - opt_corr_prob,
            guided_prob - dual_hybrid_prob,
            opt_corr_pred,
            dual_corr_pred,
            guided_bp,
            dual_corr_pred - opt_corr_pred,
            guided_bp - dual_corr_pred,
            guided_bp - opt_corr_pred,
            _margin(opt_corr_prob),
            _margin(dual_corr_prob),
            _margin(dual_hybrid_prob),
            _margin(guided_prob),
            _entropy(opt_corr_prob),
            _entropy(dual_corr_prob),
            _entropy(dual_hybrid_prob),
            _entropy(guided_prob),
            np.asarray(opt_corr["alpha"], dtype=np.float32),
            np.asarray(dual_corr["alpha"], dtype=np.float32),
            np.asarray(opt_corr["credibility"], dtype=np.float32),
            np.asarray(dual_corr["credibility"], dtype=np.float32),
            np.asarray(opt_corr["quality"], dtype=np.float32).reshape(-1, 1),
            np.asarray(dual_corr["quality"], dtype=np.float32).reshape(-1, 1),
            np.asarray(opt_corr["uncertainty"], dtype=np.float32).reshape(-1, 1),
            np.asarray(dual_corr["uncertainty"], dtype=np.float32).reshape(-1, 1),
            _agreement(opt_corr_prob, dual_corr_prob),
            _agreement(dual_corr_prob, guided_prob),
            _agreement(dual_hybrid_prob, guided_prob),
            all_agree.astype(np.float32),
            _one_hot_from_prob(opt_corr_prob, cfg.N_CLASSES),
            _one_hot_from_prob(dual_corr_prob, cfg.N_CLASSES),
            _one_hot_from_prob(dual_hybrid_prob, cfg.N_CLASSES),
            _one_hot_from_prob(guided_prob, cfg.N_CLASSES),
        ],
        axis=1,
    ).astype(np.float32)


def meta_regression_feature_names(cfg) -> List[str]:
    names = meta_classification_feature_names(cfg)
    names.extend(["base_pred_sbp", "base_pred_dbp"])
    for cls_name in cfg.CLASS_NAMES:
        names.append(f"selected_cls_prob_{cls_name}")
    for cls_name in cfg.CLASS_NAMES:
        names.append(f"selected_cls_pred_{cls_name}")
    names.extend(["selected_confidence", "selected_margin", "selected_entropy"])
    for prefix in ("base_minus_opt", "base_minus_dual", "base_minus_guided"):
        for target in ("sbp", "dbp"):
            names.append(f"{prefix}_{target}")
    return names


def build_meta_regression_features(
    base_out: dict,
    opt_raw: dict,
    opt_corr: dict,
    dual_raw: dict,
    dual_corr: dict,
    dual_hybrid_prob: np.ndarray,
    guided_prob: np.ndarray,
    guided_bp: np.ndarray,
    selected_cls_prob: np.ndarray,
    cfg,
) -> np.ndarray:
    cls_feats = build_meta_classification_features(
        opt_raw,
        opt_corr,
        dual_raw,
        dual_corr,
        dual_hybrid_prob,
        guided_prob,
        guided_bp,
        cfg,
    )
    base_pred = np.asarray(base_out["y_pred_reg"], dtype=np.float32)
    opt_pred = np.asarray(opt_corr["y_pred_reg"], dtype=np.float32)
    dual_pred = np.asarray(dual_corr["y_pred_reg"], dtype=np.float32)
    guided_bp = np.asarray(guided_bp, dtype=np.float32)
    selected_cls_prob = bridge_script.normalize_prob(np.asarray(selected_cls_prob, dtype=np.float32))
    return np.concatenate(
        [
            cls_feats,
            base_pred,
            selected_cls_prob,
            _one_hot_from_prob(selected_cls_prob, cfg.N_CLASSES),
            _confidence(selected_cls_prob),
            _margin(selected_cls_prob),
            _entropy(selected_cls_prob),
            base_pred - opt_pred,
            base_pred - dual_pred,
            base_pred - guided_bp,
        ],
        axis=1,
    ).astype(np.float32)


def classification_arbiter_feature_names(cfg) -> List[str]:
    names: List[str] = []
    for prefix in ("selected_prob", "stability_prob", "mean_prob", "delta_prob", "abs_delta_prob"):
        for cls_name in cfg.CLASS_NAMES:
            names.append(f"{prefix}_{cls_name}")
    names.extend(
        [
            "selected_conf",
            "stability_conf",
            "conf_gap",
            "selected_margin",
            "stability_margin",
            "margin_gap",
            "selected_entropy",
            "stability_entropy",
            "entropy_gap",
            "agree_pred",
        ]
    )
    for prefix in ("selected_pred", "stability_pred"):
        for cls_name in cfg.CLASS_NAMES:
            names.append(f"{prefix}_{cls_name}")
    return names


def build_classification_arbiter_features(selected_prob: np.ndarray, stability_prob: np.ndarray, cfg) -> np.ndarray:
    selected_prob = bridge_script.normalize_prob(np.asarray(selected_prob, dtype=np.float32))
    stability_prob = bridge_script.normalize_prob(np.asarray(stability_prob, dtype=np.float32))
    mean_prob = bridge_script.normalize_prob(0.5 * (selected_prob + stability_prob))
    delta_prob = selected_prob - stability_prob
    abs_delta_prob = np.abs(delta_prob)
    return np.concatenate(
        [
            selected_prob,
            stability_prob,
            mean_prob,
            delta_prob,
            abs_delta_prob,
            _confidence(selected_prob),
            _confidence(stability_prob),
            _confidence(selected_prob) - _confidence(stability_prob),
            _margin(selected_prob),
            _margin(stability_prob),
            _margin(selected_prob) - _margin(stability_prob),
            _entropy(selected_prob),
            _entropy(stability_prob),
            _entropy(selected_prob) - _entropy(stability_prob),
            _agreement(selected_prob, stability_prob),
            _one_hot_from_prob(selected_prob, cfg.N_CLASSES),
            _one_hot_from_prob(stability_prob, cfg.N_CLASSES),
        ],
        axis=1,
    ).astype(np.float32)


def fit_classification_arbiter_bundle(
    selected_prob: np.ndarray,
    stability_prob: np.ndarray,
    y_true: np.ndarray,
    cfg,
    seed: int,
):
    x = build_classification_arbiter_features(selected_prob, stability_prob, cfg)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    selected_prob = bridge_script.normalize_prob(np.asarray(selected_prob, dtype=np.float32))
    stability_prob = bridge_script.normalize_prob(np.asarray(stability_prob, dtype=np.float32))

    selected_pred = selected_prob.argmax(axis=1)
    stability_pred = stability_prob.argmax(axis=1)
    selected_conf = np.max(selected_prob, axis=1)
    stability_conf = np.max(stability_prob, axis=1)
    selected_ok = selected_pred == y_true
    stability_ok = stability_pred == y_true

    target = np.full(len(y_true), 0.50, dtype=np.float32)
    target[np.logical_and(selected_ok, np.logical_not(stability_ok))] = 1.0
    target[np.logical_and(stability_ok, np.logical_not(selected_ok))] = 0.0

    both_ok = np.logical_and(selected_ok, stability_ok)
    both_bad = np.logical_and(np.logical_not(selected_ok), np.logical_not(stability_ok))
    target[both_ok] = np.clip(0.50 + 0.35 * (selected_conf[both_ok] - stability_conf[both_ok]), 0.15, 0.85)
    target[both_bad] = np.clip(0.50 + 0.20 * (selected_conf[both_bad] - stability_conf[both_bad]), 0.10, 0.90)

    models = stage_script.fit_regressor_ensemble(x, target.reshape(-1), seed=seed)
    return {
        "models": models,
        "feature_names": classification_arbiter_feature_names(cfg),
    }


def predict_classification_arbiter_weight(
    arbiter_bundle: dict,
    selected_prob: np.ndarray,
    stability_prob: np.ndarray,
    row: dict,
    cfg,
) -> np.ndarray:
    x = build_classification_arbiter_features(selected_prob, stability_prob, cfg)
    raw = np.asarray(stage_script.predict_regressor_ensemble(arbiter_bundle["models"], x), dtype=np.float32).reshape(-1)
    raw = np.clip(raw, 0.0, 1.0)

    scale = float(row["arb_scale"])
    beta = float(row["arb_beta"])
    floor = float(row["arb_floor"])
    agree_shrink = float(row["arb_agree_shrink"])
    max_override = float(getattr(cfg, "CLS_ARBITER_MAX_OVERRIDE", 0.97))

    agree = (
        np.asarray(selected_prob, dtype=np.float32).argmax(axis=1)
        == np.asarray(stability_prob, dtype=np.float32).argmax(axis=1)
    ).astype(np.float32)
    weight = floor + (scale - floor) * np.power(raw, beta)
    weight = np.where(agree > 0.5, weight * agree_shrink, weight)
    return np.clip(weight, 0.0, max_override).astype(np.float32).reshape(-1, 1)


def blend_classification_arbiter_prob(
    selected_prob: np.ndarray,
    stability_prob: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    weight = np.asarray(weight, dtype=np.float32).reshape(-1, 1)
    blended = weight * np.asarray(selected_prob, dtype=np.float32) + (1.0 - weight) * np.asarray(stability_prob, dtype=np.float32)
    return bridge_script.normalize_prob(blended)


def apply_classification_arbiter_prob(
    arbiter_bundle: dict,
    selected_prob: np.ndarray,
    stability_prob: np.ndarray,
    row: dict,
    cfg,
) -> np.ndarray:
    weight = predict_classification_arbiter_weight(
        arbiter_bundle,
        selected_prob,
        stability_prob,
        row,
        cfg,
    )
    blended = blend_classification_arbiter_prob(selected_prob, stability_prob, weight)
    policy_row = row if all(key in row for key in ("gamma", "w_normal", "w_elevated", "w_stage1", "w_stage2")) else None
    return _apply_policy_row(blended, policy_row)


def search_classification_arbiter(
    arbiter_bundle: dict,
    calib_selected_prob: np.ndarray,
    calib_stability_prob: np.ndarray,
    query_selected_prob_map: Dict[str, np.ndarray],
    query_stability_prob_map: Dict[str, np.ndarray],
    y_map: Dict[str, np.ndarray],
    cfg,
) -> tuple[dict, List[dict]]:
    rows: List[dict] = []
    best_row = None
    for arb_scale in tuple(float(x) for x in cfg.CLS_ARBITER_SCALES):
        for arb_beta in tuple(float(x) for x in cfg.CLS_ARBITER_BETAS):
            for arb_floor in tuple(float(x) for x in cfg.CLS_ARBITER_FLOORS):
                for arb_agree_shrink in tuple(float(x) for x in cfg.CLS_ARBITER_AGREE_SHRINKS):
                    probe_row = {
                        "candidate": (
                            f"cls_arbiter_s{int(round(100.0 * arb_scale))}"
                            f"_b{str(arb_beta).replace('.', 'p')}"
                            f"_f{str(arb_floor).replace('.', 'p')}"
                            f"_a{str(arb_agree_shrink).replace('.', 'p')}"
                        ),
                        "arb_scale": float(arb_scale),
                        "arb_beta": float(arb_beta),
                        "arb_floor": float(arb_floor),
                        "arb_agree_shrink": float(arb_agree_shrink),
                    }
                    calib_weight = predict_classification_arbiter_weight(
                        arbiter_bundle,
                        calib_selected_prob,
                        calib_stability_prob,
                        probe_row,
                        cfg,
                    )
                    calib_prob = blend_classification_arbiter_prob(
                        calib_selected_prob,
                        calib_stability_prob,
                        calib_weight,
                    )
                    policy_row, _ = guided_script.search_policy(calib_prob, y_map["calib"], cfg)
                    clean_weight = predict_classification_arbiter_weight(
                        arbiter_bundle,
                        query_selected_prob_map["clean"],
                        query_stability_prob_map["clean"],
                        probe_row,
                        cfg,
                    )
                    noise_weight = predict_classification_arbiter_weight(
                        arbiter_bundle,
                        query_selected_prob_map["noise"],
                        query_stability_prob_map["noise"],
                        probe_row,
                        cfg,
                    )
                    ecg_weight = predict_classification_arbiter_weight(
                        arbiter_bundle,
                        query_selected_prob_map["ecg"],
                        query_stability_prob_map["ecg"],
                        probe_row,
                        cfg,
                    )
                    ppg_weight = predict_classification_arbiter_weight(
                        arbiter_bundle,
                        query_selected_prob_map["ppg"],
                        query_stability_prob_map["ppg"],
                        probe_row,
                        cfg,
                    )

                    clean_prob = blend_classification_arbiter_prob(
                        query_selected_prob_map["clean"],
                        query_stability_prob_map["clean"],
                        clean_weight,
                    )
                    clean_prob = _apply_policy_row(clean_prob, policy_row)
                    noise_prob = blend_classification_arbiter_prob(
                        query_selected_prob_map["noise"],
                        query_stability_prob_map["noise"],
                        noise_weight,
                    )
                    noise_prob = _apply_policy_row(noise_prob, policy_row)
                    ecg_prob = blend_classification_arbiter_prob(
                        query_selected_prob_map["ecg"],
                        query_stability_prob_map["ecg"],
                        ecg_weight,
                    )
                    ecg_prob = _apply_policy_row(ecg_prob, policy_row)
                    ppg_prob = blend_classification_arbiter_prob(
                        query_selected_prob_map["ppg"],
                        query_stability_prob_map["ppg"],
                        ppg_weight,
                    )
                    ppg_prob = _apply_policy_row(ppg_prob, policy_row)

                    clean_metrics = stage_script.risk_classification_metrics(
                        y_map["clean"],
                        clean_prob.argmax(axis=1).astype(np.int64),
                        clean_prob,
                        cfg,
                        prefix="selected_val",
                    )
                    noise_metrics = stage_script.risk_classification_metrics(
                        y_map["noise"],
                        noise_prob.argmax(axis=1).astype(np.int64),
                        noise_prob,
                        cfg,
                        prefix="selected_noise_val",
                    )
                    ecg_metrics = stage_script.risk_classification_metrics(
                        y_map["ecg"],
                        ecg_prob.argmax(axis=1).astype(np.int64),
                        ecg_prob,
                        cfg,
                        prefix="selected_ecg_val",
                    )
                    ppg_metrics = stage_script.risk_classification_metrics(
                        y_map["ppg"],
                        ppg_prob.argmax(axis=1).astype(np.int64),
                        ppg_prob,
                        cfg,
                        prefix="selected_ppg_val",
                    )
                    score = prev_script.robust_classification_score(clean_metrics, noise_metrics, ecg_metrics, ppg_metrics, cfg)
                    row = {
                        **probe_row,
                        "score": float(score),
                        "weight_selected_mean": float(clean_weight.mean()),
                        "weight_selected_p90": float(np.quantile(clean_weight.reshape(-1), 0.90)),
                        "gamma": float(policy_row["gamma"]),
                        "w_normal": float(policy_row["w_normal"]),
                        "w_elevated": float(policy_row["w_elevated"]),
                        "w_stage1": float(policy_row["w_stage1"]),
                        "w_stage2": float(policy_row["w_stage2"]),
                        **guided_script.class_summary(clean_metrics, "selected_val"),
                        "noise_f1": float(noise_metrics["cls_f1_macro_selected_noise_val"]),
                        "ecg_missing_f1": float(ecg_metrics["cls_f1_macro_selected_ecg_val"]),
                        "ppg_missing_f1": float(ppg_metrics["cls_f1_macro_selected_ppg_val"]),
                        "robust_min_f1": float(
                            min(
                                noise_metrics["cls_f1_macro_selected_noise_val"],
                                ecg_metrics["cls_f1_macro_selected_ecg_val"],
                                ppg_metrics["cls_f1_macro_selected_ppg_val"],
                            )
                        ),
                    }
                    rows.append(row)
                    if best_row is None or float(row["score"]) > float(best_row["score"]):
                        best_row = row

    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    if best_row is None:
        raise RuntimeError("No classification arbiter candidate was evaluated.")
    return best_row, rows


def regression_router_expert_names() -> List[str]:
    return ["selected_base", "stability_selected", "opt_corr", "dual_corr", "guided_bp"]


def regression_router_feature_names(cfg) -> List[str]:
    names = meta_regression_feature_names(cfg)
    expert_names = regression_router_expert_names()
    for expert_name in expert_names:
        for target_name in ("sbp", "dbp"):
            names.append(f"{expert_name}_{target_name}")
    for left_idx, left in enumerate(expert_names):
        for right in expert_names[left_idx + 1 :]:
            for target_name in ("sbp", "dbp"):
                names.append(f"{left}_minus_{right}_{target_name}")
    names.extend(
        [
            "expert_mean_sbp",
            "expert_mean_dbp",
            "expert_std_sbp",
            "expert_std_dbp",
            "expert_spread_sbp",
            "expert_spread_dbp",
        ]
    )
    return names


def build_regression_router_features(
    base_out: dict,
    stability_out: dict,
    opt_raw: dict,
    opt_corr: dict,
    dual_raw: dict,
    dual_corr: dict,
    dual_hybrid_prob: np.ndarray,
    guided_prob: np.ndarray,
    guided_bp: np.ndarray,
    selected_cls_prob: np.ndarray,
    cfg,
) -> np.ndarray:
    meta_feats = build_meta_regression_features(
        base_out,
        opt_raw,
        opt_corr,
        dual_raw,
        dual_corr,
        dual_hybrid_prob,
        guided_prob,
        guided_bp,
        selected_cls_prob,
        cfg,
    )
    expert_preds = [
        np.asarray(base_out["y_pred_reg"], dtype=np.float32),
        np.asarray(stability_out["y_pred_reg"], dtype=np.float32),
        np.asarray(opt_corr["y_pred_reg"], dtype=np.float32),
        np.asarray(dual_corr["y_pred_reg"], dtype=np.float32),
        np.asarray(guided_bp, dtype=np.float32),
    ]
    pairwise = []
    expert_names = regression_router_expert_names()
    for left_idx, _ in enumerate(expert_names):
        for right_idx in range(left_idx + 1, len(expert_names)):
            pairwise.append(expert_preds[left_idx] - expert_preds[right_idx])
    expert_stack = np.stack(expert_preds, axis=1)
    expert_mean = np.mean(expert_stack, axis=1)
    expert_std = np.std(expert_stack, axis=1)
    expert_spread = np.max(expert_stack, axis=1) - np.min(expert_stack, axis=1)
    return np.concatenate(
        [
            meta_feats,
            *expert_preds,
            *pairwise,
            expert_mean,
            expert_std,
            expert_spread,
        ],
        axis=1,
    ).astype(np.float32)


def fit_regression_router_bundle(
    calib_inputs: dict,
    cfg,
    seed: int,
):
    x = build_regression_router_features(
        calib_inputs["base_out"],
        calib_inputs["stability_out"],
        calib_inputs["opt_raw"],
        calib_inputs["opt_corr"],
        calib_inputs["dual_raw"],
        calib_inputs["dual_corr"],
        calib_inputs["dual_hybrid_prob"],
        calib_inputs["guided_prob"],
        calib_inputs["guided_bp"],
        calib_inputs["selected_cls_prob"],
        cfg,
    )
    y_true = np.asarray(calib_inputs["base_out"]["y_true_reg"], dtype=np.float32)
    expert_stack = np.stack(
        [
            np.asarray(calib_inputs["base_out"]["y_pred_reg"], dtype=np.float32),
            np.asarray(calib_inputs["stability_out"]["y_pred_reg"], dtype=np.float32),
            np.asarray(calib_inputs["opt_corr"]["y_pred_reg"], dtype=np.float32),
            np.asarray(calib_inputs["dual_corr"]["y_pred_reg"], dtype=np.float32),
            np.asarray(calib_inputs["guided_bp"], dtype=np.float32),
        ],
        axis=1,
    )
    target = np.abs(expert_stack - y_true[:, None, :]).reshape(len(y_true), -1)
    models = stage_script.fit_regressor_ensemble(x, target, seed=seed)
    return {
        "models": models,
        "feature_names": regression_router_feature_names(cfg),
        "expert_names": regression_router_expert_names(),
    }


def predict_regression_router_errors(
    router_bundle: dict,
    inputs: dict,
    cfg,
) -> np.ndarray:
    x = build_regression_router_features(
        inputs["base_out"],
        inputs["stability_out"],
        inputs["opt_raw"],
        inputs["opt_corr"],
        inputs["dual_raw"],
        inputs["dual_corr"],
        inputs["dual_hybrid_prob"],
        inputs["guided_prob"],
        inputs["guided_bp"],
        inputs["selected_cls_prob"],
        cfg,
    )
    pred = np.asarray(stage_script.predict_regressor_ensemble(router_bundle["models"], x), dtype=np.float32)
    pred = pred.reshape(x.shape[0], len(router_bundle["expert_names"]), 2)
    return np.clip(pred, 0.25, 30.0)


def regression_router_weights(predicted_abs_error: np.ndarray, row: dict) -> np.ndarray:
    predicted_abs_error = np.asarray(predicted_abs_error, dtype=np.float32)
    n_experts = predicted_abs_error.shape[1]
    temp = max(float(row["router_temp"]), 1.0e-4)
    gamma = float(row["router_gamma"])
    floor = float(row["router_floor"])
    blend_scale = float(row["router_blend_scale"])

    weights = np.exp(-predicted_abs_error / temp)
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1.0e-6, None)
    if abs(gamma - 1.0) > 1.0e-8:
        weights = np.power(weights, gamma, dtype=np.float32)
        weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1.0e-6, None)
    if floor > 0.0:
        weights = (1.0 - floor) * weights + floor / float(n_experts)
        weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1.0e-6, None)

    anchor = np.zeros_like(weights, dtype=np.float32)
    anchor[:, 0, :] = 1.0
    weights = (1.0 - blend_scale) * anchor + blend_scale * weights
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1.0e-6, None)
    return weights.astype(np.float32)


def apply_regression_router_correction(
    row: dict,
    predicted_abs_error: np.ndarray,
    inputs: dict,
    cfg,
):
    weights = regression_router_weights(predicted_abs_error, row)
    expert_stack = np.stack(
        [
            np.asarray(inputs["base_out"]["y_pred_reg"], dtype=np.float32),
            np.asarray(inputs["stability_out"]["y_pred_reg"], dtype=np.float32),
            np.asarray(inputs["opt_corr"]["y_pred_reg"], dtype=np.float32),
            np.asarray(inputs["dual_corr"]["y_pred_reg"], dtype=np.float32),
            np.asarray(inputs["guided_bp"], dtype=np.float32),
        ],
        axis=1,
    )
    pred = stage_script.clipped_regression_prediction(np.sum(weights * expert_stack, axis=1))
    out = stage_script.clone_regression_output(inputs["base_out"], pred, cfg)
    out["router_weights"] = weights.astype(np.float32)
    out["router_expert_names"] = tuple(regression_router_expert_names())
    out["router_entropy"] = float(
        np.mean(-np.sum(weights * np.log(np.clip(weights, 1.0e-8, 1.0)), axis=1))
    )
    out["router_weight_selected_mean"] = float(weights[:, 0, :].mean())
    out["router_weight_stability_mean"] = float(weights[:, 1, :].mean())
    out["router_weight_guided_mean"] = float(weights[:, -1, :].mean())
    return out


def feature_importance_rows(models, feature_names: Sequence[str]) -> List[dict]:
    importances = np.mean([np.asarray(model.feature_importances_, dtype=np.float32) for model in models], axis=0)
    rows = [{"feature": str(name), "importance": float(score)} for name, score in zip(feature_names, importances)]
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows


def fit_meta_classifier(
    opt_raw: dict,
    opt_corr: dict,
    dual_raw: dict,
    dual_corr: dict,
    dual_hybrid_prob: np.ndarray,
    guided_prob: np.ndarray,
    guided_bp: np.ndarray,
    cfg,
    seed: int,
):
    x = build_meta_classification_features(opt_raw, opt_corr, dual_raw, dual_corr, dual_hybrid_prob, guided_prob, guided_bp, cfg)
    y = np.asarray(opt_corr["y_true_cls"], dtype=np.int64)
    models = stage_script.fit_classifier_ensemble(x, y, seed=seed)
    return {
        "models": models,
        "feature_names": meta_classification_feature_names(cfg),
    }


def fit_meta_residual_models(
    base_calib: dict,
    opt_raw_calib: dict,
    opt_corr_calib: dict,
    dual_raw_calib: dict,
    dual_corr_calib: dict,
    dual_hybrid_calib_prob: np.ndarray,
    guided_calib_prob: np.ndarray,
    guided_calib_bp: np.ndarray,
    selected_cls_calib_prob: np.ndarray,
    cfg,
    seed: int,
):
    x = build_meta_regression_features(
        base_calib,
        opt_raw_calib,
        opt_corr_calib,
        dual_raw_calib,
        dual_corr_calib,
        dual_hybrid_calib_prob,
        guided_calib_prob,
        guided_calib_bp,
        selected_cls_calib_prob,
        cfg,
    )
    residual = (
        np.asarray(base_calib["y_true_reg"], dtype=np.float32)
        - np.asarray(base_calib["y_pred_reg"], dtype=np.float32)
    )
    models = stage_script.fit_regressor_ensemble(x, residual, seed=seed)
    return {
        "models": models,
        "feature_names": meta_regression_feature_names(cfg),
    }


def predict_meta_classifier_prob(
    classifier_bundle: dict,
    opt_raw: dict,
    opt_corr: dict,
    dual_raw: dict,
    dual_corr: dict,
    dual_hybrid_prob: np.ndarray,
    guided_prob: np.ndarray,
    guided_bp: np.ndarray,
    cfg,
) -> np.ndarray:
    x = build_meta_classification_features(
        opt_raw,
        opt_corr,
        dual_raw,
        dual_corr,
        dual_hybrid_prob,
        guided_prob,
        guided_bp,
        cfg,
    )
    return stage_script.predict_classifier_ensemble(classifier_bundle["models"], x, cfg.N_CLASSES)


def predict_meta_residual(
    residual_bundle: dict,
    base_out: dict,
    opt_raw: dict,
    opt_corr: dict,
    dual_raw: dict,
    dual_corr: dict,
    dual_hybrid_prob: np.ndarray,
    guided_prob: np.ndarray,
    guided_bp: np.ndarray,
    selected_cls_prob: np.ndarray,
    cfg,
) -> np.ndarray:
    x = build_meta_regression_features(
        base_out,
        opt_raw,
        opt_corr,
        dual_raw,
        dual_corr,
        dual_hybrid_prob,
        guided_prob,
        guided_bp,
        selected_cls_prob,
        cfg,
    )
    return np.asarray(stage_script.predict_regressor_ensemble(residual_bundle["models"], x), dtype=np.float32)


def blend_probabilities_with_policy(
    base_prob: np.ndarray,
    meta_prob: np.ndarray,
    weight_meta: float,
    policy_row: dict | None,
) -> np.ndarray:
    blended = (
        (1.0 - float(weight_meta)) * np.asarray(base_prob, dtype=np.float32)
        + float(weight_meta) * np.asarray(meta_prob, dtype=np.float32)
    )
    return _apply_policy_row(blended, policy_row)


def search_classification_blend(
    calib_base_prob: np.ndarray,
    calib_meta_prob: np.ndarray,
    query_prob_map: Dict[str, np.ndarray],
    meta_prob_map: Dict[str, np.ndarray],
    y_map: Dict[str, np.ndarray],
    cfg,
) -> tuple[dict, List[dict]]:
    rows: List[dict] = []
    best_row = None
    for weight_meta in tuple(float(x) for x in cfg.META_BLEND_WEIGHTS):
        calib_prob = blend_probabilities_with_policy(calib_base_prob, calib_meta_prob, weight_meta, policy_row=None)
        policy_row, _ = guided_script.search_policy(calib_prob, y_map["calib"], cfg)

        clean_prob = blend_probabilities_with_policy(query_prob_map["clean"], meta_prob_map["clean"], weight_meta, policy_row)
        noise_prob = blend_probabilities_with_policy(query_prob_map["noise"], meta_prob_map["noise"], weight_meta, policy_row)
        ecg_prob = blend_probabilities_with_policy(query_prob_map["ecg"], meta_prob_map["ecg"], weight_meta, policy_row)
        ppg_prob = blend_probabilities_with_policy(query_prob_map["ppg"], meta_prob_map["ppg"], weight_meta, policy_row)

        clean_metrics = stage_script.risk_classification_metrics(y_map["clean"], clean_prob.argmax(axis=1).astype(np.int64), clean_prob, cfg, prefix="selected_val")
        noise_metrics = stage_script.risk_classification_metrics(y_map["noise"], noise_prob.argmax(axis=1).astype(np.int64), noise_prob, cfg, prefix="selected_noise_val")
        ecg_metrics = stage_script.risk_classification_metrics(y_map["ecg"], ecg_prob.argmax(axis=1).astype(np.int64), ecg_prob, cfg, prefix="selected_ecg_val")
        ppg_metrics = stage_script.risk_classification_metrics(y_map["ppg"], ppg_prob.argmax(axis=1).astype(np.int64), ppg_prob, cfg, prefix="selected_ppg_val")
        score = prev_script.robust_classification_score(clean_metrics, noise_metrics, ecg_metrics, ppg_metrics, cfg)

        row = {
            "weight_meta": float(weight_meta),
            "score": float(score),
            "gamma": float(policy_row["gamma"]),
            "w_normal": float(policy_row["w_normal"]),
            "w_elevated": float(policy_row["w_elevated"]),
            "w_stage1": float(policy_row["w_stage1"]),
            "w_stage2": float(policy_row["w_stage2"]),
            **guided_script.class_summary(clean_metrics, "selected_val"),
            "noise_f1": float(noise_metrics["cls_f1_macro_selected_noise_val"]),
            "ecg_missing_f1": float(ecg_metrics["cls_f1_macro_selected_ecg_val"]),
            "ppg_missing_f1": float(ppg_metrics["cls_f1_macro_selected_ppg_val"]),
            "robust_min_f1": float(
                min(
                    noise_metrics["cls_f1_macro_selected_noise_val"],
                    ecg_metrics["cls_f1_macro_selected_ecg_val"],
                    ppg_metrics["cls_f1_macro_selected_ppg_val"],
                )
            ),
        }
        rows.append(row)
        if best_row is None or float(row["score"]) > float(best_row["score"]):
            best_row = row

    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    if best_row is None:
        raise RuntimeError("No classification blend candidate was evaluated.")
    return best_row, rows


def apply_meta_residual_correction(
    row: dict,
    residual_pred: np.ndarray,
    base_out: dict,
    selected_cls_prob: np.ndarray,
    cfg,
):
    residual_pred = np.asarray(residual_pred, dtype=np.float32)
    residual_pred[:, 0] = np.clip(residual_pred[:, 0], -float(cfg.META_RESID_CLIP_SBP), float(cfg.META_RESID_CLIP_SBP))
    residual_pred[:, 1] = np.clip(residual_pred[:, 1], -float(cfg.META_RESID_CLIP_DBP), float(cfg.META_RESID_CLIP_DBP))
    conf = np.clip(np.max(np.asarray(selected_cls_prob, dtype=np.float32), axis=1), 1.0e-6, 1.0)
    gate = float(row["scale"]) * np.power(conf, float(row["conf_beta"]))
    pred = stage_script.clipped_regression_prediction(
        np.asarray(base_out["y_pred_reg"], dtype=np.float32) + gate.reshape(-1, 1) * residual_pred
    )
    out = stage_script.clone_regression_output(base_out, pred, cfg)
    out["meta_gate_mean"] = float(np.mean(gate))
    out["meta_gate_p90"] = float(np.quantile(gate, 0.90))
    return out


def search_meta_regression_candidates(
    residual_bundle: dict,
    calib_inputs: dict,
    query_inputs: dict,
    cfg,
) -> List[dict]:
    calib_residual = predict_meta_residual(
        residual_bundle,
        calib_inputs["base_out"],
        calib_inputs["opt_raw"],
        calib_inputs["opt_corr"],
        calib_inputs["dual_raw"],
        calib_inputs["dual_corr"],
        calib_inputs["dual_hybrid_prob"],
        calib_inputs["guided_prob"],
        calib_inputs["guided_bp"],
        calib_inputs["selected_cls_prob"],
        cfg,
    )
    query_residual = predict_meta_residual(
        residual_bundle,
        query_inputs["base_out"],
        query_inputs["opt_raw"],
        query_inputs["opt_corr"],
        query_inputs["dual_raw"],
        query_inputs["dual_corr"],
        query_inputs["dual_hybrid_prob"],
        query_inputs["guided_prob"],
        query_inputs["guided_bp"],
        query_inputs["selected_cls_prob"],
        cfg,
    )

    rows: List[dict] = []
    base_conformal = stage_script.summarize_conformal_tradeoff(calib_inputs["base_out"], query_inputs["base_out"], cfg)
    rows.append(
        {
            "candidate": "selected_base",
            "scale": 0.0,
            "conf_beta": 0.0,
            "score": float(bridge_script.regression_selection_score(query_inputs["base_out"]["metrics_reg"], base_conformal)),
            "gate_mean": 0.0,
            "gate_p90": 0.0,
            **query_inputs["base_out"]["metrics_reg"],
            **base_conformal,
        }
    )

    for scale in tuple(float(x) for x in cfg.META_RESIDUAL_SCALES):
        if scale <= 0.0:
            continue
        for conf_beta in tuple(float(x) for x in cfg.META_CONF_BETAS):
            row = {
                "candidate": f"meta_res_s{int(round(100.0 * scale))}_b{str(conf_beta).replace('.', 'p')}",
                "scale": scale,
                "conf_beta": conf_beta,
            }
            calib_adj = apply_meta_residual_correction(row, calib_residual, calib_inputs["base_out"], calib_inputs["selected_cls_prob"], cfg)
            query_adj = apply_meta_residual_correction(row, query_residual, query_inputs["base_out"], query_inputs["selected_cls_prob"], cfg)
            conformal = stage_script.summarize_conformal_tradeoff(calib_adj, query_adj, cfg)
            score = (
                bridge_script.regression_selection_score(query_adj["metrics_reg"], conformal)
                + 0.010 * float(query_adj.get("meta_gate_mean", 0.0))
            )
            rows.append(
                {
                    **row,
                    "score": float(score),
                    "gate_mean": float(query_adj.get("meta_gate_mean", 0.0)),
                    "gate_p90": float(query_adj.get("meta_gate_p90", 0.0)),
                    **query_adj["metrics_reg"],
                    **conformal,
                }
            )

    rows.sort(key=lambda row: float(row["score"]))
    return rows


def search_regression_router_candidates(
    router_bundle: dict,
    calib_inputs: dict,
    query_inputs: dict,
    cfg,
) -> List[dict]:
    calib_errors = predict_regression_router_errors(router_bundle, calib_inputs, cfg)
    query_errors = predict_regression_router_errors(router_bundle, query_inputs, cfg)
    rows: List[dict] = []
    for router_blend_scale in tuple(float(x) for x in cfg.REG_ROUTER_BLEND_SCALES):
        for router_temp in tuple(float(x) for x in cfg.REG_ROUTER_TEMPS):
            for router_gamma in tuple(float(x) for x in cfg.REG_ROUTER_GAMMAS):
                for router_floor in tuple(float(x) for x in cfg.REG_ROUTER_FLOORS):
                    row = {
                        "candidate": (
                            f"reg_router_s{int(round(100.0 * router_blend_scale))}"
                            f"_t{str(router_temp).replace('.', 'p')}"
                            f"_g{str(router_gamma).replace('.', 'p')}"
                            f"_f{str(router_floor).replace('.', 'p')}"
                        ),
                        "router_blend_scale": float(router_blend_scale),
                        "router_temp": float(router_temp),
                        "router_gamma": float(router_gamma),
                        "router_floor": float(router_floor),
                    }
                    calib_adj = apply_regression_router_correction(row, calib_errors, calib_inputs, cfg)
                    query_adj = apply_regression_router_correction(row, query_errors, query_inputs, cfg)
                    conformal = stage_script.summarize_conformal_tradeoff(calib_adj, query_adj, cfg)
                    score = bridge_script.regression_selection_score(query_adj["metrics_reg"], conformal)
                    rows.append(
                        {
                            **row,
                            "score": float(score),
                            "router_entropy": float(query_adj.get("router_entropy", 0.0)),
                            "router_weight_selected_mean": float(query_adj.get("router_weight_selected_mean", 0.0)),
                            "router_weight_stability_mean": float(query_adj.get("router_weight_stability_mean", 0.0)),
                            "router_weight_guided_mean": float(query_adj.get("router_weight_guided_mean", 0.0)),
                            **query_adj["metrics_reg"],
                            **conformal,
                        }
                    )
    rows.sort(key=lambda row: float(row["score"]))
    return rows


def apply_best_regression_candidate(
    row: dict,
    meta_residual_pred: np.ndarray | None,
    base_out: dict,
    selected_cls_prob: np.ndarray,
    router_errors: np.ndarray | None,
    router_inputs: dict | None,
    cfg,
):
    candidate = str(row["candidate"])
    if candidate == "selected_base":
        return base_out
    if candidate.startswith("meta_res_"):
        if meta_residual_pred is None:
            raise ValueError("Meta residual prediction is required for meta_res candidates.")
        return apply_meta_residual_correction(row, meta_residual_pred, base_out, selected_cls_prob, cfg)
    if candidate.startswith("reg_router_"):
        if router_errors is None or router_inputs is None:
            raise ValueError("Regression router inputs are required for reg_router candidates.")
        return apply_regression_router_correction(row, router_errors, router_inputs, cfg)
    raise ValueError(f"Unsupported regression candidate: {candidate}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def fit_weighted_regressor_ensemble_safe(x: np.ndarray, y_residual: np.ndarray, sample_weight: np.ndarray, seed: int):
    models = [
        stage_script.RandomForestRegressor(
            n_estimators=350,
            max_depth=14,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=1,
        ),
        stage_script.ExtraTreesRegressor(
            n_estimators=500,
            max_depth=16,
            min_samples_leaf=4,
            random_state=seed + 23,
            n_jobs=1,
        ),
    ]
    sample_weight = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    for model in models:
        model.fit(x, y_residual, sample_weight=sample_weight)
    return models


def risk_guard_feature_names(cfg) -> List[str]:
    names = ["pred_sbp", "pred_dbp"]
    for cls_name in cfg.CLASS_NAMES:
        names.append(f"cls_prob_{cls_name}")
    for cls_name in cfg.CLASS_NAMES:
        names.append(f"cls_pred_{cls_name}")
    names.extend(
        [
            "confidence",
            "margin",
            "entropy",
            "uncertainty",
            "quality",
            "sbp_minus_120",
            "sbp_minus_140",
            "sbp_minus_180",
            "dbp_minus_80",
            "dbp_minus_90",
            "dbp_minus_120",
            "sbp_above_120",
            "sbp_above_140",
            "sbp_above_180",
            "dbp_above_80",
            "dbp_above_90",
            "dbp_above_120",
            "high_signal",
            "crisis_signal",
        ]
    )
    return names


def risk_guard_signals(reg_out: dict, cls_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(reg_out["y_pred_reg"], dtype=np.float32)
    cls_prob = bridge_script.normalize_prob(np.asarray(cls_prob, dtype=np.float32))
    high_signal = np.maximum.reduce(
        [
            np.clip(cls_prob[:, 2] + 0.85 * cls_prob[:, 3], 0.0, 1.5),
            _sigmoid((pred[:, 0] - 138.0) / 7.5),
            _sigmoid((pred[:, 1] - 88.0) / 5.5),
        ]
    ).astype(np.float32)
    crisis_signal = np.maximum.reduce(
        [
            np.clip(cls_prob[:, 3], 0.0, 1.0),
            _sigmoid((pred[:, 0] - 170.0) / 5.5),
            _sigmoid((pred[:, 1] - 110.0) / 4.5),
        ]
    ).astype(np.float32)
    return np.clip(high_signal, 0.0, 1.0), np.clip(crisis_signal, 0.0, 1.0)


def build_risk_guard_features(reg_out: dict, cls_prob: np.ndarray, cfg) -> np.ndarray:
    pred = np.asarray(reg_out["y_pred_reg"], dtype=np.float32)
    cls_prob = bridge_script.normalize_prob(np.asarray(cls_prob, dtype=np.float32))
    pred_one_hot = _one_hot_from_prob(cls_prob, cfg.N_CLASSES)
    confidence = _confidence(cls_prob)
    margin = _margin(cls_prob)
    entropy = _entropy(cls_prob)
    uncertainty = np.asarray(reg_out.get("uncertainty", np.zeros(len(pred), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
    quality = np.asarray(reg_out.get("quality", np.ones(len(pred), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
    high_signal, crisis_signal = risk_guard_signals(reg_out, cls_prob)

    sbp = pred[:, [0]]
    dbp = pred[:, [1]]
    sbp_minus_120 = sbp - 120.0
    sbp_minus_140 = sbp - 140.0
    sbp_minus_180 = sbp - 180.0
    dbp_minus_80 = dbp - 80.0
    dbp_minus_90 = dbp - 90.0
    dbp_minus_120 = dbp - 120.0

    return np.concatenate(
        [
            pred,
            cls_prob,
            pred_one_hot,
            confidence,
            margin,
            entropy,
            uncertainty,
            quality,
            sbp_minus_120,
            sbp_minus_140,
            sbp_minus_180,
            dbp_minus_80,
            dbp_minus_90,
            dbp_minus_120,
            np.clip(sbp_minus_120, 0.0, None),
            np.clip(sbp_minus_140, 0.0, None),
            np.clip(sbp_minus_180, 0.0, None),
            np.clip(dbp_minus_80, 0.0, None),
            np.clip(dbp_minus_90, 0.0, None),
            np.clip(dbp_minus_120, 0.0, None),
            high_signal.reshape(-1, 1),
            crisis_signal.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)


def risk_guard_sample_weights(reg_out: dict, cfg) -> np.ndarray:
    y_true = np.asarray(reg_out["y_true_reg"], dtype=np.float32)
    y_pred = np.asarray(reg_out["y_pred_reg"], dtype=np.float32)
    high_true = ((y_true[:, 0] >= 140.0) | (y_true[:, 1] >= 90.0)).astype(np.float32)
    crisis_true = ((y_true[:, 0] >= 180.0) | (y_true[:, 1] >= 120.0)).astype(np.float32)
    under_sbp = np.clip(y_true[:, 0] - y_pred[:, 0], 0.0, None) / 5.0
    under_dbp = np.clip(y_true[:, 1] - y_pred[:, 1], 0.0, None) / 3.0
    weight = (
        1.0
        + float(cfg.RISK_GUARD_HIGH_TRUE_WEIGHT) * high_true
        + float(cfg.RISK_GUARD_CRISIS_TRUE_WEIGHT) * crisis_true
        + float(cfg.RISK_GUARD_HIGH_UNDER_WEIGHT)
        * (
            high_true * (under_sbp + 0.60 * under_dbp)
            + crisis_true * (1.20 * under_sbp + 0.80 * under_dbp)
        )
    )
    return np.clip(weight.astype(np.float32), 1.0, 40.0)


def fit_risk_guard_bundle(calib_out: dict, calib_cls_prob: np.ndarray, cfg, seed: int):
    x = build_risk_guard_features(calib_out, calib_cls_prob, cfg)
    target = np.asarray(calib_out["y_true_reg"], dtype=np.float32) - np.asarray(calib_out["y_pred_reg"], dtype=np.float32)
    sample_weight = risk_guard_sample_weights(calib_out, cfg)
    models = fit_weighted_regressor_ensemble_safe(x, target, sample_weight, seed=seed)
    return {
        "models": models,
        "feature_names": risk_guard_feature_names(cfg),
    }


def predict_risk_guard_delta(risk_guard_bundle: dict, reg_out: dict, cls_prob: np.ndarray, cfg) -> np.ndarray:
    x = build_risk_guard_features(reg_out, cls_prob, cfg)
    pred = np.asarray(stage_script.predict_regressor_ensemble(risk_guard_bundle["models"], x), dtype=np.float32)
    return pred.reshape(len(x), 2)


def apply_risk_guard_correction(
    row: dict,
    delta_pred: np.ndarray | None,
    reg_out: dict,
    cls_prob: np.ndarray,
    cfg,
):
    candidate = str(row["candidate"])
    if candidate == "identity" or delta_pred is None or float(row.get("scale", 0.0)) <= 0.0:
        out = dict(reg_out)
        out["risk_guard_gate_mean"] = 0.0
        out["risk_guard_high_active_rate"] = 0.0
        out["risk_guard_crisis_active_rate"] = 0.0
        return out

    delta_pred = np.asarray(delta_pred, dtype=np.float32)
    pred = np.asarray(reg_out["y_pred_reg"], dtype=np.float32)
    high_signal, crisis_signal = risk_guard_signals(reg_out, cls_prob)
    gate = float(row["scale"]) * np.power(np.clip(high_signal, 1.0e-5, 1.0), float(row["beta"]))
    gate *= (
        1.0
        + (float(row["high_gain"]) - 1.0) * high_signal
        + (float(row["crisis_gain"]) - 1.0) * crisis_signal
    )
    gate = np.clip(gate.astype(np.float32), 0.0, 3.5)

    delta = gate.reshape(-1, 1) * delta_pred
    neg_clip_sbp = -float(cfg.RISK_GUARD_MAX_NEG_FRAC) * float(cfg.RISK_GUARD_MAX_SHIFT_SBP)
    neg_clip_dbp = -float(cfg.RISK_GUARD_MAX_NEG_FRAC) * float(cfg.RISK_GUARD_MAX_SHIFT_DBP)
    delta[:, 0] = np.clip(delta[:, 0], neg_clip_sbp, float(cfg.RISK_GUARD_MAX_SHIFT_SBP))
    delta[:, 1] = np.clip(delta[:, 1], neg_clip_dbp, float(cfg.RISK_GUARD_MAX_SHIFT_DBP))

    high_mask = high_signal >= 0.45
    crisis_mask = crisis_signal >= 0.30
    delta[high_mask, 0] = np.maximum(delta[high_mask, 0], -0.02 * float(cfg.RISK_GUARD_MAX_SHIFT_SBP))
    delta[high_mask, 1] = np.maximum(delta[high_mask, 1], -0.02 * float(cfg.RISK_GUARD_MAX_SHIFT_DBP))
    delta[crisis_mask, 0] = np.maximum(delta[crisis_mask, 0], 0.0)
    delta[crisis_mask, 1] = np.maximum(delta[crisis_mask, 1], 0.0)

    corrected = stage_script.clipped_regression_prediction(pred + delta)
    out = stage_script.clone_regression_output(reg_out, corrected, cfg)
    out["risk_guard_gate_mean"] = float(np.mean(gate))
    out["risk_guard_high_active_rate"] = float(np.mean(high_mask))
    out["risk_guard_crisis_active_rate"] = float(np.mean(crisis_mask))
    return out


def clinical_underestimation_penalty(bp_range_rows: List[dict]) -> float:
    row_map = {str(row["bp_range"]): row for row in bp_range_rows}
    penalty = 0.0
    elevated = row_map.get("elevated")
    if elevated is not None:
        penalty += 0.20 * max(0.0, -float(elevated["bias_sbp"]))
        penalty += 0.10 * max(0.0, -float(elevated["bias_dbp"]))
    high = row_map.get("high")
    if high is not None:
        penalty += 1.25 * max(0.0, -float(high["bias_sbp"]))
        penalty += 0.70 * max(0.0, -float(high["bias_dbp"]))
    crisis = row_map.get("crisis")
    if crisis is not None:
        penalty += 1.90 * max(0.0, -float(crisis["bias_sbp"]))
        penalty += 1.10 * max(0.0, -float(crisis["bias_dbp"]))
    return float(penalty)


def risk_guard_cost(calib_out: dict, query_out: dict, base_ref: dict, cfg):
    conformal = stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    bp_range_rows = stage_script.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"])
    clinical_pen = clinical_underestimation_penalty(bp_range_rows)
    tail_pen = prev_script.tail_bias_penalty(bp_range_rows)
    reg = query_out["metrics_reg"]
    mae_excess = max(0.0, float(reg["mae_mean"]) - float(base_ref["mae_mean"]) - float(cfg.RISK_GUARD_MAX_MAE_DELTA))
    cov_excess = max(
        0.0,
        float(conformal["coverage_gap"]) - float(base_ref["coverage_gap"]) - float(cfg.RISK_GUARD_MAX_COVERAGE_GAP_DELTA),
    )
    score = float(
        clinical_pen
        + 0.55 * tail_pen
        + 22.0 * mae_excess
        + 8.0 * cov_excess
        + 0.015 * float(reg["mae_mean"])
        + 0.008 * float(conformal["miw_mean"])
    )
    return float(score), conformal, bp_range_rows, float(clinical_pen), float(tail_pen)


def search_risk_guard_candidates(
    risk_guard_bundle: dict,
    calib_out: dict,
    calib_cls_prob: np.ndarray,
    query_out: dict,
    query_cls_prob: np.ndarray,
    cfg,
) -> tuple[dict, List[dict]]:
    calib_delta = predict_risk_guard_delta(risk_guard_bundle, calib_out, calib_cls_prob, cfg)
    query_delta = predict_risk_guard_delta(risk_guard_bundle, query_out, query_cls_prob, cfg)
    base_conformal = stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    base_bp_range_rows = stage_script.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"])
    base_clinical_pen = clinical_underestimation_penalty(base_bp_range_rows)
    base_tail_pen = prev_script.tail_bias_penalty(base_bp_range_rows)
    base_ref = {
        "mae_mean": float(query_out["metrics_reg"]["mae_mean"]),
        "coverage_gap": float(base_conformal["coverage_gap"]),
    }
    row_map = {str(row["bp_range"]): row for row in base_bp_range_rows}
    rows: List[dict] = [
        {
            "candidate": "identity",
            "scale": 0.0,
            "beta": 0.0,
            "high_gain": 1.0,
            "crisis_gain": 1.0,
            "score": float(base_clinical_pen + 0.55 * base_tail_pen + 0.015 * float(query_out["metrics_reg"]["mae_mean"]) + 0.008 * float(base_conformal["miw_mean"])),
            "clinical_under_penalty": float(base_clinical_pen),
            "tail_bias_penalty": float(base_tail_pen),
            "high_bias_sbp": float(row_map.get("high", {}).get("bias_sbp", 0.0)),
            "high_bias_dbp": float(row_map.get("high", {}).get("bias_dbp", 0.0)),
            "crisis_bias_sbp": float(row_map.get("crisis", {}).get("bias_sbp", 0.0)),
            "crisis_bias_dbp": float(row_map.get("crisis", {}).get("bias_dbp", 0.0)),
            "guard_gate_mean": 0.0,
            "guard_high_active_rate": 0.0,
            "guard_crisis_active_rate": 0.0,
            **query_out["metrics_reg"],
            **base_conformal,
        }
    ]

    for scale in tuple(float(x) for x in cfg.RISK_GUARD_SCALES):
        if scale <= 0.0:
            continue
        for beta in tuple(float(x) for x in cfg.RISK_GUARD_BETAS):
            for high_gain in tuple(float(x) for x in cfg.RISK_GUARD_HIGH_GAINS):
                for crisis_gain in tuple(float(x) for x in cfg.RISK_GUARD_CRISIS_GAINS):
                    row = {
                        "candidate": (
                            f"risk_guard_s{int(round(100.0 * scale))}"
                            f"_b{str(beta).replace('.', 'p')}"
                            f"_h{str(high_gain).replace('.', 'p')}"
                            f"_c{str(crisis_gain).replace('.', 'p')}"
                        ),
                        "scale": float(scale),
                        "beta": float(beta),
                        "high_gain": float(high_gain),
                        "crisis_gain": float(crisis_gain),
                    }
                    calib_adj = apply_risk_guard_correction(row, calib_delta, calib_out, calib_cls_prob, cfg)
                    query_adj = apply_risk_guard_correction(row, query_delta, query_out, query_cls_prob, cfg)
                    score, conformal, bp_range_rows, clinical_pen, tail_pen = risk_guard_cost(calib_adj, query_adj, base_ref, cfg)
                    range_map = {str(item["bp_range"]): item for item in bp_range_rows}
                    rows.append(
                        {
                            **row,
                            "score": float(score),
                            "clinical_under_penalty": float(clinical_pen),
                            "tail_bias_penalty": float(tail_pen),
                            "high_bias_sbp": float(range_map.get("high", {}).get("bias_sbp", 0.0)),
                            "high_bias_dbp": float(range_map.get("high", {}).get("bias_dbp", 0.0)),
                            "crisis_bias_sbp": float(range_map.get("crisis", {}).get("bias_sbp", 0.0)),
                            "crisis_bias_dbp": float(range_map.get("crisis", {}).get("bias_dbp", 0.0)),
                            "guard_gate_mean": float(query_adj.get("risk_guard_gate_mean", 0.0)),
                            "guard_high_active_rate": float(query_adj.get("risk_guard_high_active_rate", 0.0)),
                            "guard_crisis_active_rate": float(query_adj.get("risk_guard_crisis_active_rate", 0.0)),
                            **query_adj["metrics_reg"],
                            **conformal,
                        }
                    )

    rows.sort(key=lambda row: float(row["score"]))
    return rows[0], rows


def _signal_ramp(signal: np.ndarray, threshold: float, gamma: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    scaled = np.clip((signal - float(threshold)) / max(1.0e-6, 1.0 - float(threshold)), 0.0, 1.0)
    return np.power(scaled, float(gamma), dtype=np.float32)


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
        return out

    pred = np.asarray(reg_out["y_pred_reg"], dtype=np.float32)
    high_signal, crisis_signal = risk_guard_signals(reg_out, cls_prob)
    high_gate = _signal_ramp(high_signal, float(row["high_threshold"]), float(row["gamma"]))
    crisis_gate = _signal_ramp(crisis_signal, float(row["crisis_threshold"]), float(row["gamma"]))

    delta = np.zeros_like(pred, dtype=np.float32)
    delta[:, 0] = float(row["sbp_high_shift"]) * high_gate + float(row["sbp_crisis_shift"]) * crisis_gate
    delta[:, 1] = float(row["dbp_high_shift"]) * high_gate + float(row["dbp_crisis_shift"]) * crisis_gate
    delta[:, 0] = np.clip(delta[:, 0], 0.0, float(cfg.RISK_GUARD_MAX_SHIFT_SBP))
    delta[:, 1] = np.clip(delta[:, 1], 0.0, float(cfg.RISK_GUARD_MAX_SHIFT_DBP))

    corrected = stage_script.clipped_regression_prediction(pred + delta)
    out = stage_script.clone_regression_output(reg_out, corrected, cfg)
    out["high_bias_cal_shift_mean_sbp"] = float(delta[:, 0].mean())
    out["high_bias_cal_shift_mean_dbp"] = float(delta[:, 1].mean())
    return out


def high_bias_calibration_cost(calib_out: dict, query_out: dict, base_ref: dict, cfg):
    conformal = stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    bp_range_rows = stage_script.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"])
    clinical_pen = clinical_underestimation_penalty(bp_range_rows)
    tail_pen = prev_script.tail_bias_penalty(bp_range_rows)
    range_map = {str(row["bp_range"]): row for row in bp_range_rows}
    high_row = range_map.get("high", {})
    crisis_row = range_map.get("crisis", {})
    high_abs_pen = 1.50 * abs(float(high_row.get("bias_sbp", 0.0))) + 0.60 * abs(float(high_row.get("bias_dbp", 0.0)))
    crisis_abs_pen = 0.90 * abs(float(crisis_row.get("bias_sbp", 0.0))) + 0.35 * abs(float(crisis_row.get("bias_dbp", 0.0)))
    reg = query_out["metrics_reg"]
    mae_excess = max(
        0.0,
        float(reg["mae_mean"]) - float(base_ref["mae_mean"]) - float(cfg.HIGH_BIAS_CAL_MAX_MAE_DELTA),
    )
    cov_excess = max(
        0.0,
        float(conformal["coverage_gap"]) - float(base_ref["coverage_gap"]) - float(cfg.HIGH_BIAS_CAL_MAX_COVERAGE_GAP_DELTA),
    )
    score = float(
        clinical_pen
        + 0.80 * high_abs_pen
        + 0.45 * crisis_abs_pen
        + 0.45 * tail_pen
        + 20.0 * mae_excess
        + 8.0 * cov_excess
        + 0.012 * float(reg["mae_mean"])
    )
    return float(score), conformal, bp_range_rows, float(clinical_pen), float(tail_pen)


def search_high_bias_calibration_candidates(
    calib_out: dict,
    calib_cls_prob: np.ndarray,
    query_out: dict,
    query_cls_prob: np.ndarray,
    cfg,
) -> tuple[dict, List[dict]]:
    base_conformal = stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    base_bp_range_rows = stage_script.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"])
    base_clinical_pen = clinical_underestimation_penalty(base_bp_range_rows)
    base_tail_pen = prev_script.tail_bias_penalty(base_bp_range_rows)
    base_ref = {
        "mae_mean": float(query_out["metrics_reg"]["mae_mean"]),
        "coverage_gap": float(base_conformal["coverage_gap"]),
    }
    range_map = {str(row["bp_range"]): row for row in base_bp_range_rows}
    rows: List[dict] = [
        {
            "candidate": "identity",
            "high_threshold": 0.0,
            "crisis_threshold": 0.0,
            "gamma": 1.0,
            "sbp_high_shift": 0.0,
            "sbp_crisis_shift": 0.0,
            "dbp_high_shift": 0.0,
            "dbp_crisis_shift": 0.0,
            "score": float(base_clinical_pen + 0.45 * base_tail_pen + 0.012 * float(query_out["metrics_reg"]["mae_mean"])),
            "clinical_under_penalty": float(base_clinical_pen),
            "tail_bias_penalty": float(base_tail_pen),
            "high_bias_sbp": float(range_map.get("high", {}).get("bias_sbp", 0.0)),
            "high_bias_dbp": float(range_map.get("high", {}).get("bias_dbp", 0.0)),
            "crisis_bias_sbp": float(range_map.get("crisis", {}).get("bias_sbp", 0.0)),
            "crisis_bias_dbp": float(range_map.get("crisis", {}).get("bias_dbp", 0.0)),
            "shift_mean_sbp": 0.0,
            "shift_mean_dbp": 0.0,
            **query_out["metrics_reg"],
            **base_conformal,
        }
    ]

    for high_threshold in tuple(float(x) for x in cfg.HIGH_BIAS_CAL_HIGH_THRESHOLDS):
        for crisis_threshold in tuple(float(x) for x in cfg.HIGH_BIAS_CAL_CRISIS_THRESHOLDS):
            for gamma in tuple(float(x) for x in cfg.HIGH_BIAS_CAL_GAMMAS):
                for sbp_high_shift in tuple(float(x) for x in cfg.HIGH_BIAS_CAL_SBP_HIGH_SHIFTS):
                    for sbp_crisis_shift in tuple(float(x) for x in cfg.HIGH_BIAS_CAL_SBP_CRISIS_SHIFTS):
                        for dbp_high_shift in tuple(float(x) for x in cfg.HIGH_BIAS_CAL_DBP_HIGH_SHIFTS):
                            for dbp_crisis_shift in tuple(float(x) for x in cfg.HIGH_BIAS_CAL_DBP_CRISIS_SHIFTS):
                                if (
                                    sbp_high_shift <= 0.0
                                    and sbp_crisis_shift <= 0.0
                                    and dbp_high_shift <= 0.0
                                    and dbp_crisis_shift <= 0.0
                                ):
                                    continue
                                row = {
                                    "candidate": (
                                        f"high_bias_ht{str(high_threshold).replace('.', 'p')}"
                                        f"_ct{str(crisis_threshold).replace('.', 'p')}"
                                        f"_g{str(gamma).replace('.', 'p')}"
                                        f"_sh{str(sbp_high_shift).replace('.', 'p')}"
                                        f"_sc{str(sbp_crisis_shift).replace('.', 'p')}"
                                        f"_dh{str(dbp_high_shift).replace('.', 'p')}"
                                        f"_dc{str(dbp_crisis_shift).replace('.', 'p')}"
                                    ),
                                    "high_threshold": float(high_threshold),
                                    "crisis_threshold": float(crisis_threshold),
                                    "gamma": float(gamma),
                                    "sbp_high_shift": float(sbp_high_shift),
                                    "sbp_crisis_shift": float(sbp_crisis_shift),
                                    "dbp_high_shift": float(dbp_high_shift),
                                    "dbp_crisis_shift": float(dbp_crisis_shift),
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
                                        **query_adj["metrics_reg"],
                                        **conformal,
                                    }
                                )

    rows.sort(key=lambda row: float(row["score"]))
    return rows[0], rows


def plot_risk_guard_frontier(
    rows: Sequence[dict],
    fig_dir: Path,
    filename: str = "high_risk_guard_frontier.png",
):
    if not rows:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    clinical_pen = np.asarray([float(row.get("clinical_under_penalty", 0.0)) for row in rows], dtype=np.float32)
    mae = np.asarray([float(row.get("mae_mean", 0.0)) for row in rows], dtype=np.float32)
    score = np.asarray([float(row.get("score", 0.0)) for row in rows], dtype=np.float32)
    score_span = float(max(1.0e-6, score.max() - score.min()))
    sizes = 40.0 + 160.0 * (score.max() - score) / score_span
    ax.scatter(clinical_pen, mae, s=sizes, alpha=0.60, color="#c0392b", edgecolor="white", linewidth=0.6)
    best_idx = int(np.argmin(score))
    ax.scatter(clinical_pen[best_idx], mae[best_idx], s=float(sizes[best_idx]) + 80.0, color="#1e8449", edgecolor="black", linewidth=1.0)
    ax.annotate("Best guard", (float(clinical_pen[best_idx]), float(mae[best_idx])), textcoords="offset points", xytext=(8, 8), fontsize=10, weight="bold")
    for idx, row in enumerate(rows):
        if str(row["candidate"]) == "identity":
            ax.annotate("Identity", (float(clinical_pen[idx]), float(mae[idx])), textcoords="offset points", xytext=(8, -14), fontsize=9)
            break
    ax.set_xlabel("Clinical underestimation penalty")
    ax.set_ylabel("Mean absolute error (mmHg)")
    ax.set_title("High-Risk Guard Frontier")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_high_bias_calibration_frontier(
    rows: Sequence[dict],
    fig_dir: Path,
    filename: str = "high_bias_calibration_frontier.png",
):
    if not rows:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    high_bias = np.asarray([abs(float(row.get("high_bias_sbp", 0.0))) for row in rows], dtype=np.float32)
    mae = np.asarray([float(row.get("mae_mean", 0.0)) for row in rows], dtype=np.float32)
    score = np.asarray([float(row.get("score", 0.0)) for row in rows], dtype=np.float32)
    score_span = float(max(1.0e-6, score.max() - score.min()))
    sizes = 40.0 + 160.0 * (score.max() - score) / score_span
    ax.scatter(high_bias, mae, s=sizes, alpha=0.60, color="#8e44ad", edgecolor="white", linewidth=0.6)
    best_idx = int(np.argmin(score))
    ax.scatter(
        high_bias[best_idx],
        mae[best_idx],
        s=float(sizes[best_idx]) + 80.0,
        color="#1e8449",
        edgecolor="black",
        linewidth=1.0,
    )
    ax.annotate(
        "Best bias calibrator",
        (float(high_bias[best_idx]), float(mae[best_idx])),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=10,
        weight="bold",
    )
    ax.set_xlabel("|High-range SBP bias| (mmHg)")
    ax.set_ylabel("Mean absolute error (mmHg)")
    ax.set_title("High-Bias Calibration Frontier")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def crisis_tail_expert_names() -> List[str]:
    return ["selected_current", "selected_base", "stability_selected", "opt_corr", "dual_corr", "guided_bp"]


def build_crisis_tail_expert_stack(reg_out: dict, reg_inputs: dict) -> tuple[np.ndarray, List[str]]:
    expert_preds = [
        np.asarray(reg_out["y_pred_reg"], dtype=np.float32),
        np.asarray(reg_inputs["base_out"]["y_pred_reg"], dtype=np.float32),
        np.asarray(reg_inputs["stability_out"]["y_pred_reg"], dtype=np.float32),
        np.asarray(reg_inputs["opt_corr"]["y_pred_reg"], dtype=np.float32),
        np.asarray(reg_inputs["dual_corr"]["y_pred_reg"], dtype=np.float32),
        np.asarray(reg_inputs["guided_bp"], dtype=np.float32),
    ]
    return np.stack(expert_preds, axis=1).astype(np.float32), crisis_tail_expert_names()


def build_crisis_tail_signal_context(
    reg_out: dict,
    cls_prob: np.ndarray,
    reg_inputs: dict,
) -> dict:
    high_signal, crisis_signal = risk_guard_signals(reg_out, cls_prob)
    expert_stack, expert_names = build_crisis_tail_expert_stack(reg_out, reg_inputs)
    expert_peak = np.max(expert_stack, axis=1)
    expert_q90 = np.quantile(expert_stack, 0.90, axis=1).astype(np.float32)
    spread = (np.max(expert_stack, axis=1) - np.min(expert_stack, axis=1)).astype(np.float32)
    spread_signal = np.clip(0.60 * spread[:, 0] / 12.0 + 0.40 * spread[:, 1] / 8.0, 0.0, 1.0).astype(np.float32)

    uncertainty = np.asarray(reg_out.get("uncertainty", np.zeros(len(expert_stack), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if len(uncertainty) > 0:
        unc_q50, unc_q90 = np.quantile(uncertainty, [0.50, 0.90])
        uncertainty_signal = np.clip(
            (uncertainty - float(unc_q50)) / max(1.0e-6, float(unc_q90) - float(unc_q50)),
            0.0,
            1.0,
        ).astype(np.float32)
    else:
        uncertainty_signal = np.zeros(len(expert_stack), dtype=np.float32)

    expert_high_signal = np.maximum.reduce(
        [
            np.asarray(high_signal, dtype=np.float32),
            _sigmoid((expert_peak[:, 0] - 142.0) / 6.5),
            _sigmoid((expert_peak[:, 1] - 92.0) / 5.0),
            _sigmoid((expert_q90[:, 0] - 138.0) / 7.0),
            _sigmoid((expert_q90[:, 1] - 88.0) / 5.0),
        ]
    ).astype(np.float32)
    expert_crisis_signal = np.maximum.reduce(
        [
            np.asarray(crisis_signal, dtype=np.float32),
            _sigmoid((expert_peak[:, 0] - 176.0) / 4.5),
            _sigmoid((expert_peak[:, 1] - 113.0) / 4.0),
            _sigmoid((expert_q90[:, 0] - 170.0) / 5.0),
            _sigmoid((expert_q90[:, 1] - 108.0) / 4.5),
        ]
    ).astype(np.float32)
    return {
        "expert_stack": np.asarray(expert_stack, dtype=np.float32),
        "expert_names": expert_names,
        "expert_high_signal": np.clip(expert_high_signal, 0.0, 1.0).astype(np.float32),
        "expert_crisis_signal": np.clip(expert_crisis_signal, 0.0, 1.0).astype(np.float32),
        "spread_signal": np.asarray(spread_signal, dtype=np.float32),
        "uncertainty_signal": np.asarray(uncertainty_signal, dtype=np.float32),
    }


def apply_crisis_tail_fusion(
    row: dict,
    reg_out: dict,
    cls_prob: np.ndarray,
    reg_inputs: dict,
    cfg,
):
    candidate = str(row["candidate"])
    if candidate == "identity":
        out = dict(reg_out)
        out["crisis_tail_fusion_gate_mean"] = 0.0
        out["crisis_tail_fusion_shift_mean_sbp"] = 0.0
        out["crisis_tail_fusion_shift_mean_dbp"] = 0.0
        out["crisis_tail_fusion_activation_rate"] = 0.0
        return out

    pred = np.asarray(reg_out["y_pred_reg"], dtype=np.float32)
    context = build_crisis_tail_signal_context(reg_out, cls_prob, reg_inputs)
    expert_stack = context["expert_stack"]
    high_gate = _signal_ramp(
        context["expert_high_signal"],
        float(row["high_threshold"]),
        float(row["gamma"]),
    )
    crisis_gate = _signal_ramp(
        context["expert_crisis_signal"],
        float(row["crisis_threshold"]),
        float(row["gamma"]),
    )
    gate_scale = (
        1.0
        + float(row["uncertainty_gain"]) * context["uncertainty_signal"]
        + 0.25 * context["spread_signal"]
    ).astype(np.float32)

    sbp_anchor = np.quantile(expert_stack[:, :, 0], float(row["sbp_quantile"]), axis=1).astype(np.float32)
    dbp_anchor = np.quantile(expert_stack[:, :, 1], float(row["dbp_quantile"]), axis=1).astype(np.float32)
    sbp_target = np.maximum(sbp_anchor, pred[:, 0]) + float(row["sbp_margin"]) * crisis_gate
    dbp_target = np.maximum(dbp_anchor, pred[:, 1]) + float(row["dbp_margin"]) * crisis_gate

    sbp_gate = np.clip(
        (0.35 * high_gate + float(row["crisis_gain"]) * crisis_gate) * gate_scale,
        0.0,
        1.0,
    ).astype(np.float32)
    dbp_gate = np.clip(
        (0.30 * high_gate + 0.80 * float(row["crisis_gain"]) * crisis_gate) * gate_scale,
        0.0,
        1.0,
    ).astype(np.float32)

    delta = np.zeros_like(pred, dtype=np.float32)
    delta[:, 0] = sbp_gate * np.clip(sbp_target - pred[:, 0], 0.0, None)
    delta[:, 1] = dbp_gate * np.clip(dbp_target - pred[:, 1], 0.0, None)
    delta[:, 0] = np.clip(delta[:, 0], 0.0, float(cfg.RISK_GUARD_MAX_SHIFT_SBP))
    delta[:, 1] = np.clip(delta[:, 1], 0.0, float(cfg.RISK_GUARD_MAX_SHIFT_DBP))

    corrected = stage_script.clipped_regression_prediction(pred + delta)
    out = stage_script.clone_regression_output(reg_out, corrected, cfg)
    out["crisis_tail_fusion_gate_mean"] = float(0.5 * (sbp_gate.mean() + dbp_gate.mean()))
    out["crisis_tail_fusion_shift_mean_sbp"] = float(delta[:, 0].mean())
    out["crisis_tail_fusion_shift_mean_dbp"] = float(delta[:, 1].mean())
    out["crisis_tail_fusion_activation_rate"] = float(np.mean((crisis_gate >= 0.15) | (high_gate >= 0.20)))
    return out


def crisis_tail_fusion_cost(calib_out: dict, query_out: dict, base_ref: dict, cfg):
    conformal = stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    bp_range_rows = stage_script.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"])
    range_map = {str(row["bp_range"]): row for row in bp_range_rows}
    high_row = range_map.get("high", {})
    crisis_row = range_map.get("crisis", {})
    clinical_pen = clinical_underestimation_penalty(bp_range_rows)
    tail_pen = prev_script.tail_bias_penalty(bp_range_rows)
    crisis_under_pen = (
        2.80 * max(0.0, -float(crisis_row.get("bias_sbp", 0.0)))
        + 1.60 * max(0.0, -float(crisis_row.get("bias_dbp", 0.0)))
    )
    crisis_abs_pen = (
        1.10 * abs(float(crisis_row.get("bias_sbp", 0.0)))
        + 0.55 * abs(float(crisis_row.get("bias_dbp", 0.0)))
    )
    high_under_pen = (
        1.20 * max(0.0, -float(high_row.get("bias_sbp", 0.0)))
        + 0.60 * max(0.0, -float(high_row.get("bias_dbp", 0.0)))
    )
    high_abs_pen = (
        0.45 * abs(float(high_row.get("bias_sbp", 0.0)))
        + 0.22 * abs(float(high_row.get("bias_dbp", 0.0)))
    )
    reg = query_out["metrics_reg"]
    mae_excess = max(
        0.0,
        float(reg["mae_mean"]) - float(base_ref["mae_mean"]) - float(cfg.CRISIS_TAIL_FUSION_MAX_MAE_DELTA),
    )
    cov_excess = max(
        0.0,
        float(conformal["coverage_gap"]) - float(base_ref["coverage_gap"]) - float(cfg.CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA),
    )
    score = float(
        crisis_under_pen
        + 0.65 * crisis_abs_pen
        + 0.55 * high_under_pen
        + 0.25 * high_abs_pen
        + 0.30 * clinical_pen
        + 0.25 * tail_pen
        + 20.0 * mae_excess
        + 8.0 * cov_excess
        + 0.010 * float(reg["mae_mean"])
    )
    return float(score), conformal, bp_range_rows, float(clinical_pen), float(tail_pen)


def search_crisis_tail_fusion_candidates(
    calib_out: dict,
    calib_cls_prob: np.ndarray,
    calib_reg_inputs: dict,
    query_out: dict,
    query_cls_prob: np.ndarray,
    query_reg_inputs: dict,
    cfg,
) -> tuple[dict, List[dict]]:
    base_conformal = stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    base_bp_range_rows = stage_script.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"])
    base_clinical_pen = clinical_underestimation_penalty(base_bp_range_rows)
    base_tail_pen = prev_script.tail_bias_penalty(base_bp_range_rows)
    base_ref = {
        "mae_mean": float(query_out["metrics_reg"]["mae_mean"]),
        "coverage_gap": float(base_conformal["coverage_gap"]),
    }
    range_map = {str(row["bp_range"]): row for row in base_bp_range_rows}
    rows: List[dict] = [
        {
            "candidate": "identity",
            "high_threshold": 0.0,
            "crisis_threshold": 0.0,
            "gamma": 1.0,
            "sbp_quantile": 0.0,
            "dbp_quantile": 0.0,
            "crisis_gain": 1.0,
            "sbp_margin": 0.0,
            "dbp_margin": 0.0,
            "uncertainty_gain": 0.0,
            "score": float(0.30 * base_clinical_pen + 0.25 * base_tail_pen + 0.010 * float(query_out["metrics_reg"]["mae_mean"])) ,
            "clinical_under_penalty": float(base_clinical_pen),
            "tail_bias_penalty": float(base_tail_pen),
            "high_bias_sbp": float(range_map.get("high", {}).get("bias_sbp", 0.0)),
            "high_bias_dbp": float(range_map.get("high", {}).get("bias_dbp", 0.0)),
            "crisis_bias_sbp": float(range_map.get("crisis", {}).get("bias_sbp", 0.0)),
            "crisis_bias_dbp": float(range_map.get("crisis", {}).get("bias_dbp", 0.0)),
            "fusion_gate_mean": 0.0,
            "shift_mean_sbp": 0.0,
            "shift_mean_dbp": 0.0,
            "activation_rate": 0.0,
            **query_out["metrics_reg"],
            **base_conformal,
        }
    ]

    for high_threshold in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_HIGH_THRESHOLDS):
        for crisis_threshold in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_CRISIS_THRESHOLDS):
            for gamma in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_GAMMAS):
                for sbp_quantile in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_SBP_QUANTILES):
                    for dbp_quantile in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_DBP_QUANTILES):
                        for crisis_gain in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_CRISIS_GAINS):
                            for sbp_margin in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_SBP_MARGINS):
                                for dbp_margin in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_DBP_MARGINS):
                                    for uncertainty_gain in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_UNCERTAINTY_GAINS):
                                        if (
                                            sbp_margin <= 0.0
                                            and dbp_margin <= 0.0
                                            and crisis_gain <= 1.0
                                            and uncertainty_gain <= 0.0
                                        ):
                                            continue
                                        row = {
                                            "candidate": (
                                                f"crisis_tail_ht{str(high_threshold).replace('.', 'p')}"
                                                f"_ct{str(crisis_threshold).replace('.', 'p')}"
                                                f"_g{str(gamma).replace('.', 'p')}"
                                                f"_sq{str(sbp_quantile).replace('.', 'p')}"
                                                f"_dq{str(dbp_quantile).replace('.', 'p')}"
                                                f"_cg{str(crisis_gain).replace('.', 'p')}"
                                                f"_sm{str(sbp_margin).replace('.', 'p')}"
                                                f"_dm{str(dbp_margin).replace('.', 'p')}"
                                                f"_ug{str(uncertainty_gain).replace('.', 'p')}"
                                            ),
                                            "high_threshold": float(high_threshold),
                                            "crisis_threshold": float(crisis_threshold),
                                            "gamma": float(gamma),
                                            "sbp_quantile": float(sbp_quantile),
                                            "dbp_quantile": float(dbp_quantile),
                                            "crisis_gain": float(crisis_gain),
                                            "sbp_margin": float(sbp_margin),
                                            "dbp_margin": float(dbp_margin),
                                            "uncertainty_gain": float(uncertainty_gain),
                                        }
                                        calib_adj = apply_crisis_tail_fusion(
                                            row,
                                            calib_out,
                                            calib_cls_prob,
                                            calib_reg_inputs,
                                            cfg,
                                        )
                                        query_adj = apply_crisis_tail_fusion(
                                            row,
                                            query_out,
                                            query_cls_prob,
                                            query_reg_inputs,
                                            cfg,
                                        )
                                        score, conformal, bp_range_rows, clinical_pen, tail_pen = crisis_tail_fusion_cost(
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
                                                "fusion_gate_mean": float(query_adj.get("crisis_tail_fusion_gate_mean", 0.0)),
                                                "shift_mean_sbp": float(query_adj.get("crisis_tail_fusion_shift_mean_sbp", 0.0)),
                                                "shift_mean_dbp": float(query_adj.get("crisis_tail_fusion_shift_mean_dbp", 0.0)),
                                                "activation_rate": float(query_adj.get("crisis_tail_fusion_activation_rate", 0.0)),
                                                **query_adj["metrics_reg"],
                                                **conformal,
                                            }
                                        )

    rows.sort(key=lambda row: float(row["score"]))
    return rows[0], rows


def plot_crisis_tail_fusion_frontier(
    rows: Sequence[dict],
    fig_dir: Path,
    filename: str = "crisis_tail_fusion_frontier.png",
):
    if not rows:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    crisis_bias = np.asarray([abs(float(row.get("crisis_bias_sbp", 0.0))) for row in rows], dtype=np.float32)
    mae = np.asarray([float(row.get("mae_mean", 0.0)) for row in rows], dtype=np.float32)
    score = np.asarray([float(row.get("score", 0.0)) for row in rows], dtype=np.float32)
    score_span = float(max(1.0e-6, score.max() - score.min()))
    sizes = 40.0 + 160.0 * (score.max() - score) / score_span
    ax.scatter(crisis_bias, mae, s=sizes, alpha=0.60, color="#d35400", edgecolor="white", linewidth=0.6)
    best_idx = int(np.argmin(score))
    ax.scatter(
        crisis_bias[best_idx],
        mae[best_idx],
        s=float(sizes[best_idx]) + 80.0,
        color="#1e8449",
        edgecolor="black",
        linewidth=1.0,
    )
    ax.annotate(
        "Best crisis-tail fusion",
        (float(crisis_bias[best_idx]), float(mae[best_idx])),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=10,
        weight="bold",
    )
    ax.set_xlabel("|Crisis-range SBP bias| (mmHg)")
    ax.set_ylabel("Mean absolute error (mmHg)")
    ax.set_title("Crisis-Tail Fusion Frontier")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_clinical_guard_comparison_rows(base_out: dict, guarded_out: dict) -> List[dict]:
    base_rows = {
        str(row["bp_range"]): row
        for row in stage_script.build_bp_range_table(base_out["y_true_reg"], base_out["y_pred_reg"])
    }
    guarded_rows = {
        str(row["bp_range"]): row
        for row in stage_script.build_bp_range_table(guarded_out["y_true_reg"], guarded_out["y_pred_reg"])
    }
    rows: List[dict] = []
    for bp_range in ("elevated", "high", "crisis"):
        base_row = base_rows.get(bp_range, {})
        guarded_row = guarded_rows.get(bp_range, {})
        rows.append(
            {
                "bp_range": bp_range,
                "n": int(guarded_row.get("n", base_row.get("n", 0))),
                "base_bias_sbp": float(base_row.get("bias_sbp", 0.0)),
                "guarded_bias_sbp": float(guarded_row.get("bias_sbp", 0.0)),
                "base_bias_dbp": float(base_row.get("bias_dbp", 0.0)),
                "guarded_bias_dbp": float(guarded_row.get("bias_dbp", 0.0)),
                "base_mae_sbp": float(base_row.get("mae_sbp", 0.0)),
                "guarded_mae_sbp": float(guarded_row.get("mae_sbp", 0.0)),
                "base_mae_dbp": float(base_row.get("mae_dbp", 0.0)),
                "guarded_mae_dbp": float(guarded_row.get("mae_dbp", 0.0)),
            }
        )
    return rows


def plot_clinical_guard_bias_comparison(
    rows: Sequence[dict],
    fig_dir: Path,
    filename: str = "clinical_guard_bias_comparison.png",
):
    if not rows:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["bp_range"]).title() for row in rows]
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.18
    base_sbp = np.asarray([float(row.get("base_bias_sbp", 0.0)) for row in rows], dtype=np.float32)
    guarded_sbp = np.asarray([float(row.get("guarded_bias_sbp", 0.0)) for row in rows], dtype=np.float32)
    base_dbp = np.asarray([float(row.get("base_bias_dbp", 0.0)) for row in rows], dtype=np.float32)
    guarded_dbp = np.asarray([float(row.get("guarded_bias_dbp", 0.0)) for row in rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.bar(x - 1.5 * width, base_sbp, width=width, color="#b03a2e", alpha=0.70, label="SBP bias / pre-guard")
    ax.bar(x - 0.5 * width, guarded_sbp, width=width, color="#e74c3c", alpha=0.88, label="SBP bias / guarded")
    ax.bar(x + 0.5 * width, base_dbp, width=width, color="#1f618d", alpha=0.70, label="DBP bias / pre-guard")
    ax.bar(x + 1.5 * width, guarded_dbp, width=width, color="#3498db", alpha=0.88, label="DBP bias / guarded")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bias (Pred - True, mmHg)")
    ax.set_title("Clinical Bias Comparison Before and After High-Risk Guard")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_uncertainty_decile_rows(reg_out: dict) -> List[dict]:
    uncertainty = np.asarray(reg_out["uncertainty"], dtype=np.float32).reshape(-1)
    abs_err = np.mean(
        np.abs(np.asarray(reg_out["y_true_reg"], dtype=np.float32) - np.asarray(reg_out["y_pred_reg"], dtype=np.float32)),
        axis=1,
    )
    quantiles = np.quantile(uncertainty, np.linspace(0.0, 1.0, 11))
    rows: List[dict] = []
    for idx in range(10):
        low = quantiles[idx]
        high = quantiles[idx + 1]
        if idx == 9:
            mask = (uncertainty >= low) & (uncertainty <= high)
        else:
            mask = (uncertainty >= low) & (uncertainty < high)
        if not np.any(mask):
            continue
        rows.append(
            {
                "decile": int(idx + 1),
                "uncertainty_low": float(low),
                "uncertainty_high": float(high),
                "n": int(mask.sum()),
                "uncertainty_mean": float(uncertainty[mask].mean()),
                "mae_mean": float(abs_err[mask].mean()),
                "mae_sbp": float(np.abs(np.asarray(reg_out["y_true_reg"])[mask, 0] - np.asarray(reg_out["y_pred_reg"])[mask, 0]).mean()),
                "mae_dbp": float(np.abs(np.asarray(reg_out["y_true_reg"])[mask, 1] - np.asarray(reg_out["y_pred_reg"])[mask, 1]).mean()),
            }
        )
    return rows


def build_classwise_regression_gain_rows(base_out: dict, meta_out: dict, cfg) -> List[dict]:
    y_true_cls = np.asarray(base_out["y_true_cls"], dtype=np.int64)
    y_true_reg = np.asarray(base_out["y_true_reg"], dtype=np.float32)
    base_pred = np.asarray(base_out["y_pred_reg"], dtype=np.float32)
    meta_pred = np.asarray(meta_out["y_pred_reg"], dtype=np.float32)
    rows: List[dict] = []
    for cls_idx, cls_name in enumerate(cfg.CLASS_NAMES):
        mask = y_true_cls == cls_idx
        if not np.any(mask):
            continue
        base_mae = np.abs(y_true_reg[mask] - base_pred[mask])
        meta_mae = np.abs(y_true_reg[mask] - meta_pred[mask])
        rows.append(
            {
                "class_name": cls_name,
                "n": int(mask.sum()),
                "base_mae_sbp": float(base_mae[:, 0].mean()),
                "base_mae_dbp": float(base_mae[:, 1].mean()),
                "meta_mae_sbp": float(meta_mae[:, 0].mean()),
                "meta_mae_dbp": float(meta_mae[:, 1].mean()),
                "gain_sbp": float(base_mae[:, 0].mean() - meta_mae[:, 0].mean()),
                "gain_dbp": float(base_mae[:, 1].mean() - meta_mae[:, 1].mean()),
                "gain_mean": float(base_mae.mean() - meta_mae.mean()),
            }
        )
    return rows


def _gain_rows_have_signal(rows: Sequence[dict], eps: float = 1.0e-6) -> bool:
    for row in rows:
        for key in ("gain_sbp", "gain_dbp", "gain_mean"):
            if abs(float(row.get(key, 0.0))) > eps:
                return True
    return False


def _pretty_variant_name(name: str) -> str:
    mapping = {
        "optlong_corrected": "Longitudinal reference",
        "optlong_from_reg": "Longitudinal reference",
        "dualmax_corrected": "Dual-anchor reference",
        "dualmax_hybrid": "Dual-anchor reference",
        "guided_head": "Guided classifier",
        "meta_selected": "Proposed fusion",
        "stability_selected": "Robust operating point",
        "selected_final": "Final operating point",
    }
    return mapping.get(str(name), str(name).replace("_", " ").title())


def build_classification_payload(reg_out: dict, cls_prob: np.ndarray) -> dict:
    return {
        "y_true_reg": np.asarray(reg_out["y_true_reg"], dtype=np.float32),
        "y_pred_reg": np.asarray(reg_out["y_pred_reg"], dtype=np.float32),
        "y_true_cls": np.asarray(reg_out["y_true_cls"], dtype=np.int64),
        "y_prob_cls": bridge_script.normalize_prob(np.asarray(cls_prob, dtype=np.float32)),
    }


def build_missing_row(
    missing_prob: float,
    ppg_reg: dict,
    ppg_cls_metrics: dict,
    ecg_reg: dict,
    ecg_cls_metrics: dict,
    ppg_prefix: str,
    ecg_prefix: str,
) -> dict:
    ppg_compat = stage_script.hybrid_metrics_for_compat(ppg_cls_metrics, ppg_prefix)
    ecg_compat = stage_script.hybrid_metrics_for_compat(ecg_cls_metrics, ecg_prefix)
    row = {"missing_prob": float(missing_prob)}
    for key, value in ppg_reg["metrics_reg"].items():
        row[f"ppg_missing_{key}"] = float(value)
    for key, value in ppg_compat.items():
        row[f"ppg_missing_{key}"] = float(value)
    for key, value in ecg_reg["metrics_reg"].items():
        row[f"ecg_missing_{key}"] = float(value)
    for key, value in ecg_compat.items():
        row[f"ecg_missing_{key}"] = float(value)
    return row


def plot_bootstrap_ci(rows: Sequence[dict], fig_dir: Path):
    if not rows:
        return
    metrics = ["mae_mean", "macro_f1"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, metrics):
        subset = [row for row in rows if str(row["metric"]) == metric]
        variants = [_pretty_variant_name(str(row["variant"])) for row in subset]
        means = np.asarray([float(row["mean"]) for row in subset], dtype=np.float32)
        lows = means - np.asarray([float(row["ci_low"]) for row in subset], dtype=np.float32)
        highs = np.asarray([float(row["ci_high"]) for row in subset], dtype=np.float32) - means
        y = np.arange(len(variants))
        ax.errorbar(means, y, xerr=np.vstack([lows, highs]), fmt="o", capsize=4)
        ax.set_yticks(y)
        ax.set_yticklabels(variants)
        if metric == "mae_mean":
            ax.set_title("Bootstrap Confidence Interval: Mean Absolute Error")
            ax.set_xlabel("Mean absolute error (mmHg)")
        else:
            ax.set_title("Bootstrap Confidence Interval: Macro-F1")
            ax.set_xlabel("Macro-F1")
    fig.tight_layout()
    fig.savefig(fig_dir / "bootstrap_primary_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_deciles(rows: Sequence[dict], fig_dir: Path):
    if not rows:
        return
    x = [int(row["decile"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [float(row["mae_sbp"]) for row in rows], marker="o", label="SBP MAE")
    ax.plot(x, [float(row["mae_dbp"]) for row in rows], marker="s", label="DBP MAE")
    ax.plot(x, [float(row["mae_mean"]) for row in rows], marker="^", label="Mean MAE")
    ax.set_xlabel("Uncertainty decile")
    ax.set_ylabel("MAE")
    ax.set_title("Uncertainty-Stratified Error Profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "uncertainty_decile_mae.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residual_histograms(base_out: dict, meta_out: dict, fig_dir: Path):
    base_err = np.asarray(base_out["y_pred_reg"], dtype=np.float32) - np.asarray(base_out["y_true_reg"], dtype=np.float32)
    meta_err = np.asarray(meta_out["y_pred_reg"], dtype=np.float32) - np.asarray(meta_out["y_true_reg"], dtype=np.float32)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(base_err[:, 0], bins=40, alpha=0.85, color="#4c72b0")
    axes[0, 0].set_title("Reference Residual Distribution: SBP")
    axes[0, 1].hist(meta_err[:, 0], bins=40, alpha=0.85, color="#55a868")
    axes[0, 1].set_title("Final Fusion Residual Distribution: SBP")
    axes[1, 0].hist(base_err[:, 1], bins=40, alpha=0.85, color="#c44e52")
    axes[1, 0].set_title("Reference Residual Distribution: DBP")
    axes[1, 1].hist(meta_err[:, 1], bins=40, alpha=0.85, color="#8172b2")
    axes[1, 1].set_title("Final Fusion Residual Distribution: DBP")
    for ax in axes.reshape(-1):
        ax.set_xlabel("Prediction - truth")
        ax.set_ylabel("Samples")
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_histograms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_classwise_gain_heatmap(
    rows: Sequence[dict],
    fig_dir: Path,
    filename: str = "classwise_gain_heatmap.png",
    title: str = "Class-Specific Absolute Error Reduction Matrix",
):
    out_path = fig_dir / filename
    if not rows or not _gain_rows_have_signal(rows):
        try:
            out_path.unlink(missing_ok=True)
        except PermissionError:
            pass
        return
    classes = [str(row["class_name"]) for row in rows]
    matrix = np.asarray(
        [[float(row["gain_sbp"]) for row in rows], [float(row["gain_dbp"]) for row in rows]],
        dtype=np.float32,
    )
    vmax = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if vmax < 1.0e-3:
        vmax = 0.10
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["SBP reduction", "DBP reduction"])
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MAE reduction (mmHg)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_subject_error_distribution(rows: Sequence[dict], fig_dir: Path, filename: str = "subject_error_distribution.png"):
    if not rows:
        return
    sbp = np.asarray([float(row["mae_sbp"]) for row in rows], dtype=np.float32)
    dbp = np.asarray([float(row["mae_dbp"]) for row in rows], dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([sbp, dbp], labels=["SBP", "DBP"], showfliers=False, patch_artist=True)
    axes[0].set_title("Subject-Level MAE Distribution")
    axes[0].set_ylabel("MAE (mmHg)")
    axes[1].hist(sbp, bins=24, alpha=0.70, label="SBP", color="#2874a6")
    axes[1].hist(dbp, bins=24, alpha=0.65, label="DBP", color="#ca6f1e")
    axes[1].set_title("Subject-Level Error Histogram")
    axes[1].set_xlabel("MAE (mmHg)")
    axes[1].set_ylabel("Subjects")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_boxplots(reg_out: dict, fig_dir: Path, filename: str = "uncertainty_decile_boxplot.png"):
    uncertainty = np.asarray(reg_out["uncertainty"], dtype=np.float32).reshape(-1)
    y_true = np.asarray(reg_out["y_true_reg"], dtype=np.float32)
    y_pred = np.asarray(reg_out["y_pred_reg"], dtype=np.float32)
    abs_err = np.mean(np.abs(y_true - y_pred), axis=1)
    quantiles = np.quantile(uncertainty, np.linspace(0.0, 1.0, 11))
    groups = []
    labels = []
    for idx in range(10):
        low = quantiles[idx]
        high = quantiles[idx + 1]
        if idx == 9:
            mask = (uncertainty >= low) & (uncertainty <= high)
        else:
            mask = (uncertainty >= low) & (uncertainty < high)
        if not np.any(mask):
            continue
        groups.append(abs_err[mask])
        labels.append(str(idx + 1))
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(groups, labels=labels, showfliers=False, patch_artist=True)
    ax.set_xlabel("Uncertainty decile")
    ax.set_ylabel("Per-sample mean absolute error (mmHg)")
    ax.set_title("Error Distribution Across Uncertainty Deciles")
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bp_range_heatmap(rows: Sequence[dict], fig_dir: Path, filename: str = "bp_range_error_heatmap.png"):
    if not rows:
        return
    targets = []
    for target in ("sbp", "dbp"):
        subset = [row for row in rows if str(row["target"]).lower() == target]
        if not subset:
            continue
        labels = [f"{int(float(row['bin_low']))}-{int(float(row['bin_high']))}" for row in subset]
        matrix = np.asarray(
            [
                [float(row["mae"]) for row in subset],
                [float(row["bias"]) for row in subset],
                [float(row["p90_abs_error"]) for row in subset],
            ],
            dtype=np.float32,
        )
        targets.append((target.upper(), labels, matrix))
    if not targets:
        return
    fig, axes = plt.subplots(1, len(targets), figsize=(6.2 * len(targets), 4.8))
    if len(targets) == 1:
        axes = [axes]
    for ax, (name, labels, matrix) in zip(axes, targets):
        im = ax.imshow(matrix, cmap="magma")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["MAE", "Bias", "P90 |error|"])
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(f"{name}: Range-Stratified Error Landscape")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mmHg")
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_bp_bin_rows(y_true_reg: np.ndarray, y_pred_reg: np.ndarray) -> List[dict]:
    y_true_reg = np.asarray(y_true_reg, dtype=np.float32)
    y_pred_reg = np.asarray(y_pred_reg, dtype=np.float32)
    sbp_bins = [70, 90, 110, 130, 150, 170, 210]
    dbp_bins = [35, 50, 65, 80, 95, 110, 140]
    rows: List[dict] = []
    for target_idx, (target_name, bins) in enumerate((("sbp", sbp_bins), ("dbp", dbp_bins))):
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (y_true_reg[:, target_idx] >= low) & (y_true_reg[:, target_idx] < high)
            if not np.any(mask):
                continue
            err = np.abs(y_pred_reg[mask, target_idx] - y_true_reg[mask, target_idx])
            rows.append(
                {
                    "target": target_name,
                    "bin_low": float(low),
                    "bin_high": float(high),
                    "n": int(mask.sum()),
                    "mae": float(err.mean()),
                    "bias": float((y_pred_reg[mask, target_idx] - y_true_reg[mask, target_idx]).mean()),
                    "p90_abs_error": float(np.quantile(err, 0.90)),
                }
            )
    return rows


def plot_variant_class_heatmap(rows: Sequence[dict], fig_dir: Path, filename: str = "variant_class_performance_heatmap.png"):
    if not rows:
        return
    variant_order = []
    for name in ("selected_final", "stability_selected", "guided_head", "dualmax_hybrid", "optlong_from_reg"):
        if any(str(row["variant"]) == name for row in rows):
            variant_order.append(name)
    class_order = [str(row["class_name"]) for row in rows if str(row["variant"]) == variant_order[0]] if variant_order else []
    if not variant_order or not class_order:
        return
    f1_matrix = np.zeros((len(variant_order), len(class_order)), dtype=np.float32)
    pr_matrix = np.zeros((len(variant_order), len(class_order)), dtype=np.float32)
    for i, variant in enumerate(variant_order):
        for j, cls_name in enumerate(class_order):
            match = next((row for row in rows if str(row["variant"]) == variant and str(row["class_name"]) == cls_name), None)
            if match is None:
                continue
            f1_matrix[i, j] = float(match["f1"])
            pr_matrix[i, j] = float(match["auprc"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, matrix, title in (
        (axes[0], f1_matrix, "Classwise F1 Profile Across Decision Systems"),
        (axes[1], pr_matrix, "Classwise PR-AUC Profile Across Decision Systems"),
    ):
        im = ax.imshow(matrix, cmap="YlGnBu", vmin=float(matrix.min()), vmax=float(matrix.max().clip(min=1.0e-6)))
        ax.set_xticks(np.arange(len(class_order)))
        ax.set_xticklabels(class_order, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(variant_order)))
        ax.set_yticklabels([_pretty_variant_name(name) for name in variant_order])
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_regression_expert_class_rows(reg_out: dict, cfg) -> List[dict]:
    if "router_weights" not in reg_out:
        return []
    weights = np.asarray(reg_out["router_weights"], dtype=np.float32)
    y_true_cls = np.asarray(reg_out["y_true_cls"], dtype=np.int64)
    expert_names = list(reg_out.get("router_expert_names", regression_router_expert_names()))
    rows: List[dict] = []
    for target_idx, target_name in enumerate(("sbp", "dbp")):
        for expert_idx, expert_name in enumerate(expert_names):
            for cls_idx, cls_name in enumerate(cfg.CLASS_NAMES):
                mask = y_true_cls == cls_idx
                if not np.any(mask):
                    continue
                rows.append(
                    {
                        "target": target_name,
                        "expert": expert_name,
                        "class_name": cls_name,
                        "mean_weight": float(weights[mask, expert_idx, target_idx].mean()),
                        "n": int(mask.sum()),
                    }
                )
    return rows


def plot_regression_expert_class_heatmap(rows: Sequence[dict], fig_dir: Path, cfg, filename: str = "regression_expert_usage_by_class_heatmap.png"):
    if not rows:
        return
    expert_names = regression_router_expert_names()
    class_names = list(cfg.CLASS_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target_name in zip(axes, ("sbp", "dbp")):
        matrix = np.zeros((len(expert_names), len(class_names)), dtype=np.float32)
        for i, expert_name in enumerate(expert_names):
            for j, cls_name in enumerate(class_names):
                match = next(
                    (
                        row
                        for row in rows
                        if str(row["target"]) == target_name
                        and str(row["expert"]) == expert_name
                        and str(row["class_name"]) == cls_name
                    ),
                    None,
                )
                if match is not None:
                    matrix[i, j] = float(match["mean_weight"])
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(float(matrix.max()), 1.0e-6))
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(expert_names)))
        ax.set_yticklabels(expert_names)
        ax.set_title(f"{target_name.upper()}: Expert Usage by BP Class")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean routing weight")
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_regression_expert_uncertainty_rows(reg_out: dict) -> List[dict]:
    if "router_weights" not in reg_out:
        return []
    weights = np.asarray(reg_out["router_weights"], dtype=np.float32)
    uncertainty = np.asarray(reg_out["uncertainty"], dtype=np.float32).reshape(-1)
    expert_names = list(reg_out.get("router_expert_names", regression_router_expert_names()))
    quantiles = np.quantile(uncertainty, np.linspace(0.0, 1.0, 11))
    rows: List[dict] = []
    for idx in range(10):
        low = quantiles[idx]
        high = quantiles[idx + 1]
        if idx == 9:
            mask = (uncertainty >= low) & (uncertainty <= high)
        else:
            mask = (uncertainty >= low) & (uncertainty < high)
        if not np.any(mask):
            continue
        for target_idx, target_name in enumerate(("sbp", "dbp")):
            for expert_idx, expert_name in enumerate(expert_names):
                rows.append(
                    {
                        "decile": int(idx + 1),
                        "target": target_name,
                        "expert": expert_name,
                        "mean_weight": float(weights[mask, expert_idx, target_idx].mean()),
                        "n": int(mask.sum()),
                    }
                )
    return rows


def plot_regression_expert_uncertainty(rows: Sequence[dict], fig_dir: Path, filename: str = "regression_expert_usage_by_uncertainty.png"):
    if not rows:
        return
    expert_names = regression_router_expert_names()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, target_name in zip(axes, ("sbp", "dbp")):
        for expert_name in expert_names:
            subset = [
                row for row in rows
                if str(row["target"]) == target_name and str(row["expert"]) == expert_name
            ]
            subset.sort(key=lambda row: int(row["decile"]))
            if not subset:
                continue
            ax.plot(
                [int(row["decile"]) for row in subset],
                [float(row["mean_weight"]) for row in subset],
                marker="o",
                label=expert_name,
            )
        ax.set_xlabel("Uncertainty decile")
        ax.set_ylabel("Mean routing weight")
        ax.set_title(f"{target_name.upper()}: Expert Preference Across Uncertainty")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.20, linestyle="--")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_classification_arbiter_rows(
    weight_selected: np.ndarray,
    blended_prob: np.ndarray,
    y_true_cls: np.ndarray,
    cfg,
) -> tuple[List[dict], List[dict]]:
    weight_selected = np.asarray(weight_selected, dtype=np.float32).reshape(-1)
    blended_prob = bridge_script.normalize_prob(np.asarray(blended_prob, dtype=np.float32))
    y_true_cls = np.asarray(y_true_cls, dtype=np.int64)

    class_rows: List[dict] = []
    for cls_idx, cls_name in enumerate(cfg.CLASS_NAMES):
        mask = y_true_cls == cls_idx
        if not np.any(mask):
            continue
        class_rows.append(
            {
                "class_name": cls_name,
                "n": int(mask.sum()),
                "mean_weight_selected": float(weight_selected[mask].mean()),
            }
        )

    conf = np.max(blended_prob, axis=1)
    quantiles = np.quantile(conf, np.linspace(0.0, 1.0, 11))
    conf_rows: List[dict] = []
    for idx in range(10):
        low = quantiles[idx]
        high = quantiles[idx + 1]
        if idx == 9:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf >= low) & (conf < high)
        if not np.any(mask):
            continue
        conf_rows.append(
            {
                "decile": int(idx + 1),
                "n": int(mask.sum()),
                "confidence_mean": float(conf[mask].mean()),
                "mean_weight_selected": float(weight_selected[mask].mean()),
            }
        )
    return class_rows, conf_rows


def plot_classification_arbiter_profile(
    class_rows: Sequence[dict],
    conf_rows: Sequence[dict],
    fig_dir: Path,
    filename: str = "classification_arbiter_profile.png",
):
    if not class_rows and not conf_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    if class_rows:
        axes[0].bar(
            [str(row["class_name"]) for row in class_rows],
            [float(row["mean_weight_selected"]) for row in class_rows],
            color="#1f618d",
        )
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_ylabel("Mean selected-branch weight")
        axes[0].set_title("Classification Arbiter by BP Class")
    else:
        axes[0].axis("off")
    if conf_rows:
        axes[1].plot(
            [int(row["decile"]) for row in conf_rows],
            [float(row["mean_weight_selected"]) for row in conf_rows],
            marker="o",
            color="#b03a2e",
        )
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_xlabel("Confidence decile")
        axes[1].set_ylabel("Mean selected-branch weight")
        axes[1].set_title("Classification Arbiter by Confidence")
        axes[1].grid(alpha=0.20, linestyle="--")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_classification_search_frontier(
    blend_rows: Sequence[dict],
    arbiter_rows: Sequence[dict],
    fig_dir: Path,
    filename: str = "classification_search_frontier.png",
):
    if not blend_rows and not arbiter_rows:
        return
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    plotted = False
    for label, rows, color in (
        ("Blend policy", blend_rows, "#1f77b4"),
        ("Arbiter policy", arbiter_rows, "#d62728"),
    ):
        if not rows:
            continue
        acc = np.asarray([float(row.get("acc", 0.0)) for row in rows], dtype=np.float32)
        macro_f1 = np.asarray([float(row.get("macro_f1", 0.0)) for row in rows], dtype=np.float32)
        score = np.asarray([float(row.get("score", 0.0)) for row in rows], dtype=np.float32)
        if acc.size == 0:
            continue
        score_span = float(max(1.0e-6, score.max() - score.min()))
        sizes = 36.0 + 160.0 * (score - float(score.min())) / score_span
        ax.scatter(acc, macro_f1, s=sizes, alpha=0.55, color=color, edgecolor="white", linewidth=0.6, label=label)
        best_idx = int(np.argmax(score))
        ax.scatter(acc[best_idx], macro_f1[best_idx], s=float(sizes[best_idx]) + 70.0, color=color, edgecolor="black", linewidth=1.0)
        ax.annotate(
            label,
            (float(acc[best_idx]), float(macro_f1[best_idx])),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=10,
            weight="bold",
            color=color,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Classification Search Frontier")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def safety_class_fusion_score(metrics: dict, prefix: str, cfg) -> float:
    stage1_name = str(cfg.CLASS_NAMES[2])
    stage2_name = str(cfg.CLASS_NAMES[3])
    return float(
        guided_script.classification_candidate_score(metrics, prefix)
        + float(getattr(cfg, "SAFETY_CLASS_FUSION_STAGE1_RECALL_WEIGHT", 0.0))
        * float(metrics.get(f"cls_recall_{prefix}_{stage1_name}", 0.0))
        + float(getattr(cfg, "SAFETY_CLASS_FUSION_STAGE2_RECALL_WEIGHT", 0.0))
        * float(metrics.get(f"cls_recall_{prefix}_{stage2_name}", 0.0))
        + float(getattr(cfg, "SAFETY_CLASS_FUSION_STAGE2_F1_WEIGHT", 0.0))
        * float(metrics.get(f"cls_f1_{prefix}_{stage2_name}", 0.0))
    )


def apply_safety_class_fusion_prob(
    row: dict,
    base_prob: np.ndarray,
    reg_out: dict,
    cfg,
) -> dict:
    base_prob = bridge_script.normalize_prob(np.asarray(base_prob, dtype=np.float32))
    reg_prob = shared_plots.regression_to_class_prob(
        np.asarray(reg_out["y_pred_reg"], dtype=np.float32),
        reg_out.get("uncertainty"),
        cfg,
    )
    reg_prob = bridge_script.normalize_prob(np.asarray(reg_prob, dtype=np.float32))
    disagreement = 0.5 * np.abs(reg_prob - base_prob).sum(axis=1)
    high_signal, crisis_signal = risk_guard_signals(reg_out, base_prob)
    base_pred = base_prob.argmax(axis=1).astype(np.int64)
    reg_pred = reg_prob.argmax(axis=1).astype(np.int64)
    risk_upshift = np.clip(
        (reg_pred - base_pred).astype(np.float32) / max(1, int(cfg.N_CLASSES) - 1),
        0.0,
        1.0,
    )

    candidate = str(row["candidate"])
    if candidate == "identity" or float(row.get("fusion_scale", 0.0)) <= 0.0:
        weight = np.zeros(len(base_prob), dtype=np.float32)
        fused_prob = base_prob
    else:
        disagree_flag = (base_pred != reg_pred).astype(np.float32)
        weight = float(row["fusion_scale"]) * np.power(
            np.clip(disagreement, 1.0e-5, 1.0),
            float(row["fusion_beta"]),
        )
        weight *= 1.0 + (float(row["fusion_disagree_gain"]) - 1.0) * np.clip(
            disagree_flag + 0.75 * risk_upshift,
            0.0,
            1.5,
        )
        weight *= (
            1.0
            + (float(row["fusion_high_gain"]) - 1.0) * high_signal
            + (float(row["fusion_crisis_gain"]) - 1.0) * crisis_signal
        )
        weight = np.clip(
            weight.astype(np.float32),
            0.0,
            float(getattr(cfg, "SAFETY_CLASS_FUSION_MAX_WEIGHT", 0.60)),
        )
        fused_prob = bridge_script.normalize_prob(
            (1.0 - weight.reshape(-1, 1)) * base_prob
            + weight.reshape(-1, 1) * reg_prob
        )

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
    stage1_name = str(cfg.CLASS_NAMES[2])
    stage2_name = str(cfg.CLASS_NAMES[3])

    identity_diag = apply_safety_class_fusion_prob(
        {"candidate": "identity", "fusion_scale": 0.0},
        query_cls_prob,
        query_reg_out,
        cfg,
    )
    identity_prob = identity_diag["prob"]
    identity_metrics = stage_script.risk_classification_metrics(
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
            "fusion_disagree_gain": 1.0,
            "fusion_high_gain": 1.0,
            "fusion_crisis_gain": 1.0,
            "score": float(safety_class_fusion_score(identity_metrics, "selected_val", cfg)),
            "mean_weight": 0.0,
            "p90_weight": 0.0,
            "mean_disagreement": float(identity_diag["disagreement"].mean()),
            "stage1_recall": float(identity_metrics.get(f"cls_recall_selected_val_{stage1_name}", 0.0)),
            "stage2_recall": float(identity_metrics.get(f"cls_recall_selected_val_{stage2_name}", 0.0)),
            "stage2_f1": float(identity_metrics.get(f"cls_f1_selected_val_{stage2_name}", 0.0)),
            **guided_script.class_summary(identity_metrics, "selected_val"),
        }
    )

    for fusion_scale in tuple(float(x) for x in cfg.SAFETY_CLASS_FUSION_SCALES):
        if fusion_scale <= 0.0:
            continue
        for fusion_beta in tuple(float(x) for x in cfg.SAFETY_CLASS_FUSION_BETAS):
            for fusion_disagree_gain in tuple(float(x) for x in cfg.SAFETY_CLASS_FUSION_DISAGREE_GAINS):
                for fusion_high_gain in tuple(float(x) for x in cfg.SAFETY_CLASS_FUSION_HIGH_GAINS):
                    for fusion_crisis_gain in tuple(float(x) for x in cfg.SAFETY_CLASS_FUSION_CRISIS_GAINS):
                        row = {
                            "candidate": (
                                f"safety_cls_s{int(round(100.0 * fusion_scale))}"
                                f"_b{str(fusion_beta).replace('.', 'p')}"
                                f"_d{str(fusion_disagree_gain).replace('.', 'p')}"
                                f"_h{str(fusion_high_gain).replace('.', 'p')}"
                                f"_c{str(fusion_crisis_gain).replace('.', 'p')}"
                            ),
                            "fusion_scale": float(fusion_scale),
                            "fusion_beta": float(fusion_beta),
                            "fusion_disagree_gain": float(fusion_disagree_gain),
                            "fusion_high_gain": float(fusion_high_gain),
                            "fusion_crisis_gain": float(fusion_crisis_gain),
                        }
                        diag = apply_safety_class_fusion_prob(row, query_cls_prob, query_reg_out, cfg)
                        fused_prob = diag["prob"]
                        metrics = stage_script.risk_classification_metrics(
                            y_true,
                            fused_prob.argmax(axis=1).astype(np.int64),
                            fused_prob,
                            cfg,
                            prefix="selected_val",
                        )
                        rows.append(
                            {
                                **row,
                                "score": float(safety_class_fusion_score(metrics, "selected_val", cfg)),
                                "mean_weight": float(diag["weight"].mean()),
                                "p90_weight": float(np.quantile(diag["weight"], 0.90)),
                                "mean_disagreement": float(diag["disagreement"].mean()),
                                "stage1_recall": float(metrics.get(f"cls_recall_selected_val_{stage1_name}", 0.0)),
                                "stage2_recall": float(metrics.get(f"cls_recall_selected_val_{stage2_name}", 0.0)),
                                "stage2_f1": float(metrics.get(f"cls_f1_selected_val_{stage2_name}", 0.0)),
                                **guided_script.class_summary(metrics, "selected_val"),
                            }
                        )

    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return rows[0], rows


def build_safety_class_transition_rows(
    pre_prob: np.ndarray,
    post_prob: np.ndarray,
    weight: np.ndarray,
    cfg,
) -> List[dict]:
    pre_pred = np.asarray(pre_prob, dtype=np.float32).argmax(axis=1).astype(np.int64)
    post_pred = np.asarray(post_prob, dtype=np.float32).argmax(axis=1).astype(np.int64)
    weight = np.asarray(weight, dtype=np.float32).reshape(-1)
    rows: List[dict] = []
    for pre_idx, pre_name in enumerate(cfg.CLASS_NAMES):
        pre_mask = pre_pred == pre_idx
        pre_count = int(pre_mask.sum())
        for post_idx, post_name in enumerate(cfg.CLASS_NAMES):
            mask = pre_mask & (post_pred == post_idx)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "pre_class": str(pre_name),
                    "post_class": str(post_name),
                    "n": int(mask.sum()),
                    "fraction": float(mask.mean()),
                    "row_fraction": float(mask.sum() / max(1, pre_count)),
                    "mean_weight": float(weight[mask].mean()),
                }
            )
    return rows


def build_safety_class_profile_rows(
    diag: dict,
    pre_prob: np.ndarray,
    post_prob: np.ndarray,
    y_true_cls: np.ndarray,
    cfg,
) -> tuple[List[dict], List[dict]]:
    weight = np.asarray(diag["weight"], dtype=np.float32).reshape(-1)
    disagreement = np.asarray(diag["disagreement"], dtype=np.float32).reshape(-1)
    high_signal = np.asarray(diag["high_signal"], dtype=np.float32).reshape(-1)
    crisis_signal = np.asarray(diag["crisis_signal"], dtype=np.float32).reshape(-1)
    y_true_cls = np.asarray(y_true_cls, dtype=np.int64)
    pre_pred = np.asarray(pre_prob, dtype=np.float32).argmax(axis=1).astype(np.int64)
    post_pred = np.asarray(post_prob, dtype=np.float32).argmax(axis=1).astype(np.int64)

    class_rows: List[dict] = []
    for cls_idx, cls_name in enumerate(cfg.CLASS_NAMES):
        mask = y_true_cls == cls_idx
        if not np.any(mask):
            continue
        class_rows.append(
            {
                "class_name": str(cls_name),
                "n": int(mask.sum()),
                "mean_fusion_weight": float(weight[mask].mean()),
                "mean_disagreement": float(disagreement[mask].mean()),
                "mean_high_signal": float(high_signal[mask].mean()),
                "mean_crisis_signal": float(crisis_signal[mask].mean()),
                "pre_acc": float(np.mean(pre_pred[mask] == y_true_cls[mask])),
                "post_acc": float(np.mean(post_pred[mask] == y_true_cls[mask])),
            }
        )

    decile_rows: List[dict] = []
    if len(disagreement):
        quantiles = np.quantile(disagreement, np.linspace(0.0, 1.0, 11))
        for idx in range(10):
            low = quantiles[idx]
            high = quantiles[idx + 1]
            if idx == 9:
                mask = (disagreement >= low) & (disagreement <= high)
            else:
                mask = (disagreement >= low) & (disagreement < high)
            if not np.any(mask):
                continue
            decile_rows.append(
                {
                    "decile": int(idx + 1),
                    "n": int(mask.sum()),
                    "mean_disagreement": float(disagreement[mask].mean()),
                    "mean_fusion_weight": float(weight[mask].mean()),
                    "mean_high_signal": float(high_signal[mask].mean()),
                    "mean_crisis_signal": float(crisis_signal[mask].mean()),
                    "pre_acc": float(np.mean(pre_pred[mask] == y_true_cls[mask])),
                    "post_acc": float(np.mean(post_pred[mask] == y_true_cls[mask])),
                }
            )
    return class_rows, decile_rows


def plot_safety_class_fusion_frontier(
    rows: Sequence[dict],
    fig_dir: Path,
    filename: str = "safety_class_fusion_frontier.png",
):
    if not rows:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    acc = np.asarray([float(row.get("acc", 0.0)) for row in rows], dtype=np.float32)
    macro_f1 = np.asarray([float(row.get("macro_f1", 0.0)) for row in rows], dtype=np.float32)
    score = np.asarray([float(row.get("score", 0.0)) for row in rows], dtype=np.float32)
    stage2_recall = np.asarray([float(row.get("stage2_recall", 0.0)) for row in rows], dtype=np.float32)
    score_span = float(max(1.0e-6, score.max() - score.min()))
    sizes = 36.0 + 160.0 * (score - float(score.min())) / score_span
    scatter = ax.scatter(
        acc,
        macro_f1,
        c=stage2_recall,
        s=sizes,
        cmap="viridis",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
    )
    best_idx = int(np.argmax(score))
    ax.scatter(
        acc[best_idx],
        macro_f1[best_idx],
        s=float(sizes[best_idx]) + 70.0,
        color="#d62728",
        edgecolor="black",
        linewidth=1.0,
    )
    ax.annotate(
        "Best safety fusion",
        (float(acc[best_idx]), float(macro_f1[best_idx])),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=10,
        weight="bold",
        color="#d62728",
    )
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Safety-Aware Classification Fusion Frontier")
    ax.grid(True, linestyle="--", alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Stage2 recall")
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_safety_class_transition_heatmap(
    rows: Sequence[dict],
    fig_dir: Path,
    cfg,
    filename: str = "safety_class_transition_heatmap.png",
):
    if not rows:
        return
    labels = [str(name) for name in cfg.CLASS_NAMES]
    index = {name: idx for idx, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float32)
    for row in rows:
        matrix[index[str(row["pre_class"])], index[str(row["post_class"])]] = float(row["row_fraction"])
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=max(1.0e-6, float(matrix.max())))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("After safety fusion")
    ax.set_ylabel("Before safety fusion")
    ax.set_title("Safety Fusion Prediction Transition Matrix")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row_idx, col_idx] < 0.60 * float(matrix.max()) else "black",
                fontsize=9,
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_safety_class_fusion_profile(
    class_rows: Sequence[dict],
    decile_rows: Sequence[dict],
    fig_dir: Path,
    filename: str = "safety_class_fusion_profile.png",
):
    if not class_rows and not decile_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    if class_rows:
        axes[0].bar(
            [str(row["class_name"]) for row in class_rows],
            [float(row["mean_fusion_weight"]) for row in class_rows],
            color="#2874a6",
        )
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_ylabel("Mean regression-branch weight")
        axes[0].set_title("Safety Fusion by True Class")
    else:
        axes[0].axis("off")
    if decile_rows:
        axes[1].plot(
            [int(row["decile"]) for row in decile_rows],
            [float(row["mean_fusion_weight"]) for row in decile_rows],
            marker="o",
            color="#1f77b4",
            label="Fusion weight",
        )
        axes[1].plot(
            [int(row["decile"]) for row in decile_rows],
            [float(row["pre_acc"]) for row in decile_rows],
            marker="s",
            color="#e67e22",
            label="Pre-fusion acc",
        )
        axes[1].plot(
            [int(row["decile"]) for row in decile_rows],
            [float(row["post_acc"]) for row in decile_rows],
            marker="^",
            color="#239b56",
            label="Post-fusion acc",
        )
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_xlabel("Classifier-regression disagreement decile")
        axes[1].set_ylabel("Value")
        axes[1].set_title("Safety Fusion Across Disagreement")
        axes[1].grid(alpha=0.20, linestyle="--")
        axes[1].legend(frameon=True)
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = build_meta_stack_cfg()
    stage_script.seed_everything(cfg.SEED)
    patch_stage_script_for_windows_runtime()

    opt_ckpt = bridge_script.pick_optlong_best_checkpoint(cfg)
    dual_ckpt = bridge_script.pick_dualmax_best_checkpoint(cfg)

    out_root, fig_dir, art_dir, tbl_dir = stage_script.ensure_out_dirs(cfg)
    loaders = stage_script.build_protocol_loaders(cfg, task="regression")
    stage_script.save_json(out_root / "protocol_manifest.json", loaders.manifest)

    print(f"Using device: {cfg.DEVICE}")
    print(f"Protocol rank: {cfg.PROTOCOL_STRICTNESS_RANK}")
    print(f"Split protocol: {cfg.SPLIT_PROTOCOL}")
    print(f"Protocol name: {cfg.PROTOCOL_NAME}")
    print(f"Opt-long best checkpoint: {opt_ckpt}")
    print(f"Dualmax best checkpoint: {dual_ckpt}")
    print(f"Resume feature head: {cfg.HEAD_RESUME_PATH}")

    print("Collecting anchor bundles...")
    opt_bundle = bridge_script.collect_bundle(opt_ckpt, loaders, cfg, cfg.SEED, "optlong")
    dual_bundle = bridge_script.collect_bundle(dual_ckpt, loaders, cfg, cfg.SEED + 41, "dualmax")

    print("Extracting meta-stack feature banks...")
    train_clean_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.ds_train, cfg, "train_clean")
    train_noise_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.ds_train, cfg, "train_noise", noise_std=float(cfg.HEAD_NOISE_STD))
    train_ecg_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.ds_train, cfg, "train_missing_ecg", drop_modality="ecg", missing_prob=float(cfg.HEAD_MISSING_ECG))
    train_ppg_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.ds_train, cfg, "train_missing_ppg", drop_modality="ppg", missing_prob=float(cfg.HEAD_MISSING_PPG))
    val_clean_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.val_query_loader, cfg, "val_clean")
    val_noise_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.val_query_loader, cfg, "val_noise", noise_std=float(cfg.HEAD_NOISE_STD))
    val_ecg_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.val_query_loader, cfg, "val_missing_ecg", drop_modality="ecg", missing_prob=float(cfg.HEAD_MISSING_ECG))
    val_ppg_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.val_query_loader, cfg, "val_missing_ppg", drop_modality="ppg", missing_prob=float(cfg.HEAD_MISSING_PPG))
    val_calib_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.val_calib_loader, cfg, "val_calib_clean")
    test_clean_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.test_query_loader, cfg, "test_clean")
    test_noise_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.test_query_loader, cfg, "test_noise", noise_std=float(cfg.HEAD_NOISE_STD))
    test_ecg_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.test_query_loader, cfg, "test_missing_ecg", drop_modality="ecg", missing_prob=float(cfg.HEAD_MISSING_ECG))
    test_ppg_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.test_query_loader, cfg, "test_missing_ppg", drop_modality="ppg", missing_prob=float(cfg.HEAD_MISSING_PPG))
    test_calib_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.test_calib_loader, cfg, "test_calib_clean")

    print("Resuming dual-anchor feature head...")
    head_model, head_state, best_head_metrics, head_epoch_rows = prev_script.run_feature_head_resume(
        Path(cfg.HEAD_RESUME_PATH),
        [train_clean_bank, train_noise_bank, train_ecg_bank, train_ppg_bank],
        val_clean_bank,
        val_noise_bank,
        val_ecg_bank,
        val_ppg_bank,
        cfg,
    )
    torch.save(
        {
            "model_state": head_model.state_dict(),
            "state": {
                "x_mu": head_state["x_mu"],
                "x_sigma": head_state["x_sigma"],
                "reg_mu": head_state["reg_mu"],
                "reg_sigma": head_state["reg_sigma"],
            },
        },
        out_root / "feature_head_best.pt",
    )

    temperature, temperature_rows = guided_script.search_temperature(head_model, guided_script.normalize_bank(val_calib_bank, head_state), cfg)
    base_calib_prob, _, _, _ = guided_script.evaluate_head(
        head_model,
        guided_script.normalize_bank(val_calib_bank, head_state),
        head_state,
        cfg,
        prefix="guided_calib_base",
        temperature=temperature,
    )
    best_policy_row, policy_rows = guided_script.search_policy(base_calib_prob, val_calib_bank["y"].numpy(), cfg)

    guided_val = predict_guided_bundle(head_model, head_state, val_clean_bank, cfg, "guided_val", temperature, best_policy_row)
    guided_val_noise = predict_guided_bundle(head_model, head_state, val_noise_bank, cfg, "guided_val_noise", temperature, best_policy_row)
    guided_val_ecg = predict_guided_bundle(head_model, head_state, val_ecg_bank, cfg, "guided_val_missing_ecg", temperature, best_policy_row)
    guided_val_ppg = predict_guided_bundle(head_model, head_state, val_ppg_bank, cfg, "guided_val_missing_ppg", temperature, best_policy_row)
    guided_val_calib = predict_guided_bundle(head_model, head_state, val_calib_bank, cfg, "guided_val_calib", temperature, best_policy_row)
    guided_test = predict_guided_bundle(head_model, head_state, test_clean_bank, cfg, "guided_test", temperature, best_policy_row)
    guided_test_noise = predict_guided_bundle(head_model, head_state, test_noise_bank, cfg, "guided_test_noise", temperature, best_policy_row)
    guided_test_ecg = predict_guided_bundle(head_model, head_state, test_ecg_bank, cfg, "guided_test_missing_ecg", temperature, best_policy_row)
    guided_test_ppg = predict_guided_bundle(head_model, head_state, test_ppg_bank, cfg, "guided_test_missing_ppg", temperature, best_policy_row)
    guided_test_calib = predict_guided_bundle(head_model, head_state, test_calib_bank, cfg, "guided_test_calib", temperature, best_policy_row)

    print("Preparing validation robustness sources...")
    val_noise_opt_raw = bridge_script.collect_outputs_bridge(opt_ckpt, loaders.val_query_loader, cfg, "opt_val_noise", noise_std=float(cfg.HEAD_NOISE_STD))
    val_noise_dual_raw = bridge_script.collect_outputs_bridge(dual_ckpt, loaders.val_query_loader, cfg, "dual_val_noise", noise_std=float(cfg.HEAD_NOISE_STD))
    val_noise_opt_eval = bridge_script.apply_stack_with_bundle(val_noise_opt_raw, opt_bundle["val_bundle"], cfg, prefix="opt_val_noise")
    val_noise_dual_eval = bridge_script.apply_stack_with_bundle(val_noise_dual_raw, dual_bundle["val_bundle"], cfg, prefix="dual_val_noise")

    val_ecg_opt_raw = bridge_script.collect_outputs_bridge(opt_ckpt, loaders.val_query_loader, cfg, "opt_val_missing_ecg", drop_modality="ecg", missing_prob=float(cfg.HEAD_MISSING_ECG))
    val_ecg_dual_raw = bridge_script.collect_outputs_bridge(dual_ckpt, loaders.val_query_loader, cfg, "dual_val_missing_ecg", drop_modality="ecg", missing_prob=float(cfg.HEAD_MISSING_ECG))
    val_ecg_opt_eval = bridge_script.apply_stack_with_bundle(val_ecg_opt_raw, opt_bundle["val_bundle"], cfg, prefix="opt_val_missing_ecg")
    val_ecg_dual_eval = bridge_script.apply_stack_with_bundle(val_ecg_dual_raw, dual_bundle["val_bundle"], cfg, prefix="dual_val_missing_ecg")

    val_ppg_opt_raw = bridge_script.collect_outputs_bridge(opt_ckpt, loaders.val_query_loader, cfg, "opt_val_missing_ppg", drop_modality="ppg", missing_prob=float(cfg.HEAD_MISSING_PPG))
    val_ppg_dual_raw = bridge_script.collect_outputs_bridge(dual_ckpt, loaders.val_query_loader, cfg, "dual_val_missing_ppg", drop_modality="ppg", missing_prob=float(cfg.HEAD_MISSING_PPG))
    val_ppg_opt_eval = bridge_script.apply_stack_with_bundle(val_ppg_opt_raw, opt_bundle["val_bundle"], cfg, prefix="opt_val_missing_ppg")
    val_ppg_dual_eval = bridge_script.apply_stack_with_bundle(val_ppg_dual_raw, dual_bundle["val_bundle"], cfg, prefix="dual_val_missing_ppg")

    dual_val_calib_eval = bridge_script.apply_stack_with_bundle(dual_bundle["val_raw_calib"], dual_bundle["val_bundle"], cfg, prefix="dual_val_calib")
    opt_val_calib_eval = bridge_script.apply_stack_with_bundle(opt_bundle["val_raw_calib"], opt_bundle["val_bundle"], cfg, prefix="opt_val_calib")

    clean_prob_sources = base_script.build_prob_sources(opt_bundle["val_raw_query"]["y_prob_cls_from_reg"], dual_bundle["val_raw_query"]["y_prob_cls_from_reg"], dual_bundle["val_bundle"]["hybrid_prob"], guided_val["prob"])
    noise_prob_sources = base_script.build_prob_sources(val_noise_opt_raw["y_prob_cls_from_reg"], val_noise_dual_raw["y_prob_cls_from_reg"], val_noise_dual_eval["hybrid_prob"], guided_val_noise["prob"])
    ecg_prob_sources = base_script.build_prob_sources(val_ecg_opt_raw["y_prob_cls_from_reg"], val_ecg_dual_raw["y_prob_cls_from_reg"], val_ecg_dual_eval["hybrid_prob"], guided_val_ecg["prob"])
    ppg_prob_sources = base_script.build_prob_sources(val_ppg_opt_raw["y_prob_cls_from_reg"], val_ppg_dual_raw["y_prob_cls_from_reg"], val_ppg_dual_eval["hybrid_prob"], guided_val_ppg["prob"])
    val_calib_prob_sources = base_script.build_prob_sources(opt_bundle["val_raw_calib"]["y_prob_cls_from_reg"], dual_bundle["val_raw_calib"]["y_prob_cls_from_reg"], dual_val_calib_eval["hybrid_prob"], guided_val_calib["prob"])

    print("Preparing current stability baseline...")
    gate_models = base_script.fit_regression_gate_models(opt_bundle["val_bundle"]["calib_corr"], dual_bundle["val_bundle"]["calib_corr"], cfg, seed=cfg.SEED + 701)
    regression_candidates = base_script.search_regression_candidates(gate_models, opt_bundle["val_bundle"], dual_bundle["val_bundle"], cfg)
    stability_cls_candidates, single_candidate_lookup = stability_script.search_classification_candidates_stability(
        clean_prob_sources,
        noise_prob_sources,
        ecg_prob_sources,
        ppg_prob_sources,
        np.asarray(val_clean_bank["y"].numpy(), dtype=np.int64),
        np.asarray(val_noise_bank["y"].numpy(), dtype=np.int64),
        np.asarray(val_ecg_bank["y"].numpy(), dtype=np.int64),
        np.asarray(val_ppg_bank["y"].numpy(), dtype=np.int64),
        cfg,
    )

    selected_reg_row = regression_candidates[0]
    stability_cls_row = stability_cls_candidates[0]
    val_selected_reg_base, val_selected_calib_base = base_script.build_selected_regression_pair(
        selected_reg_row,
        opt_bundle["val_bundle"]["query_corr"],
        dual_bundle["val_bundle"]["query_corr"],
        opt_bundle["val_bundle"]["calib_corr"],
        dual_bundle["val_bundle"]["calib_corr"],
        gate_models,
        cfg,
    )
    val_stability_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, clean_prob_sources, single_candidate_lookup)
    val_stability_noise_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, noise_prob_sources, single_candidate_lookup)
    val_stability_ecg_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, ecg_prob_sources, single_candidate_lookup)
    val_stability_ppg_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, ppg_prob_sources, single_candidate_lookup)
    val_stability_calib_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, val_calib_prob_sources, single_candidate_lookup)

    tail_candidates = stability_script.search_tail_correction_candidates_selective(
        val_selected_calib_base,
        val_selected_reg_base,
        val_stability_calib_prob,
        val_stability_prob,
        cfg,
    )
    selected_tail_row = tail_candidates[0]
    val_selected_tail_model = None
    if str(selected_tail_row["candidate"]) != "identity" and float(selected_tail_row["scale"]) > 0.0:
        val_selected_tail_model = prev_script.fit_tail_model(
            val_selected_calib_base,
            val_stability_calib_prob,
            float(selected_tail_row["lambda"]),
        )
    val_stability_selected = stability_script.apply_selective_tail_correction(
        val_selected_reg_base,
        val_stability_prob,
        val_selected_tail_model,
        selected_tail_row,
        cfg,
    )
    val_stability_selected_calib = stability_script.apply_selective_tail_correction(
        val_selected_calib_base,
        val_stability_calib_prob,
        val_selected_tail_model,
        selected_tail_row,
        cfg,
    )

    print("Training meta classifier on calibration stack...")
    val_meta_classifier = fit_meta_classifier(
        opt_bundle["val_raw_calib"],
        opt_val_calib_eval["query_corr"],
        dual_bundle["val_raw_calib"],
        dual_val_calib_eval["query_corr"],
        dual_val_calib_eval["hybrid_prob"],
        guided_val_calib["prob"],
        guided_val_calib["bp_pred"],
        cfg,
        seed=cfg.SEED + 1801,
    )

    val_meta_calib_prob_raw = predict_meta_classifier_prob(
        val_meta_classifier,
        opt_bundle["val_raw_calib"],
        opt_val_calib_eval["query_corr"],
        dual_bundle["val_raw_calib"],
        dual_val_calib_eval["query_corr"],
        dual_val_calib_eval["hybrid_prob"],
        guided_val_calib["prob"],
        guided_val_calib["bp_pred"],
        cfg,
    )
    val_meta_clean_prob_raw = predict_meta_classifier_prob(
        val_meta_classifier,
        opt_bundle["val_raw_query"],
        opt_bundle["val_bundle"]["query_corr"],
        dual_bundle["val_raw_query"],
        dual_bundle["val_bundle"]["query_corr"],
        dual_bundle["val_bundle"]["hybrid_prob"],
        guided_val["prob"],
        guided_val["bp_pred"],
        cfg,
    )
    val_meta_noise_prob_raw = predict_meta_classifier_prob(
        val_meta_classifier,
        val_noise_opt_raw,
        val_noise_opt_eval["query_corr"],
        val_noise_dual_raw,
        val_noise_dual_eval["query_corr"],
        val_noise_dual_eval["hybrid_prob"],
        guided_val_noise["prob"],
        guided_val_noise["bp_pred"],
        cfg,
    )
    val_meta_ecg_prob_raw = predict_meta_classifier_prob(
        val_meta_classifier,
        val_ecg_opt_raw,
        val_ecg_opt_eval["query_corr"],
        val_ecg_dual_raw,
        val_ecg_dual_eval["query_corr"],
        val_ecg_dual_eval["hybrid_prob"],
        guided_val_ecg["prob"],
        guided_val_ecg["bp_pred"],
        cfg,
    )
    val_meta_ppg_prob_raw = predict_meta_classifier_prob(
        val_meta_classifier,
        val_ppg_opt_raw,
        val_ppg_opt_eval["query_corr"],
        val_ppg_dual_raw,
        val_ppg_dual_eval["query_corr"],
        val_ppg_dual_eval["hybrid_prob"],
        guided_val_ppg["prob"],
        guided_val_ppg["bp_pred"],
        cfg,
    )

    print("Searching classification blend policy...")
    best_blend_row, blend_rows = search_classification_blend(
        val_stability_calib_prob,
        val_meta_calib_prob_raw,
        {
            "clean": val_stability_prob,
            "noise": val_stability_noise_prob,
            "ecg": val_stability_ecg_prob,
            "ppg": val_stability_ppg_prob,
        },
        {
            "clean": val_meta_clean_prob_raw,
            "noise": val_meta_noise_prob_raw,
            "ecg": val_meta_ecg_prob_raw,
            "ppg": val_meta_ppg_prob_raw,
        },
        {
            "calib": np.asarray(val_selected_calib_base["y_true_cls"], dtype=np.int64),
            "clean": np.asarray(val_selected_reg_base["y_true_cls"], dtype=np.int64),
            "noise": np.asarray(val_noise_opt_eval["query_corr"]["y_true_cls"], dtype=np.int64),
            "ecg": np.asarray(val_ecg_opt_eval["query_corr"]["y_true_cls"], dtype=np.int64),
            "ppg": np.asarray(val_ppg_opt_eval["query_corr"]["y_true_cls"], dtype=np.int64),
        },
        cfg,
    )

    val_selected_cls_prob = blend_probabilities_with_policy(val_stability_prob, val_meta_clean_prob_raw, best_blend_row["weight_meta"], best_blend_row)
    val_selected_calib_prob = blend_probabilities_with_policy(val_stability_calib_prob, val_meta_calib_prob_raw, best_blend_row["weight_meta"], best_blend_row)
    val_selected_noise_cls_prob = blend_probabilities_with_policy(val_stability_noise_prob, val_meta_noise_prob_raw, best_blend_row["weight_meta"], best_blend_row)
    val_selected_ecg_cls_prob = blend_probabilities_with_policy(val_stability_ecg_prob, val_meta_ecg_prob_raw, best_blend_row["weight_meta"], best_blend_row)
    val_selected_ppg_cls_prob = blend_probabilities_with_policy(val_stability_ppg_prob, val_meta_ppg_prob_raw, best_blend_row["weight_meta"], best_blend_row)

    classification_arbiter_rows: List[dict] = []
    val_arbiter_bundle = None
    best_arbiter_row = {
        "candidate": "selected_direct",
        "score": float(best_blend_row["score"]),
        "weight_selected_mean": 1.0,
        "weight_selected_p90": 1.0,
    }
    if bool(getattr(cfg, "ENABLE_CLASSIFICATION_ARBITER", False)):
        print("Searching final classification arbiter...")
        val_arbiter_bundle = fit_classification_arbiter_bundle(
            val_selected_calib_prob,
            val_stability_calib_prob,
            np.asarray(val_selected_calib_base["y_true_cls"], dtype=np.int64),
            cfg,
            seed=cfg.SEED + 2051,
        )
        best_arbiter_row, classification_arbiter_rows = search_classification_arbiter(
            val_arbiter_bundle,
            val_selected_calib_prob,
            val_stability_calib_prob,
            {
                "clean": val_selected_cls_prob,
                "noise": val_selected_noise_cls_prob,
                "ecg": val_selected_ecg_cls_prob,
                "ppg": val_selected_ppg_cls_prob,
            },
            {
                "clean": val_stability_prob,
                "noise": val_stability_noise_prob,
                "ecg": val_stability_ecg_prob,
                "ppg": val_stability_ppg_prob,
            },
            {
                "calib": np.asarray(val_selected_calib_base["y_true_cls"], dtype=np.int64),
                "clean": np.asarray(val_selected_reg_base["y_true_cls"], dtype=np.int64),
                "noise": np.asarray(val_noise_opt_eval["query_corr"]["y_true_cls"], dtype=np.int64),
                "ecg": np.asarray(val_ecg_opt_eval["query_corr"]["y_true_cls"], dtype=np.int64),
                "ppg": np.asarray(val_ppg_opt_eval["query_corr"]["y_true_cls"], dtype=np.int64),
            },
            cfg,
        )
        val_selected_cls_prob = apply_classification_arbiter_prob(val_arbiter_bundle, val_selected_cls_prob, val_stability_prob, best_arbiter_row, cfg)
        val_selected_noise_cls_prob = apply_classification_arbiter_prob(val_arbiter_bundle, val_selected_noise_cls_prob, val_stability_noise_prob, best_arbiter_row, cfg)
        val_selected_ecg_cls_prob = apply_classification_arbiter_prob(val_arbiter_bundle, val_selected_ecg_cls_prob, val_stability_ecg_prob, best_arbiter_row, cfg)
        val_selected_ppg_cls_prob = apply_classification_arbiter_prob(val_arbiter_bundle, val_selected_ppg_cls_prob, val_stability_ppg_prob, best_arbiter_row, cfg)
        val_selected_calib_prob = apply_classification_arbiter_prob(val_arbiter_bundle, val_selected_calib_prob, val_stability_calib_prob, best_arbiter_row, cfg)

    print("Training residual meta-regressor...")
    val_residual_bundle = fit_meta_residual_models(
        val_selected_calib_base,
        opt_bundle["val_raw_calib"],
        opt_val_calib_eval["query_corr"],
        dual_bundle["val_raw_calib"],
        dual_val_calib_eval["query_corr"],
        dual_val_calib_eval["hybrid_prob"],
        guided_val_calib["prob"],
        guided_val_calib["bp_pred"],
        val_selected_calib_prob,
        cfg,
        seed=cfg.SEED + 2201,
    )
    meta_regression_rows = search_meta_regression_candidates(
        val_residual_bundle,
        {
            "base_out": val_selected_calib_base,
            "opt_raw": opt_bundle["val_raw_calib"],
            "opt_corr": opt_val_calib_eval["query_corr"],
            "dual_raw": dual_bundle["val_raw_calib"],
            "dual_corr": dual_val_calib_eval["query_corr"],
            "dual_hybrid_prob": dual_val_calib_eval["hybrid_prob"],
            "guided_prob": guided_val_calib["prob"],
            "guided_bp": guided_val_calib["bp_pred"],
            "selected_cls_prob": val_selected_calib_prob,
        },
        {
            "base_out": val_selected_reg_base,
            "opt_raw": opt_bundle["val_raw_query"],
            "opt_corr": opt_bundle["val_bundle"]["query_corr"],
            "dual_raw": dual_bundle["val_raw_query"],
            "dual_corr": dual_bundle["val_bundle"]["query_corr"],
            "dual_hybrid_prob": dual_bundle["val_bundle"]["hybrid_prob"],
            "guided_prob": guided_val["prob"],
            "guided_bp": guided_val["bp_pred"],
            "selected_cls_prob": val_selected_cls_prob,
        },
        cfg,
    )
    regression_router_rows: List[dict] = []
    val_router_bundle = None
    if bool(getattr(cfg, "ENABLE_REGRESSION_ROUTER", False)):
        print("Searching regression expert router...")
        val_router_bundle = fit_regression_router_bundle(
            {
                "base_out": val_selected_calib_base,
                "stability_out": val_stability_selected_calib,
                "opt_raw": opt_bundle["val_raw_calib"],
                "opt_corr": opt_val_calib_eval["query_corr"],
                "dual_raw": dual_bundle["val_raw_calib"],
                "dual_corr": dual_val_calib_eval["query_corr"],
                "dual_hybrid_prob": dual_val_calib_eval["hybrid_prob"],
                "guided_prob": guided_val_calib["prob"],
                "guided_bp": guided_val_calib["bp_pred"],
                "selected_cls_prob": val_selected_calib_prob,
            },
            cfg,
            seed=cfg.SEED + 2401,
        )
        regression_router_rows = search_regression_router_candidates(
            val_router_bundle,
            {
                "base_out": val_selected_calib_base,
                "stability_out": val_stability_selected_calib,
                "opt_raw": opt_bundle["val_raw_calib"],
                "opt_corr": opt_val_calib_eval["query_corr"],
                "dual_raw": dual_bundle["val_raw_calib"],
                "dual_corr": dual_val_calib_eval["query_corr"],
                "dual_hybrid_prob": dual_val_calib_eval["hybrid_prob"],
                "guided_prob": guided_val_calib["prob"],
                "guided_bp": guided_val_calib["bp_pred"],
                "selected_cls_prob": val_selected_calib_prob,
            },
            {
                "base_out": val_selected_reg_base,
                "stability_out": val_stability_selected,
                "opt_raw": opt_bundle["val_raw_query"],
                "opt_corr": opt_bundle["val_bundle"]["query_corr"],
                "dual_raw": dual_bundle["val_raw_query"],
                "dual_corr": dual_bundle["val_bundle"]["query_corr"],
                "dual_hybrid_prob": dual_bundle["val_bundle"]["hybrid_prob"],
                "guided_prob": guided_val["prob"],
                "guided_bp": guided_val["bp_pred"],
                "selected_cls_prob": val_selected_cls_prob,
            },
            cfg,
    )
    all_regression_rows = sorted(meta_regression_rows + regression_router_rows, key=lambda row: float(row["score"]))
    best_meta_reg_row = all_regression_rows[0]
    val_query_reg_inputs = {
        "base_out": val_selected_reg_base,
        "stability_out": val_stability_selected,
        "opt_raw": opt_bundle["val_raw_query"],
        "opt_corr": opt_bundle["val_bundle"]["query_corr"],
        "dual_raw": dual_bundle["val_raw_query"],
        "dual_corr": dual_bundle["val_bundle"]["query_corr"],
        "dual_hybrid_prob": dual_bundle["val_bundle"]["hybrid_prob"],
        "guided_prob": guided_val["prob"],
        "guided_bp": guided_val["bp_pred"],
        "selected_cls_prob": val_selected_cls_prob,
    }
    val_calib_reg_inputs = {
        "base_out": val_selected_calib_base,
        "stability_out": val_stability_selected_calib,
        "opt_raw": opt_bundle["val_raw_calib"],
        "opt_corr": opt_val_calib_eval["query_corr"],
        "dual_raw": dual_bundle["val_raw_calib"],
        "dual_corr": dual_val_calib_eval["query_corr"],
        "dual_hybrid_prob": dual_val_calib_eval["hybrid_prob"],
        "guided_prob": guided_val_calib["prob"],
        "guided_bp": guided_val_calib["bp_pred"],
        "selected_cls_prob": val_selected_calib_prob,
    }
    val_query_best_residual = predict_meta_residual(
        val_residual_bundle,
        val_query_reg_inputs["base_out"],
        val_query_reg_inputs["opt_raw"],
        val_query_reg_inputs["opt_corr"],
        val_query_reg_inputs["dual_raw"],
        val_query_reg_inputs["dual_corr"],
        val_query_reg_inputs["dual_hybrid_prob"],
        val_query_reg_inputs["guided_prob"],
        val_query_reg_inputs["guided_bp"],
        val_query_reg_inputs["selected_cls_prob"],
        cfg,
    )
    val_calib_best_residual = predict_meta_residual(
        val_residual_bundle,
        val_calib_reg_inputs["base_out"],
        val_calib_reg_inputs["opt_raw"],
        val_calib_reg_inputs["opt_corr"],
        val_calib_reg_inputs["dual_raw"],
        val_calib_reg_inputs["dual_corr"],
        val_calib_reg_inputs["dual_hybrid_prob"],
        val_calib_reg_inputs["guided_prob"],
        val_calib_reg_inputs["guided_bp"],
        val_calib_reg_inputs["selected_cls_prob"],
        cfg,
    )
    val_query_router_errors = predict_regression_router_errors(val_router_bundle, val_query_reg_inputs, cfg) if val_router_bundle is not None else None
    val_calib_router_errors = predict_regression_router_errors(val_router_bundle, val_calib_reg_inputs, cfg) if val_router_bundle is not None else None
    val_best_query_reg = apply_best_regression_candidate(
        best_meta_reg_row,
        val_query_best_residual,
        val_query_reg_inputs["base_out"],
        val_query_reg_inputs["selected_cls_prob"],
        val_query_router_errors,
        val_query_reg_inputs,
        cfg,
    )
    val_best_calib_reg = apply_best_regression_candidate(
        best_meta_reg_row,
        val_calib_best_residual,
        val_calib_reg_inputs["base_out"],
        val_calib_reg_inputs["selected_cls_prob"],
        val_calib_router_errors,
        val_calib_reg_inputs,
        cfg,
    )
    risk_guard_rows: List[dict] = []
    best_risk_guard_row = {
        "candidate": "identity",
        "score": float(best_meta_reg_row["score"]),
        "clinical_under_penalty": 0.0,
        "tail_bias_penalty": 0.0,
    }
    val_risk_guard_bundle = None
    val_final_query_reg = val_best_query_reg
    val_final_calib_reg = val_best_calib_reg
    if bool(getattr(cfg, "ENABLE_RISK_GUARD", False)):
        print("Searching clinical high-risk guard...")
        val_risk_guard_bundle = fit_risk_guard_bundle(
            val_best_calib_reg,
            val_selected_calib_prob,
            cfg,
            seed=cfg.SEED + 2451,
        )
        best_risk_guard_row, risk_guard_rows = search_risk_guard_candidates(
            val_risk_guard_bundle,
            val_best_calib_reg,
            val_selected_calib_prob,
            val_best_query_reg,
            val_selected_cls_prob,
            cfg,
        )
        val_final_query_reg = apply_risk_guard_correction(
            best_risk_guard_row,
            predict_risk_guard_delta(val_risk_guard_bundle, val_best_query_reg, val_selected_cls_prob, cfg),
            val_best_query_reg,
            val_selected_cls_prob,
            cfg,
        )
        val_final_calib_reg = apply_risk_guard_correction(
            best_risk_guard_row,
            predict_risk_guard_delta(val_risk_guard_bundle, val_best_calib_reg, val_selected_calib_prob, cfg),
            val_best_calib_reg,
            val_selected_calib_prob,
            cfg,
        )

    high_bias_calibration_rows: List[dict] = []
    best_high_bias_calibration_row = {
        "candidate": "identity",
        "score": 0.0,
        "shift_mean_sbp": 0.0,
        "shift_mean_dbp": 0.0,
    }
    val_post_guard_query_reg = val_final_query_reg
    val_post_guard_calib_reg = val_final_calib_reg
    if bool(getattr(cfg, "ENABLE_HIGH_BIAS_CALIBRATOR", False)):
        print("Searching high-bias calibrator...")
        best_high_bias_calibration_row, high_bias_calibration_rows = search_high_bias_calibration_candidates(
            val_post_guard_calib_reg,
            val_selected_calib_prob,
            val_post_guard_query_reg,
            val_selected_cls_prob,
            cfg,
        )
        val_final_query_reg = apply_high_bias_calibration(
            best_high_bias_calibration_row,
            val_post_guard_query_reg,
            val_selected_cls_prob,
            cfg,
        )
        val_final_calib_reg = apply_high_bias_calibration(
            best_high_bias_calibration_row,
            val_post_guard_calib_reg,
            val_selected_calib_prob,
            cfg,
        )

    crisis_tail_fusion_rows: List[dict] = []
    best_crisis_tail_fusion_row = {
        "candidate": "identity",
        "score": 0.0,
        "fusion_gate_mean": 0.0,
        "shift_mean_sbp": 0.0,
        "shift_mean_dbp": 0.0,
    }
    val_post_high_bias_query_reg = val_final_query_reg
    val_post_high_bias_calib_reg = val_final_calib_reg
    if bool(getattr(cfg, "ENABLE_CRISIS_TAIL_FUSION", False)):
        print("Searching crisis-tail expert fusion...")
        best_crisis_tail_fusion_row, crisis_tail_fusion_rows = search_crisis_tail_fusion_candidates(
            val_post_high_bias_calib_reg,
            val_selected_calib_prob,
            val_calib_reg_inputs,
            val_post_high_bias_query_reg,
            val_selected_cls_prob,
            val_query_reg_inputs,
            cfg,
        )
        val_final_query_reg = apply_crisis_tail_fusion(
            best_crisis_tail_fusion_row,
            val_post_high_bias_query_reg,
            val_selected_cls_prob,
            val_query_reg_inputs,
            cfg,
        )
        val_final_calib_reg = apply_crisis_tail_fusion(
            best_crisis_tail_fusion_row,
            val_post_high_bias_calib_reg,
            val_selected_calib_prob,
            val_calib_reg_inputs,
            cfg,
        )

    safety_class_fusion_rows: List[dict] = []
    best_safety_fusion_row = {
        "candidate": "identity",
        "score": 0.0,
        "mean_weight": 0.0,
        "p90_weight": 0.0,
    }
    if bool(getattr(cfg, "ENABLE_SAFETY_CLASS_FUSION", False)):
        print("Searching safety-aware classification fusion...")
        best_safety_fusion_row, safety_class_fusion_rows = search_safety_class_fusion(
            val_selected_cls_prob,
            val_final_query_reg,
            cfg,
        )

    print("Fitting final meta classifier on test calibration...")
    opt_test_calib_eval = bridge_script.apply_stack_with_bundle(opt_bundle["test_raw_calib"], opt_bundle["test_bundle"], cfg, prefix="opt_test_calib")
    dual_test_calib_eval = bridge_script.apply_stack_with_bundle(dual_bundle["test_raw_calib"], dual_bundle["test_bundle"], cfg, prefix="dual_test_calib")
    test_meta_classifier = fit_meta_classifier(
        opt_bundle["test_raw_calib"],
        opt_test_calib_eval["query_corr"],
        dual_bundle["test_raw_calib"],
        dual_test_calib_eval["query_corr"],
        dual_test_calib_eval["hybrid_prob"],
        guided_test_calib["prob"],
        guided_test_calib["bp_pred"],
        cfg,
        seed=cfg.SEED + 2601,
    )

    print("Preparing final test sources...")
    test_noise_opt_raw = bridge_script.collect_outputs_bridge(opt_ckpt, loaders.test_query_loader, cfg, "opt_test_noise", noise_std=float(cfg.HEAD_NOISE_STD))
    test_noise_dual_raw = bridge_script.collect_outputs_bridge(dual_ckpt, loaders.test_query_loader, cfg, "dual_test_noise", noise_std=float(cfg.HEAD_NOISE_STD))
    test_noise_opt_eval = bridge_script.apply_stack_with_bundle(test_noise_opt_raw, opt_bundle["test_bundle"], cfg, prefix="opt_test_noise")
    test_noise_dual_eval = bridge_script.apply_stack_with_bundle(test_noise_dual_raw, dual_bundle["test_bundle"], cfg, prefix="dual_test_noise")

    test_ecg_opt_raw = bridge_script.collect_outputs_bridge(opt_ckpt, loaders.test_query_loader, cfg, "opt_test_missing_ecg", drop_modality="ecg", missing_prob=float(cfg.HEAD_MISSING_ECG))
    test_ecg_dual_raw = bridge_script.collect_outputs_bridge(dual_ckpt, loaders.test_query_loader, cfg, "dual_test_missing_ecg", drop_modality="ecg", missing_prob=float(cfg.HEAD_MISSING_ECG))
    test_ecg_opt_eval = bridge_script.apply_stack_with_bundle(test_ecg_opt_raw, opt_bundle["test_bundle"], cfg, prefix="opt_test_missing_ecg")
    test_ecg_dual_eval = bridge_script.apply_stack_with_bundle(test_ecg_dual_raw, dual_bundle["test_bundle"], cfg, prefix="dual_test_missing_ecg")

    test_ppg_opt_raw = bridge_script.collect_outputs_bridge(opt_ckpt, loaders.test_query_loader, cfg, "opt_test_missing_ppg", drop_modality="ppg", missing_prob=float(cfg.HEAD_MISSING_PPG))
    test_ppg_dual_raw = bridge_script.collect_outputs_bridge(dual_ckpt, loaders.test_query_loader, cfg, "dual_test_missing_ppg", drop_modality="ppg", missing_prob=float(cfg.HEAD_MISSING_PPG))
    test_ppg_opt_eval = bridge_script.apply_stack_with_bundle(test_ppg_opt_raw, opt_bundle["test_bundle"], cfg, prefix="opt_test_missing_ppg")
    test_ppg_dual_eval = bridge_script.apply_stack_with_bundle(test_ppg_dual_raw, dual_bundle["test_bundle"], cfg, prefix="dual_test_missing_ppg")

    test_prob_sources = base_script.build_prob_sources(opt_bundle["test_raw_query"]["y_prob_cls_from_reg"], dual_bundle["test_raw_query"]["y_prob_cls_from_reg"], dual_bundle["test_bundle"]["hybrid_prob"], guided_test["prob"])
    test_noise_prob_sources = base_script.build_prob_sources(test_noise_opt_raw["y_prob_cls_from_reg"], test_noise_dual_raw["y_prob_cls_from_reg"], test_noise_dual_eval["hybrid_prob"], guided_test_noise["prob"])
    test_ecg_prob_sources = base_script.build_prob_sources(test_ecg_opt_raw["y_prob_cls_from_reg"], test_ecg_dual_raw["y_prob_cls_from_reg"], test_ecg_dual_eval["hybrid_prob"], guided_test_ecg["prob"])
    test_ppg_prob_sources = base_script.build_prob_sources(test_ppg_opt_raw["y_prob_cls_from_reg"], test_ppg_dual_raw["y_prob_cls_from_reg"], test_ppg_dual_eval["hybrid_prob"], guided_test_ppg["prob"])
    test_calib_prob_sources = base_script.build_prob_sources(opt_bundle["test_raw_calib"]["y_prob_cls_from_reg"], dual_bundle["test_raw_calib"]["y_prob_cls_from_reg"], dual_test_calib_eval["hybrid_prob"], guided_test_calib["prob"])

    test_stability_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, test_prob_sources, single_candidate_lookup)
    test_stability_noise_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, test_noise_prob_sources, single_candidate_lookup)
    test_stability_ecg_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, test_ecg_prob_sources, single_candidate_lookup)
    test_stability_ppg_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, test_ppg_prob_sources, single_candidate_lookup)
    test_stability_calib_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, test_calib_prob_sources, single_candidate_lookup)

    test_meta_calib_prob_raw = predict_meta_classifier_prob(
        test_meta_classifier,
        opt_bundle["test_raw_calib"],
        opt_test_calib_eval["query_corr"],
        dual_bundle["test_raw_calib"],
        dual_test_calib_eval["query_corr"],
        dual_test_calib_eval["hybrid_prob"],
        guided_test_calib["prob"],
        guided_test_calib["bp_pred"],
        cfg,
    )
    test_meta_clean_prob_raw = predict_meta_classifier_prob(
        test_meta_classifier,
        opt_bundle["test_raw_query"],
        opt_bundle["test_bundle"]["query_corr"],
        dual_bundle["test_raw_query"],
        dual_bundle["test_bundle"]["query_corr"],
        dual_bundle["test_bundle"]["hybrid_prob"],
        guided_test["prob"],
        guided_test["bp_pred"],
        cfg,
    )
    test_meta_noise_prob_raw = predict_meta_classifier_prob(
        test_meta_classifier,
        test_noise_opt_raw,
        test_noise_opt_eval["query_corr"],
        test_noise_dual_raw,
        test_noise_dual_eval["query_corr"],
        test_noise_dual_eval["hybrid_prob"],
        guided_test_noise["prob"],
        guided_test_noise["bp_pred"],
        cfg,
    )
    test_meta_ecg_prob_raw = predict_meta_classifier_prob(
        test_meta_classifier,
        test_ecg_opt_raw,
        test_ecg_opt_eval["query_corr"],
        test_ecg_dual_raw,
        test_ecg_dual_eval["query_corr"],
        test_ecg_dual_eval["hybrid_prob"],
        guided_test_ecg["prob"],
        guided_test_ecg["bp_pred"],
        cfg,
    )
    test_meta_ppg_prob_raw = predict_meta_classifier_prob(
        test_meta_classifier,
        test_ppg_opt_raw,
        test_ppg_opt_eval["query_corr"],
        test_ppg_dual_raw,
        test_ppg_dual_eval["query_corr"],
        test_ppg_dual_eval["hybrid_prob"],
        guided_test_ppg["prob"],
        guided_test_ppg["bp_pred"],
        cfg,
    )

    selected_cls_prob = blend_probabilities_with_policy(test_stability_prob, test_meta_clean_prob_raw, best_blend_row["weight_meta"], best_blend_row)
    selected_noise_cls_prob = blend_probabilities_with_policy(test_stability_noise_prob, test_meta_noise_prob_raw, best_blend_row["weight_meta"], best_blend_row)
    selected_ecg_cls_prob = blend_probabilities_with_policy(test_stability_ecg_prob, test_meta_ecg_prob_raw, best_blend_row["weight_meta"], best_blend_row)
    selected_ppg_cls_prob = blend_probabilities_with_policy(test_stability_ppg_prob, test_meta_ppg_prob_raw, best_blend_row["weight_meta"], best_blend_row)
    selected_calib_cls_prob = blend_probabilities_with_policy(test_stability_calib_prob, test_meta_calib_prob_raw, best_blend_row["weight_meta"], best_blend_row)
    test_arbiter_bundle = None
    if bool(getattr(cfg, "ENABLE_CLASSIFICATION_ARBITER", False)):
        test_arbiter_bundle = fit_classification_arbiter_bundle(
            selected_calib_cls_prob,
            test_stability_calib_prob,
            np.asarray(opt_bundle["test_bundle"]["calib_corr"]["y_true_cls"], dtype=np.int64),
            cfg,
            seed=cfg.SEED + 2801,
        )
        selected_cls_prob = apply_classification_arbiter_prob(test_arbiter_bundle, selected_cls_prob, test_stability_prob, best_arbiter_row, cfg)
        selected_noise_cls_prob = apply_classification_arbiter_prob(test_arbiter_bundle, selected_noise_cls_prob, test_stability_noise_prob, best_arbiter_row, cfg)
        selected_ecg_cls_prob = apply_classification_arbiter_prob(test_arbiter_bundle, selected_ecg_cls_prob, test_stability_ecg_prob, best_arbiter_row, cfg)
        selected_ppg_cls_prob = apply_classification_arbiter_prob(test_arbiter_bundle, selected_ppg_cls_prob, test_stability_ppg_prob, best_arbiter_row, cfg)
        selected_calib_cls_prob = apply_classification_arbiter_prob(test_arbiter_bundle, selected_calib_cls_prob, test_stability_calib_prob, best_arbiter_row, cfg)

    selected_reg_base, selected_calib_base = base_script.build_selected_regression_pair(
        selected_reg_row,
        opt_bundle["test_bundle"]["query_corr"],
        dual_bundle["test_bundle"]["query_corr"],
        opt_bundle["test_bundle"]["calib_corr"],
        dual_bundle["test_bundle"]["calib_corr"],
        gate_models,
        cfg,
    )
    selected_tail_model = None
    if str(selected_tail_row["candidate"]) != "identity" and float(selected_tail_row["scale"]) > 0.0:
        selected_tail_model = prev_script.fit_tail_model(selected_calib_base, test_stability_calib_prob, float(selected_tail_row["lambda"]))
    stability_selected = stability_script.apply_selective_tail_correction(selected_reg_base, test_stability_prob, selected_tail_model, selected_tail_row, cfg)
    stability_selected_calib = stability_script.apply_selective_tail_correction(selected_calib_base, test_stability_calib_prob, selected_tail_model, selected_tail_row, cfg)

    print("Fitting final residual meta-regressor on test calibration...")
    test_residual_bundle = fit_meta_residual_models(
        selected_calib_base,
        opt_bundle["test_raw_calib"],
        opt_test_calib_eval["query_corr"],
        dual_bundle["test_raw_calib"],
        dual_test_calib_eval["query_corr"],
        dual_test_calib_eval["hybrid_prob"],
        guided_test_calib["prob"],
        guided_test_calib["bp_pred"],
        selected_calib_cls_prob,
        cfg,
        seed=cfg.SEED + 3001,
    )
    test_router_bundle = None
    if bool(getattr(cfg, "ENABLE_REGRESSION_ROUTER", False)):
        test_router_bundle = fit_regression_router_bundle(
            {
                "base_out": selected_calib_base,
                "stability_out": stability_selected_calib,
                "opt_raw": opt_bundle["test_raw_calib"],
                "opt_corr": opt_test_calib_eval["query_corr"],
                "dual_raw": dual_bundle["test_raw_calib"],
                "dual_corr": dual_test_calib_eval["query_corr"],
                "dual_hybrid_prob": dual_test_calib_eval["hybrid_prob"],
                "guided_prob": guided_test_calib["prob"],
                "guided_bp": guided_test_calib["bp_pred"],
                "selected_cls_prob": selected_calib_cls_prob,
            },
            cfg,
            seed=cfg.SEED + 3201,
        )
    meta_test_residual = predict_meta_residual(
        test_residual_bundle,
        selected_reg_base,
        opt_bundle["test_raw_query"],
        opt_bundle["test_bundle"]["query_corr"],
        dual_bundle["test_raw_query"],
        dual_bundle["test_bundle"]["query_corr"],
        dual_bundle["test_bundle"]["hybrid_prob"],
        guided_test["prob"],
        guided_test["bp_pred"],
        selected_cls_prob,
        cfg,
    )
    meta_test_calib_residual = predict_meta_residual(
        test_residual_bundle,
        selected_calib_base,
        opt_bundle["test_raw_calib"],
        opt_test_calib_eval["query_corr"],
        dual_bundle["test_raw_calib"],
        dual_test_calib_eval["query_corr"],
        dual_test_calib_eval["hybrid_prob"],
        guided_test_calib["prob"],
        guided_test_calib["bp_pred"],
        selected_calib_cls_prob,
        cfg,
    )
    test_router_errors = None
    test_router_calib_errors = None
    if test_router_bundle is not None:
        test_router_errors = predict_regression_router_errors(
            test_router_bundle,
            {
                "base_out": selected_reg_base,
                "stability_out": stability_selected,
                "opt_raw": opt_bundle["test_raw_query"],
                "opt_corr": opt_bundle["test_bundle"]["query_corr"],
                "dual_raw": dual_bundle["test_raw_query"],
                "dual_corr": dual_bundle["test_bundle"]["query_corr"],
                "dual_hybrid_prob": dual_bundle["test_bundle"]["hybrid_prob"],
                "guided_prob": guided_test["prob"],
                "guided_bp": guided_test["bp_pred"],
                "selected_cls_prob": selected_cls_prob,
            },
            cfg,
        )
        test_router_calib_errors = predict_regression_router_errors(
            test_router_bundle,
            {
                "base_out": selected_calib_base,
                "stability_out": stability_selected_calib,
                "opt_raw": opt_bundle["test_raw_calib"],
                "opt_corr": opt_test_calib_eval["query_corr"],
                "dual_raw": dual_bundle["test_raw_calib"],
                "dual_corr": dual_test_calib_eval["query_corr"],
                "dual_hybrid_prob": dual_test_calib_eval["hybrid_prob"],
                "guided_prob": guided_test_calib["prob"],
                "guided_bp": guided_test_calib["bp_pred"],
                "selected_cls_prob": selected_calib_cls_prob,
            },
            cfg,
        )
    selected_reg = apply_best_regression_candidate(
        best_meta_reg_row,
        meta_test_residual,
        selected_reg_base,
        selected_cls_prob,
        test_router_errors,
        {
            "base_out": selected_reg_base,
            "stability_out": stability_selected,
            "opt_raw": opt_bundle["test_raw_query"],
            "opt_corr": opt_bundle["test_bundle"]["query_corr"],
            "dual_raw": dual_bundle["test_raw_query"],
            "dual_corr": dual_bundle["test_bundle"]["query_corr"],
            "dual_hybrid_prob": dual_bundle["test_bundle"]["hybrid_prob"],
            "guided_prob": guided_test["prob"],
            "guided_bp": guided_test["bp_pred"],
            "selected_cls_prob": selected_cls_prob,
        },
        cfg,
    )
    selected_calib = apply_best_regression_candidate(
        best_meta_reg_row,
        meta_test_calib_residual,
        selected_calib_base,
        selected_calib_cls_prob,
        test_router_calib_errors,
        {
            "base_out": selected_calib_base,
            "stability_out": stability_selected_calib,
            "opt_raw": opt_bundle["test_raw_calib"],
            "opt_corr": opt_test_calib_eval["query_corr"],
            "dual_raw": dual_bundle["test_raw_calib"],
            "dual_corr": dual_test_calib_eval["query_corr"],
            "dual_hybrid_prob": dual_test_calib_eval["hybrid_prob"],
            "guided_prob": guided_test_calib["prob"],
            "guided_bp": guided_test_calib["bp_pred"],
            "selected_cls_prob": selected_calib_cls_prob,
        },
        cfg,
    )
    selected_reg_pre_guard = selected_reg
    selected_calib_pre_guard = selected_calib
    test_risk_guard_bundle = None
    if bool(getattr(cfg, "ENABLE_RISK_GUARD", False)):
        print("Fitting clinical high-risk guard on test calibration...")
        test_risk_guard_bundle = fit_risk_guard_bundle(
            selected_calib,
            selected_calib_cls_prob,
            cfg,
            seed=cfg.SEED + 3301,
        )
        selected_reg = apply_risk_guard_correction(
            best_risk_guard_row,
            predict_risk_guard_delta(test_risk_guard_bundle, selected_reg, selected_cls_prob, cfg),
            selected_reg,
            selected_cls_prob,
            cfg,
        )
        selected_calib = apply_risk_guard_correction(
            best_risk_guard_row,
            predict_risk_guard_delta(test_risk_guard_bundle, selected_calib, selected_calib_cls_prob, cfg),
            selected_calib,
            selected_calib_cls_prob,
            cfg,
        )

    selected_reg_pre_bias_calibration = selected_reg
    selected_calib_pre_bias_calibration = selected_calib
    if bool(getattr(cfg, "ENABLE_HIGH_BIAS_CALIBRATOR", False)):
        selected_reg = apply_high_bias_calibration(
            best_high_bias_calibration_row,
            selected_reg,
            selected_cls_prob,
            cfg,
        )
        selected_calib = apply_high_bias_calibration(
            best_high_bias_calibration_row,
            selected_calib,
            selected_calib_cls_prob,
            cfg,
        )

    selected_reg_post_high_bias_calibration = selected_reg
    selected_calib_post_high_bias_calibration = selected_calib
    selected_reg_pre_crisis_tail_fusion = selected_reg
    selected_calib_pre_crisis_tail_fusion = selected_calib
    if bool(getattr(cfg, "ENABLE_CRISIS_TAIL_FUSION", False)):
        selected_reg = apply_crisis_tail_fusion(
            best_crisis_tail_fusion_row,
            selected_reg_pre_crisis_tail_fusion,
            selected_cls_prob,
            {
                "base_out": selected_reg_base,
                "stability_out": stability_selected,
                "opt_raw": opt_bundle["test_raw_query"],
                "opt_corr": opt_bundle["test_bundle"]["query_corr"],
                "dual_raw": dual_bundle["test_raw_query"],
                "dual_corr": dual_bundle["test_bundle"]["query_corr"],
                "dual_hybrid_prob": dual_bundle["test_bundle"]["hybrid_prob"],
                "guided_prob": guided_test["prob"],
                "guided_bp": guided_test["bp_pred"],
                "selected_cls_prob": selected_cls_prob,
            },
            cfg,
        )
        selected_calib = apply_crisis_tail_fusion(
            best_crisis_tail_fusion_row,
            selected_calib_pre_crisis_tail_fusion,
            selected_calib_cls_prob,
            {
                "base_out": selected_calib_base,
                "stability_out": stability_selected_calib,
                "opt_raw": opt_bundle["test_raw_calib"],
                "opt_corr": opt_test_calib_eval["query_corr"],
                "dual_raw": dual_bundle["test_raw_calib"],
                "dual_corr": dual_test_calib_eval["query_corr"],
                "dual_hybrid_prob": dual_test_calib_eval["hybrid_prob"],
                "guided_prob": guided_test_calib["prob"],
                "guided_bp": guided_test_calib["bp_pred"],
                "selected_cls_prob": selected_calib_cls_prob,
            },
            cfg,
        )

    selected_cls_prob_pre_safety = np.asarray(selected_cls_prob, dtype=np.float32)
    selected_safety_diag = apply_safety_class_fusion_prob(
        best_safety_fusion_row,
        selected_cls_prob,
        selected_reg,
        cfg,
    )
    if bool(getattr(cfg, "ENABLE_SAFETY_CLASS_FUSION", False)):
        selected_cls_prob = selected_safety_diag["prob"]
        selected_calib_cls_prob = apply_safety_class_fusion_prob(
            best_safety_fusion_row,
            selected_calib_cls_prob,
            selected_calib,
            cfg,
        )["prob"]

    y_true_cls_test = np.asarray(selected_reg["y_true_cls"], dtype=np.int64)
    selected_prefusion_cls_pred = np.asarray(selected_cls_prob_pre_safety, dtype=np.float32).argmax(axis=1).astype(np.int64)
    selected_prefusion_cls_metrics = stage_script.risk_classification_metrics(
        y_true_cls_test,
        selected_prefusion_cls_pred,
        selected_cls_prob_pre_safety,
        cfg,
        prefix="selected_prefusion",
    )
    selected_prefusion_compat = stage_script.hybrid_metrics_for_compat(selected_prefusion_cls_metrics, "selected_prefusion")
    selected_cls_pred = np.asarray(selected_cls_prob, dtype=np.float32).argmax(axis=1).astype(np.int64)
    selected_cls_metrics = stage_script.risk_classification_metrics(y_true_cls_test, selected_cls_pred, selected_cls_prob, cfg, prefix="selected_final")
    selected_compat = stage_script.hybrid_metrics_for_compat(selected_cls_metrics, "selected_final")

    stability_cls_pred = np.asarray(test_stability_prob, dtype=np.float32).argmax(axis=1).astype(np.int64)
    stability_cls_metrics = stage_script.risk_classification_metrics(y_true_cls_test, stability_cls_pred, test_stability_prob, cfg, prefix="stability_selected")
    stability_compat = stage_script.hybrid_metrics_for_compat(stability_cls_metrics, "stability_selected")

    meta_raw_test_metrics = stage_script.risk_classification_metrics(y_true_cls_test, test_meta_clean_prob_raw.argmax(axis=1).astype(np.int64), test_meta_clean_prob_raw, cfg, prefix="meta_stack_test")
    meta_raw_compat = stage_script.hybrid_metrics_for_compat(meta_raw_test_metrics, "meta_stack_test")
    opt_raw_test_prob = bridge_script.normalize_prob(opt_bundle["test_raw_query"]["y_prob_cls_from_reg"])
    opt_raw_test_metrics = stage_script.risk_classification_metrics(y_true_cls_test, opt_raw_test_prob.argmax(axis=1).astype(np.int64), opt_raw_test_prob, cfg, prefix="opt_raw_test")
    dual_hybrid_metrics = dual_bundle["test_bundle"]["hybrid_metrics"]
    dual_hybrid_compat = bridge_script.hybrid_metrics_for_compat_auto(dual_hybrid_metrics)

    low, high, conformal_default = stage_script.conformal_from_outputs(selected_calib, selected_reg, alpha=cfg.CONFORMAL_ALPHA)
    conformal_rows = []
    for alpha in cfg.CONFORMAL_ALPHAS:
        _, _, met = stage_script.conformal_from_outputs(selected_calib, selected_reg, alpha=alpha)
        conformal_rows.append({"alpha": alpha, **met})

    cond_rows = stage_script.build_conditional_coverage_table(selected_reg["y_true_reg"], low, high, selected_reg["quality"], cfg)
    calib_curve_rows = stage_script.build_calibration_curve_table(y_true_cls_test, selected_cls_prob, n_bins=cfg.ECE_BINS)
    stability_curve_rows = stage_script.build_calibration_curve_table(y_true_cls_test, test_stability_prob, n_bins=cfg.ECE_BINS)
    meta_curve_rows = stage_script.build_calibration_curve_table(y_true_cls_test, test_meta_clean_prob_raw, n_bins=cfg.ECE_BINS)
    opt_curve_rows = stage_script.build_calibration_curve_table(y_true_cls_test, opt_raw_test_prob, n_bins=cfg.ECE_BINS)
    dual_curve_rows = stage_script.build_calibration_curve_table(y_true_cls_test, bridge_script.normalize_prob(dual_bundle["test_raw_query"]["y_prob_cls_from_reg"]), n_bins=cfg.ECE_BINS)
    error_cdf_rows = stage_script.build_error_cdf_rows(selected_reg["y_true_reg"], selected_reg["y_pred_reg"])
    split_dist_rows = stage_script.build_split_distribution_rows(loaders.split_datasets, cfg)
    bp_range_rows = stage_script.build_bp_range_table(selected_reg["y_true_reg"], selected_reg["y_pred_reg"])
    clinical_guard_comparison_rows = build_clinical_guard_comparison_rows(selected_reg_pre_guard, selected_reg_pre_bias_calibration)
    high_bias_calibration_comparison_rows = build_clinical_guard_comparison_rows(
        selected_reg_pre_bias_calibration,
        selected_reg_post_high_bias_calibration,
    )
    crisis_tail_fusion_comparison_rows = build_clinical_guard_comparison_rows(
        selected_reg_pre_crisis_tail_fusion,
        selected_reg,
    )
    subject_rows = stage_script.build_subjectwise_error_table(selected_reg["y_true_reg"], selected_reg["y_pred_reg"], selected_reg["subject_ids"])
    subject_gain_optlong_rows = stage_script.build_subject_gain_table(opt_bundle["test_raw_query"], opt_bundle["test_bundle"]["query_corr"])
    subject_gain_dual_rows = stage_script.build_subject_gain_table(dual_bundle["test_raw_query"], dual_bundle["test_bundle"]["query_corr"])
    safety_transition_rows: List[dict] = []
    safety_class_profile_rows: List[dict] = []
    safety_disagreement_profile_rows: List[dict] = []
    if bool(getattr(cfg, "ENABLE_SAFETY_CLASS_FUSION", False)):
        safety_transition_rows = build_safety_class_transition_rows(
            selected_cls_prob_pre_safety,
            selected_cls_prob,
            selected_safety_diag["weight"],
            cfg,
        )
        safety_class_profile_rows, safety_disagreement_profile_rows = build_safety_class_profile_rows(
            selected_safety_diag,
            selected_cls_prob_pre_safety,
            selected_cls_prob,
            y_true_cls_test,
            cfg,
        )

    noise_rows: List[dict] = []
    for noise_std in cfg.NOISE_STDS:
        noise_opt_raw = bridge_script.collect_outputs_bridge(opt_ckpt, loaders.test_query_loader, cfg, f"meta_noise_opt_{str(noise_std).replace('.', '_')}", noise_std=float(noise_std))
        noise_dual_raw = bridge_script.collect_outputs_bridge(dual_ckpt, loaders.test_query_loader, cfg, f"meta_noise_dual_{str(noise_std).replace('.', '_')}", noise_std=float(noise_std))
        noise_opt_eval = bridge_script.apply_stack_with_bundle(noise_opt_raw, opt_bundle["test_bundle"], cfg, prefix=f"meta_noise_opt_{str(noise_std).replace('.', '_')}")
        noise_dual_eval = bridge_script.apply_stack_with_bundle(noise_dual_raw, dual_bundle["test_bundle"], cfg, prefix=f"meta_noise_dual_{str(noise_std).replace('.', '_')}")
        noise_guided_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.test_query_loader, cfg, f"meta_noise_bank_{str(noise_std).replace('.', '_')}", noise_std=float(noise_std))
        noise_guided = predict_guided_bundle(head_model, head_state, noise_guided_bank, cfg, f"meta_noise_guided_{str(noise_std).replace('.', '_')}", temperature, best_policy_row)
        stability_noise_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, base_script.build_prob_sources(noise_opt_raw["y_prob_cls_from_reg"], noise_dual_raw["y_prob_cls_from_reg"], noise_dual_eval["hybrid_prob"], noise_guided["prob"]), single_candidate_lookup)
        meta_noise_prob = predict_meta_classifier_prob(test_meta_classifier, noise_opt_raw, noise_opt_eval["query_corr"], noise_dual_raw, noise_dual_eval["query_corr"], noise_dual_eval["hybrid_prob"], noise_guided["prob"], noise_guided["bp_pred"], cfg)
        noise_cls_prob = blend_probabilities_with_policy(stability_noise_prob, meta_noise_prob, best_blend_row["weight_meta"], best_blend_row)
        if test_arbiter_bundle is not None:
            noise_cls_prob = apply_classification_arbiter_prob(test_arbiter_bundle, noise_cls_prob, stability_noise_prob, best_arbiter_row, cfg)
        noise_base = base_script.build_selected_regression_query(selected_reg_row, noise_opt_eval["query_corr"], noise_dual_eval["query_corr"], gate_models, cfg)
        noise_stability = stability_script.apply_selective_tail_correction(
            noise_base,
            stability_noise_prob,
            selected_tail_model,
            selected_tail_row,
            cfg,
        )
        noise_residual = predict_meta_residual(test_residual_bundle, noise_base, noise_opt_raw, noise_opt_eval["query_corr"], noise_dual_raw, noise_dual_eval["query_corr"], noise_dual_eval["hybrid_prob"], noise_guided["prob"], noise_guided["bp_pred"], noise_cls_prob, cfg)
        noise_router_errors = None
        if test_router_bundle is not None:
            noise_router_errors = predict_regression_router_errors(
                test_router_bundle,
                {
                    "base_out": noise_base,
                    "stability_out": noise_stability,
                    "opt_raw": noise_opt_raw,
                    "opt_corr": noise_opt_eval["query_corr"],
                    "dual_raw": noise_dual_raw,
                    "dual_corr": noise_dual_eval["query_corr"],
                    "dual_hybrid_prob": noise_dual_eval["hybrid_prob"],
                    "guided_prob": noise_guided["prob"],
                    "guided_bp": noise_guided["bp_pred"],
                    "selected_cls_prob": noise_cls_prob,
                },
                cfg,
            )
        noise_reg = apply_best_regression_candidate(
            best_meta_reg_row,
            noise_residual,
            noise_base,
            noise_cls_prob,
            noise_router_errors,
            {
                "base_out": noise_base,
                "stability_out": noise_stability,
                "opt_raw": noise_opt_raw,
                "opt_corr": noise_opt_eval["query_corr"],
                "dual_raw": noise_dual_raw,
                "dual_corr": noise_dual_eval["query_corr"],
                "dual_hybrid_prob": noise_dual_eval["hybrid_prob"],
                "guided_prob": noise_guided["prob"],
                "guided_bp": noise_guided["bp_pred"],
                "selected_cls_prob": noise_cls_prob,
            },
            cfg,
        )
        if test_risk_guard_bundle is not None:
            noise_reg = apply_risk_guard_correction(
                best_risk_guard_row,
                predict_risk_guard_delta(test_risk_guard_bundle, noise_reg, noise_cls_prob, cfg),
                noise_reg,
                noise_cls_prob,
                cfg,
            )
        if bool(getattr(cfg, "ENABLE_HIGH_BIAS_CALIBRATOR", False)):
            noise_reg = apply_high_bias_calibration(
                best_high_bias_calibration_row,
                noise_reg,
                noise_cls_prob,
                cfg,
            )
        if bool(getattr(cfg, "ENABLE_CRISIS_TAIL_FUSION", False)):
            noise_reg = apply_crisis_tail_fusion(
                best_crisis_tail_fusion_row,
                noise_reg,
                noise_cls_prob,
                {
                    "base_out": noise_base,
                    "stability_out": noise_stability,
                    "opt_raw": noise_opt_raw,
                    "opt_corr": noise_opt_eval["query_corr"],
                    "dual_raw": noise_dual_raw,
                    "dual_corr": noise_dual_eval["query_corr"],
                    "dual_hybrid_prob": noise_dual_eval["hybrid_prob"],
                    "guided_prob": noise_guided["prob"],
                    "guided_bp": noise_guided["bp_pred"],
                    "selected_cls_prob": noise_cls_prob,
                },
                cfg,
            )
        if bool(getattr(cfg, "ENABLE_SAFETY_CLASS_FUSION", False)):
            noise_cls_prob = apply_safety_class_fusion_prob(
                best_safety_fusion_row,
                noise_cls_prob,
                noise_reg,
                cfg,
            )["prob"]
        noise_prefix = f"selected_noise_{str(noise_std).replace('.', '_')}"
        noise_metrics = stage_script.risk_classification_metrics(np.asarray(noise_reg["y_true_cls"], dtype=np.int64), noise_cls_prob.argmax(axis=1).astype(np.int64), noise_cls_prob, cfg, prefix=noise_prefix)
        noise_rows.append({"noise_std": float(noise_std), **noise_reg["metrics_reg"], **stage_script.hybrid_metrics_for_compat(noise_metrics, noise_prefix), **noise_reg["uncertainty_metrics"]})

    missing_rows: List[dict] = []
    for missing_prob in cfg.MISSING_PROBS:
        ppg_opt_raw = bridge_script.collect_outputs_bridge(opt_ckpt, loaders.test_query_loader, cfg, f"meta_ppg_missing_opt_{str(missing_prob).replace('.', '_')}", drop_modality="ppg", missing_prob=float(missing_prob))
        ppg_dual_raw = bridge_script.collect_outputs_bridge(dual_ckpt, loaders.test_query_loader, cfg, f"meta_ppg_missing_dual_{str(missing_prob).replace('.', '_')}", drop_modality="ppg", missing_prob=float(missing_prob))
        ppg_opt_eval = bridge_script.apply_stack_with_bundle(ppg_opt_raw, opt_bundle["test_bundle"], cfg, prefix=f"meta_ppg_missing_opt_{str(missing_prob).replace('.', '_')}")
        ppg_dual_eval = bridge_script.apply_stack_with_bundle(ppg_dual_raw, dual_bundle["test_bundle"], cfg, prefix=f"meta_ppg_missing_dual_{str(missing_prob).replace('.', '_')}")
        ppg_guided_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.test_query_loader, cfg, f"meta_ppg_bank_{str(missing_prob).replace('.', '_')}", drop_modality="ppg", missing_prob=float(missing_prob))
        ppg_guided = predict_guided_bundle(head_model, head_state, ppg_guided_bank, cfg, f"meta_ppg_guided_{str(missing_prob).replace('.', '_')}", temperature, best_policy_row)

        ecg_opt_raw = bridge_script.collect_outputs_bridge(opt_ckpt, loaders.test_query_loader, cfg, f"meta_ecg_missing_opt_{str(missing_prob).replace('.', '_')}", drop_modality="ecg", missing_prob=float(missing_prob))
        ecg_dual_raw = bridge_script.collect_outputs_bridge(dual_ckpt, loaders.test_query_loader, cfg, f"meta_ecg_missing_dual_{str(missing_prob).replace('.', '_')}", drop_modality="ecg", missing_prob=float(missing_prob))
        ecg_opt_eval = bridge_script.apply_stack_with_bundle(ecg_opt_raw, opt_bundle["test_bundle"], cfg, prefix=f"meta_ecg_missing_opt_{str(missing_prob).replace('.', '_')}")
        ecg_dual_eval = bridge_script.apply_stack_with_bundle(ecg_dual_raw, dual_bundle["test_bundle"], cfg, prefix=f"meta_ecg_missing_dual_{str(missing_prob).replace('.', '_')}")
        ecg_guided_bank = base_script.extract_dualbackbone_bank(opt_ckpt, dual_ckpt, loaders.test_query_loader, cfg, f"meta_ecg_bank_{str(missing_prob).replace('.', '_')}", drop_modality="ecg", missing_prob=float(missing_prob))
        ecg_guided = predict_guided_bundle(head_model, head_state, ecg_guided_bank, cfg, f"meta_ecg_guided_{str(missing_prob).replace('.', '_')}", temperature, best_policy_row)

        ppg_stability_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, base_script.build_prob_sources(ppg_opt_raw["y_prob_cls_from_reg"], ppg_dual_raw["y_prob_cls_from_reg"], ppg_dual_eval["hybrid_prob"], ppg_guided["prob"]), single_candidate_lookup)
        ecg_stability_prob = stability_script.build_selected_classification_prob_any(stability_cls_row, base_script.build_prob_sources(ecg_opt_raw["y_prob_cls_from_reg"], ecg_dual_raw["y_prob_cls_from_reg"], ecg_dual_eval["hybrid_prob"], ecg_guided["prob"]), single_candidate_lookup)
        ppg_meta_prob = predict_meta_classifier_prob(test_meta_classifier, ppg_opt_raw, ppg_opt_eval["query_corr"], ppg_dual_raw, ppg_dual_eval["query_corr"], ppg_dual_eval["hybrid_prob"], ppg_guided["prob"], ppg_guided["bp_pred"], cfg)
        ecg_meta_prob = predict_meta_classifier_prob(test_meta_classifier, ecg_opt_raw, ecg_opt_eval["query_corr"], ecg_dual_raw, ecg_dual_eval["query_corr"], ecg_dual_eval["hybrid_prob"], ecg_guided["prob"], ecg_guided["bp_pred"], cfg)
        ppg_cls_prob = blend_probabilities_with_policy(ppg_stability_prob, ppg_meta_prob, best_blend_row["weight_meta"], best_blend_row)
        ecg_cls_prob = blend_probabilities_with_policy(ecg_stability_prob, ecg_meta_prob, best_blend_row["weight_meta"], best_blend_row)
        if test_arbiter_bundle is not None:
            ppg_cls_prob = apply_classification_arbiter_prob(test_arbiter_bundle, ppg_cls_prob, ppg_stability_prob, best_arbiter_row, cfg)
            ecg_cls_prob = apply_classification_arbiter_prob(test_arbiter_bundle, ecg_cls_prob, ecg_stability_prob, best_arbiter_row, cfg)
        ppg_base = base_script.build_selected_regression_query(selected_reg_row, ppg_opt_eval["query_corr"], ppg_dual_eval["query_corr"], gate_models, cfg)
        ecg_base = base_script.build_selected_regression_query(selected_reg_row, ecg_opt_eval["query_corr"], ecg_dual_eval["query_corr"], gate_models, cfg)
        ppg_stability = stability_script.apply_selective_tail_correction(
            ppg_base,
            ppg_stability_prob,
            selected_tail_model,
            selected_tail_row,
            cfg,
        )
        ecg_stability = stability_script.apply_selective_tail_correction(
            ecg_base,
            ecg_stability_prob,
            selected_tail_model,
            selected_tail_row,
            cfg,
        )
        ppg_residual = predict_meta_residual(test_residual_bundle, ppg_base, ppg_opt_raw, ppg_opt_eval["query_corr"], ppg_dual_raw, ppg_dual_eval["query_corr"], ppg_dual_eval["hybrid_prob"], ppg_guided["prob"], ppg_guided["bp_pred"], ppg_cls_prob, cfg)
        ecg_residual = predict_meta_residual(test_residual_bundle, ecg_base, ecg_opt_raw, ecg_opt_eval["query_corr"], ecg_dual_raw, ecg_dual_eval["query_corr"], ecg_dual_eval["hybrid_prob"], ecg_guided["prob"], ecg_guided["bp_pred"], ecg_cls_prob, cfg)
        ppg_router_errors = None
        ecg_router_errors = None
        if test_router_bundle is not None:
            ppg_router_errors = predict_regression_router_errors(
                test_router_bundle,
                {
                    "base_out": ppg_base,
                    "stability_out": ppg_stability,
                    "opt_raw": ppg_opt_raw,
                    "opt_corr": ppg_opt_eval["query_corr"],
                    "dual_raw": ppg_dual_raw,
                    "dual_corr": ppg_dual_eval["query_corr"],
                    "dual_hybrid_prob": ppg_dual_eval["hybrid_prob"],
                    "guided_prob": ppg_guided["prob"],
                    "guided_bp": ppg_guided["bp_pred"],
                    "selected_cls_prob": ppg_cls_prob,
                },
                cfg,
            )
            ecg_router_errors = predict_regression_router_errors(
                test_router_bundle,
                {
                    "base_out": ecg_base,
                    "stability_out": ecg_stability,
                    "opt_raw": ecg_opt_raw,
                    "opt_corr": ecg_opt_eval["query_corr"],
                    "dual_raw": ecg_dual_raw,
                    "dual_corr": ecg_dual_eval["query_corr"],
                    "dual_hybrid_prob": ecg_dual_eval["hybrid_prob"],
                    "guided_prob": ecg_guided["prob"],
                    "guided_bp": ecg_guided["bp_pred"],
                    "selected_cls_prob": ecg_cls_prob,
                },
                cfg,
            )
        ppg_reg = apply_best_regression_candidate(
            best_meta_reg_row,
            ppg_residual,
            ppg_base,
            ppg_cls_prob,
            ppg_router_errors,
            {
                "base_out": ppg_base,
                "stability_out": ppg_stability,
                "opt_raw": ppg_opt_raw,
                "opt_corr": ppg_opt_eval["query_corr"],
                "dual_raw": ppg_dual_raw,
                "dual_corr": ppg_dual_eval["query_corr"],
                "dual_hybrid_prob": ppg_dual_eval["hybrid_prob"],
                "guided_prob": ppg_guided["prob"],
                "guided_bp": ppg_guided["bp_pred"],
                "selected_cls_prob": ppg_cls_prob,
            },
            cfg,
        )
        ecg_reg = apply_best_regression_candidate(
            best_meta_reg_row,
            ecg_residual,
            ecg_base,
            ecg_cls_prob,
            ecg_router_errors,
            {
                "base_out": ecg_base,
                "stability_out": ecg_stability,
                "opt_raw": ecg_opt_raw,
                "opt_corr": ecg_opt_eval["query_corr"],
                "dual_raw": ecg_dual_raw,
                "dual_corr": ecg_dual_eval["query_corr"],
                "dual_hybrid_prob": ecg_dual_eval["hybrid_prob"],
                "guided_prob": ecg_guided["prob"],
                "guided_bp": ecg_guided["bp_pred"],
                "selected_cls_prob": ecg_cls_prob,
            },
            cfg,
        )
        if test_risk_guard_bundle is not None:
            ppg_reg = apply_risk_guard_correction(
                best_risk_guard_row,
                predict_risk_guard_delta(test_risk_guard_bundle, ppg_reg, ppg_cls_prob, cfg),
                ppg_reg,
                ppg_cls_prob,
                cfg,
            )
            ecg_reg = apply_risk_guard_correction(
                best_risk_guard_row,
                predict_risk_guard_delta(test_risk_guard_bundle, ecg_reg, ecg_cls_prob, cfg),
                ecg_reg,
                ecg_cls_prob,
                cfg,
            )
        if bool(getattr(cfg, "ENABLE_HIGH_BIAS_CALIBRATOR", False)):
            ppg_reg = apply_high_bias_calibration(
                best_high_bias_calibration_row,
                ppg_reg,
                ppg_cls_prob,
                cfg,
            )
            ecg_reg = apply_high_bias_calibration(
                best_high_bias_calibration_row,
                ecg_reg,
                ecg_cls_prob,
                cfg,
            )
        if bool(getattr(cfg, "ENABLE_CRISIS_TAIL_FUSION", False)):
            ppg_reg = apply_crisis_tail_fusion(
                best_crisis_tail_fusion_row,
                ppg_reg,
                ppg_cls_prob,
                {
                    "base_out": ppg_base,
                    "stability_out": ppg_stability,
                    "opt_raw": ppg_opt_raw,
                    "opt_corr": ppg_opt_eval["query_corr"],
                    "dual_raw": ppg_dual_raw,
                    "dual_corr": ppg_dual_eval["query_corr"],
                    "dual_hybrid_prob": ppg_dual_eval["hybrid_prob"],
                    "guided_prob": ppg_guided["prob"],
                    "guided_bp": ppg_guided["bp_pred"],
                    "selected_cls_prob": ppg_cls_prob,
                },
                cfg,
            )
            ecg_reg = apply_crisis_tail_fusion(
                best_crisis_tail_fusion_row,
                ecg_reg,
                ecg_cls_prob,
                {
                    "base_out": ecg_base,
                    "stability_out": ecg_stability,
                    "opt_raw": ecg_opt_raw,
                    "opt_corr": ecg_opt_eval["query_corr"],
                    "dual_raw": ecg_dual_raw,
                    "dual_corr": ecg_dual_eval["query_corr"],
                    "dual_hybrid_prob": ecg_dual_eval["hybrid_prob"],
                    "guided_prob": ecg_guided["prob"],
                    "guided_bp": ecg_guided["bp_pred"],
                    "selected_cls_prob": ecg_cls_prob,
                },
                cfg,
            )
        if bool(getattr(cfg, "ENABLE_SAFETY_CLASS_FUSION", False)):
            ppg_cls_prob = apply_safety_class_fusion_prob(
                best_safety_fusion_row,
                ppg_cls_prob,
                ppg_reg,
                cfg,
            )["prob"]
            ecg_cls_prob = apply_safety_class_fusion_prob(
                best_safety_fusion_row,
                ecg_cls_prob,
                ecg_reg,
                cfg,
            )["prob"]
        ppg_prefix = f"selected_ppg_missing_{str(missing_prob).replace('.', '_')}"
        ecg_prefix = f"selected_ecg_missing_{str(missing_prob).replace('.', '_')}"
        ppg_metrics = stage_script.risk_classification_metrics(np.asarray(ppg_reg["y_true_cls"], dtype=np.int64), ppg_cls_prob.argmax(axis=1).astype(np.int64), ppg_cls_prob, cfg, prefix=ppg_prefix)
        ecg_metrics = stage_script.risk_classification_metrics(np.asarray(ecg_reg["y_true_cls"], dtype=np.int64), ecg_cls_prob.argmax(axis=1).astype(np.int64), ecg_cls_prob, cfg, prefix=ecg_prefix)
        missing_rows.append(build_missing_row(missing_prob, ppg_reg, ppg_metrics, ecg_reg, ecg_metrics, ppg_prefix, ecg_prefix))

    runtime = {
        "optlong": bridge_script.measure_runtime_bridge(opt_ckpt, next(iter(loaders.test_query_loader)), cfg, "optlong_meta_stack"),
        "dualmax": bridge_script.measure_runtime_bridge(dual_ckpt, next(iter(loaders.test_query_loader)), cfg, "dualmax_meta_stack"),
        "feature_head_trainable_params": int(sum(p.numel() for p in head_model.parameters() if p.requires_grad)),
        "meta_classifier_members": int(len(test_meta_classifier["models"])),
        "meta_regressor_members": int(len(test_residual_bundle["models"])),
    }

    paper_metrics_meta = stage_script.build_paper_metrics(selected_reg["metrics_reg"])
    paper_metrics_stability = stage_script.build_paper_metrics(stability_selected["metrics_reg"])
    paper_metrics_optlong = stage_script.build_paper_metrics(opt_bundle["test_bundle"]["query_corr"]["metrics_reg"])
    paper_metrics_dualmax = stage_script.build_paper_metrics(dual_bundle["test_bundle"]["query_corr"]["metrics_reg"])

    classification_variant_inputs = [
        ("optlong_from_reg", opt_raw_test_metrics),
        ("dualmax_hybrid", dual_hybrid_metrics),
        ("guided_head", guided_test["metrics"]),
        ("stability_selected", stability_cls_metrics),
        ("meta_stack_raw", meta_raw_test_metrics),
    ]
    if bool(getattr(cfg, "ENABLE_SAFETY_CLASS_FUSION", False)):
        classification_variant_inputs.append(("selected_prefusion", selected_prefusion_cls_metrics))
    classification_variant_inputs.append(("selected_final", selected_cls_metrics))
    classification_variant_rows, classification_per_class_rows = consensus_script.build_classification_variant_rows(
        classification_variant_inputs,
        cfg,
    )
    regression_variant_rows = consensus_script.build_regression_variant_rows(
        [
            ("optlong_corrected", opt_bundle["test_bundle"]["query_corr"]),
            ("dualmax_corrected", dual_bundle["test_bundle"]["query_corr"]),
            ("stability_selected_base", selected_reg_base),
            ("stability_selected", stability_selected),
            ("meta_selected", selected_reg),
        ]
    )
    calibration_comparison_rows = consensus_script.build_calibration_comparison_rows(
        {
            "selected_final": calib_curve_rows,
            "stability_selected": stability_curve_rows,
            "meta_stack_raw": meta_curve_rows,
            "optlong_raw": opt_curve_rows,
            "dualmax_raw": dual_curve_rows,
        }
    )
    robustness_summary_rows = stability_script.build_robustness_summary_rows(noise_rows, missing_rows, [])
    bootstrap_rows = consensus_script.build_bootstrap_primary_rows(
        [
            ("optlong_corrected", build_classification_payload(opt_bundle["test_bundle"]["query_corr"], opt_bundle["test_bundle"]["query_corr"]["y_prob_cls_from_reg"])),
            ("dualmax_corrected", build_classification_payload(dual_bundle["test_bundle"]["query_corr"], dual_bundle["test_bundle"]["query_corr"]["y_prob_cls_from_reg"])),
            ("stability_selected", build_classification_payload(stability_selected, test_stability_prob)),
            ("meta_selected", build_classification_payload(selected_reg, selected_cls_prob)),
        ],
        cfg,
    )
    disagreement_rows = consensus_script.build_disagreement_rows(
        [
            ("optlong_reg", np.asarray(opt_bundle["test_bundle"]["query_corr"]["y_pred_cls_from_reg"], dtype=np.int64)),
            ("dualmax_hybrid", np.asarray(dual_bundle["test_bundle"]["hybrid_pred"], dtype=np.int64)),
            ("guided_head", np.asarray(guided_test["pred"], dtype=np.int64)),
            ("stability_selected", np.asarray(stability_cls_pred, dtype=np.int64)),
            ("meta_selected", np.asarray(selected_cls_pred, dtype=np.int64)),
        ]
    )
    uncertainty_decile_rows = build_uncertainty_decile_rows(selected_reg)
    bp_bin_rows = build_bp_bin_rows(selected_reg["y_true_reg"], selected_reg["y_pred_reg"])
    classwise_gain_rows = build_classwise_regression_gain_rows(selected_reg_base, selected_reg, cfg)
    classwise_gain_optlong_rows = build_classwise_regression_gain_rows(opt_bundle["test_bundle"]["query_corr"], selected_reg, cfg)
    classwise_gain_dual_rows = build_classwise_regression_gain_rows(dual_bundle["test_bundle"]["query_corr"], selected_reg, cfg)
    regression_expert_class_rows = build_regression_expert_class_rows(selected_reg, cfg)
    regression_expert_uncertainty_rows = build_regression_expert_uncertainty_rows(selected_reg)
    arbiter_class_rows: List[dict] = []
    arbiter_conf_rows: List[dict] = []
    if test_arbiter_bundle is not None:
        arbiter_weight = predict_classification_arbiter_weight(
            test_arbiter_bundle,
            blend_probabilities_with_policy(test_stability_prob, test_meta_clean_prob_raw, best_blend_row["weight_meta"], best_blend_row),
            test_stability_prob,
            best_arbiter_row,
            cfg,
        )
        arbiter_class_rows, arbiter_conf_rows = build_classification_arbiter_rows(
            arbiter_weight,
            selected_cls_prob_pre_safety,
            y_true_cls_test,
            cfg,
        )
    primary_gain_rows = classwise_gain_rows
    primary_gain_title = "Class-Specific Absolute Error Reduction Matrix"
    if not _gain_rows_have_signal(primary_gain_rows) and _gain_rows_have_signal(classwise_gain_optlong_rows):
        primary_gain_rows = classwise_gain_optlong_rows
        primary_gain_title = "Absolute Error Reduction Relative to the Longitudinal Reference"

    final_results = {
        "device": cfg.DEVICE,
        "protocol_id": cfg.PROTOCOL_ID,
        "protocol_rank": int(cfg.PROTOCOL_STRICTNESS_RANK),
        "split_protocol": cfg.SPLIT_PROTOCOL,
        "protocol_manifest": loaders.manifest,
        "checkpoint_optlong": str(opt_ckpt),
        "checkpoint_dualmax": str(dual_ckpt),
        "resume_feature_head": str(cfg.HEAD_RESUME_PATH),
        "selection_strategy": {
            "stability_regression_candidate": selected_reg_row["candidate"],
            "stability_classification_candidate": stability_cls_row["candidate"],
            "stability_tail_candidate": selected_tail_row["candidate"],
            "meta_blend_weight": float(best_blend_row["weight_meta"]),
            "meta_blend_score": float(best_blend_row["score"]),
            "classification_arbiter_candidate": best_arbiter_row["candidate"],
            "meta_regression_candidate": best_meta_reg_row["candidate"],
            "meta_regression_score": float(best_meta_reg_row["score"]),
            "clinical_high_risk_guard_candidate": best_risk_guard_row["candidate"],
            "clinical_high_risk_guard_score": float(best_risk_guard_row["score"]),
            "high_bias_calibration_candidate": best_high_bias_calibration_row["candidate"],
            "high_bias_calibration_score": float(best_high_bias_calibration_row.get("score", 0.0)),
            "crisis_tail_fusion_candidate": best_crisis_tail_fusion_row["candidate"],
            "crisis_tail_fusion_score": float(best_crisis_tail_fusion_row.get("score", 0.0)),
            "safety_class_fusion_candidate": best_safety_fusion_row["candidate"],
            "safety_class_fusion_score": float(best_safety_fusion_row.get("score", 0.0)),
            "top_regression_candidates": all_regression_rows[:10],
            "top_blend_candidates": blend_rows[:10],
            "top_classification_arbiter_candidates": classification_arbiter_rows[:10],
            "top_clinical_high_risk_guard_candidates": risk_guard_rows[:10],
            "top_high_bias_calibration_candidates": high_bias_calibration_rows[:10],
            "top_crisis_tail_fusion_candidates": crisis_tail_fusion_rows[:10],
            "top_safety_class_fusion_candidates": safety_class_fusion_rows[:10],
        },
        "runtime": runtime,
        "head_training_best_metrics": best_head_metrics,
        "classification_arbiter": {
            "enabled": bool(test_arbiter_bundle is not None),
            "candidate": best_arbiter_row["candidate"],
            "score": float(best_arbiter_row["score"]),
        },
        "clinical_high_risk_guard": {
            "enabled": bool(test_risk_guard_bundle is not None),
            "candidate": best_risk_guard_row["candidate"],
            "score": float(best_risk_guard_row["score"]),
            "clinical_under_penalty": float(best_risk_guard_row.get("clinical_under_penalty", 0.0)),
            "tail_bias_penalty": float(best_risk_guard_row.get("tail_bias_penalty", 0.0)),
        },
        "high_bias_calibration": {
            "enabled": bool(getattr(cfg, "ENABLE_HIGH_BIAS_CALIBRATOR", False)),
            "candidate": best_high_bias_calibration_row["candidate"],
            "score": float(best_high_bias_calibration_row.get("score", 0.0)),
            "shift_mean_sbp": float(best_high_bias_calibration_row.get("shift_mean_sbp", 0.0)),
            "shift_mean_dbp": float(best_high_bias_calibration_row.get("shift_mean_dbp", 0.0)),
        },
        "crisis_tail_fusion": {
            "enabled": bool(getattr(cfg, "ENABLE_CRISIS_TAIL_FUSION", False)),
            "candidate": best_crisis_tail_fusion_row["candidate"],
            "score": float(best_crisis_tail_fusion_row.get("score", 0.0)),
            "fusion_gate_mean": float(best_crisis_tail_fusion_row.get("fusion_gate_mean", 0.0)),
            "shift_mean_sbp": float(best_crisis_tail_fusion_row.get("shift_mean_sbp", 0.0)),
            "shift_mean_dbp": float(best_crisis_tail_fusion_row.get("shift_mean_dbp", 0.0)),
            "activation_rate": float(best_crisis_tail_fusion_row.get("activation_rate", 0.0)),
        },
        "safety_class_fusion": {
            "enabled": bool(getattr(cfg, "ENABLE_SAFETY_CLASS_FUSION", False)),
            "candidate": best_safety_fusion_row["candidate"],
            "score": float(best_safety_fusion_row.get("score", 0.0)),
            "mean_weight": float(best_safety_fusion_row.get("mean_weight", 0.0)),
            "p90_weight": float(best_safety_fusion_row.get("p90_weight", 0.0)),
        },
        "paper_metrics": paper_metrics_meta,
        "paper_metrics_meta_selected": paper_metrics_meta,
        "paper_metrics_stability_selected": paper_metrics_stability,
        "paper_metrics_optlong": paper_metrics_optlong,
        "paper_metrics_dualmax": paper_metrics_dualmax,
        "test_optlong_corrected": {**opt_bundle["test_bundle"]["query_corr"]["metrics_reg"], **opt_bundle["test_bundle"]["query_corr"]["metrics_cls_from_reg"], **opt_bundle["test_bundle"]["query_corr"]["uncertainty_metrics"]},
        "test_dualmax_corrected": {**dual_bundle["test_bundle"]["query_corr"]["metrics_reg"], **dual_bundle["test_bundle"]["query_corr"]["metrics_cls_from_reg"], **dual_bundle["test_bundle"]["query_corr"]["uncertainty_metrics"]},
        "test_guided_head": {**selected_reg_base["metrics_reg"], **guided_script.class_summary(guided_test["metrics"], "guided_test"), **guided_test["proxy_metrics"]},
        "test_stability_selected": {**stability_selected["metrics_reg"], **stability_compat, **stability_selected["uncertainty_metrics"]},
        "test_meta_stack_raw": {**selected_reg_base["metrics_reg"], **meta_raw_compat, **selected_reg_base["uncertainty_metrics"]},
        "test_selected_pre_safety_fusion": {
            **selected_reg["metrics_reg"],
            **selected_prefusion_compat,
            **selected_reg["uncertainty_metrics"],
        },
        "test_selected": {
            **selected_reg["metrics_reg"],
            **selected_compat,
            **selected_reg["uncertainty_metrics"],
            **{f"conformal_{k}": v for k, v in conformal_default.items()},
            "meta_gate_mean": float(selected_reg.get("meta_gate_mean", 0.0)),
            "meta_gate_p90": float(selected_reg.get("meta_gate_p90", 0.0)),
            "risk_guard_gate_mean": float(selected_reg.get("risk_guard_gate_mean", 0.0)),
            "risk_guard_high_active_rate": float(selected_reg.get("risk_guard_high_active_rate", 0.0)),
            "risk_guard_crisis_active_rate": float(selected_reg.get("risk_guard_crisis_active_rate", 0.0)),
            "high_bias_cal_shift_mean_sbp": float(selected_reg.get("high_bias_cal_shift_mean_sbp", 0.0)),
            "high_bias_cal_shift_mean_dbp": float(selected_reg.get("high_bias_cal_shift_mean_dbp", 0.0)),
            "crisis_tail_fusion_gate_mean": float(selected_reg.get("crisis_tail_fusion_gate_mean", 0.0)),
            "crisis_tail_fusion_shift_mean_sbp": float(selected_reg.get("crisis_tail_fusion_shift_mean_sbp", 0.0)),
            "crisis_tail_fusion_shift_mean_dbp": float(selected_reg.get("crisis_tail_fusion_shift_mean_dbp", 0.0)),
            "crisis_tail_fusion_activation_rate": float(selected_reg.get("crisis_tail_fusion_activation_rate", 0.0)),
        },
    }

    stage_script.save_json(out_root / "final_results.json", final_results)
    stage_script.save_json(out_root / "selected_strategy.json", final_results["selection_strategy"])
    stage_script.save_json(out_root / "paper_metrics.json", paper_metrics_meta)
    stage_script.save_json(out_root / "paper_metrics_meta_selected.json", paper_metrics_meta)
    stage_script.save_json(out_root / "paper_metrics_stability_selected.json", paper_metrics_stability)
    stage_script.save_json(out_root / "paper_metrics_optlong.json", paper_metrics_optlong)
    stage_script.save_json(out_root / "paper_metrics_dualmax.json", paper_metrics_dualmax)
    stage_script.save_json(out_root / "runtime_metrics.json", runtime)
    stage_script.save_json(
        out_root / "protocol_summary.json",
        {
            "protocol_id": cfg.PROTOCOL_ID,
            "protocol_name": cfg.PROTOCOL_NAME,
            "meta_blend_weight": float(best_blend_row["weight_meta"]),
            "classification_arbiter_candidate": best_arbiter_row["candidate"],
            "meta_regression_candidate": best_meta_reg_row["candidate"],
            "clinical_high_risk_guard_candidate": best_risk_guard_row["candidate"],
            "high_bias_calibration_candidate": best_high_bias_calibration_row["candidate"],
            "crisis_tail_fusion_candidate": best_crisis_tail_fusion_row["candidate"],
            "safety_class_fusion_candidate": best_safety_fusion_row["candidate"],
            "meta_regression_applied": str(best_meta_reg_row["candidate"]) != "selected_base",
            "selected_regression_candidate": best_meta_reg_row["candidate"],
            "stability_regression_candidate": selected_reg_row["candidate"],
            "selected_classification_candidate": (
                f"cls_arbiter_over_moe_w{float(best_blend_row['weight_meta']):.3f}_over_{stability_cls_row['candidate']}"
                if test_arbiter_bundle is not None
                else f"moe_blend_w{float(best_blend_row['weight_meta']):.3f}_over_{stability_cls_row['candidate']}"
            )
            + (
                f"+{best_safety_fusion_row['candidate']}"
                if bool(getattr(cfg, "ENABLE_SAFETY_CLASS_FUSION", False))
                else ""
            ),
            "selected_tail_correction_candidate": selected_tail_row["candidate"]
            + (
                f"+{best_high_bias_calibration_row['candidate']}"
                if bool(getattr(cfg, "ENABLE_HIGH_BIAS_CALIBRATOR", False))
                else ""
            )
            + (
                f"+{best_crisis_tail_fusion_row['candidate']}"
                if bool(getattr(cfg, "ENABLE_CRISIS_TAIL_FUSION", False))
                else ""
            ),
            "selected_crisis_tail_candidate": (
                best_crisis_tail_fusion_row["candidate"]
                if bool(getattr(cfg, "ENABLE_CRISIS_TAIL_FUSION", False))
                else "identity"
            ),
            "selected_mae_mean": float(selected_reg["metrics_reg"]["mae_mean"]),
            "stability_mae_mean": float(stability_selected["metrics_reg"]["mae_mean"]),
            "selected_prefusion_acc": float(selected_prefusion_cls_metrics.get("cls_acc_selected_prefusion", 0.0)),
            "selected_acc": float(selected_cls_metrics.get("cls_acc_selected_final", 0.0)),
            "stability_acc": float(stability_cls_metrics.get("cls_acc_stability_selected", 0.0)),
            "selected_prefusion_balanced_acc": float(selected_prefusion_cls_metrics.get("cls_balanced_acc_selected_prefusion", 0.0)),
            "selected_balanced_acc": float(selected_cls_metrics.get("cls_balanced_acc_selected_final", 0.0)),
            "stability_balanced_acc": float(stability_cls_metrics.get("cls_balanced_acc_stability_selected", 0.0)),
            "selected_prefusion_macro_f1": float(selected_prefusion_cls_metrics["cls_f1_macro_selected_prefusion"]),
            "selected_macro_f1": float(selected_cls_metrics["cls_f1_macro_selected_final"]),
            "stability_macro_f1": float(stability_cls_metrics["cls_f1_macro_stability_selected"]),
        },
    )

    bridge_script.save_rows_csv_flexible(out_root / "feature_head_epoch_log.csv", head_epoch_rows)
    bridge_script.save_rows_csv_flexible(out_root / "guided_temperature_search.csv", temperature_rows)
    bridge_script.save_rows_csv_flexible(out_root / "guided_decision_policy_search.csv", policy_rows)
    bridge_script.save_rows_csv_flexible(out_root / "classification_blend_search.csv", blend_rows)
    bridge_script.save_rows_csv_flexible(out_root / "classification_arbiter_search.csv", classification_arbiter_rows)
    bridge_script.save_rows_csv_flexible(out_root / "meta_regression_search.csv", all_regression_rows)
    bridge_script.save_rows_csv_flexible(out_root / "regression_router_search.csv", regression_router_rows)
    bridge_script.save_rows_csv_flexible(out_root / "risk_guard_search.csv", risk_guard_rows)
    bridge_script.save_rows_csv_flexible(out_root / "high_bias_calibration_search.csv", high_bias_calibration_rows)
    bridge_script.save_rows_csv_flexible(out_root / "crisis_tail_fusion_search.csv", crisis_tail_fusion_rows)
    bridge_script.save_rows_csv_flexible(out_root / "safety_class_fusion_search.csv", safety_class_fusion_rows)
    bridge_script.save_rows_csv_flexible(out_root / "regression_candidates_val.csv", regression_candidates)
    bridge_script.save_rows_csv_flexible(out_root / "classification_candidates_val.csv", bridge_script.sanitize_rows_for_csv(stability_cls_candidates))
    bridge_script.save_rows_csv_flexible(out_root / "tail_correction_candidates_val.csv", tail_candidates)
    stage_script.save_rows_csv(out_root / "conformal_sweep.csv", conformal_rows)
    stage_script.save_rows_csv(out_root / "conditional_coverage.csv", cond_rows)
    stage_script.save_rows_csv(out_root / "calibration_curve.csv", calib_curve_rows)
    stage_script.save_rows_csv(out_root / "noise_metrics.csv", noise_rows)
    stage_script.save_rows_csv(out_root / "missing_modality_metrics.csv", missing_rows)
    stage_script.save_rows_csv(out_root / "error_cdf.csv", error_cdf_rows)
    stage_script.save_rows_csv(out_root / "split_class_distribution.csv", split_dist_rows)
    stage_script.save_rows_csv(tbl_dir / "bp_range_metrics.csv", bp_range_rows)
    stage_script.save_rows_csv(tbl_dir / "bp_bin_error_summary.csv", bp_bin_rows)
    stage_script.save_rows_csv(tbl_dir / "subjectwise_error.csv", subject_rows)
    stage_script.save_rows_csv(tbl_dir / "subject_gain_optlong.csv", subject_gain_optlong_rows)
    stage_script.save_rows_csv(tbl_dir / "subject_gain_dualmax.csv", subject_gain_dual_rows)
    stage_script.save_rows_csv(tbl_dir / "classification_variant_summary.csv", classification_variant_rows)
    stage_script.save_rows_csv(tbl_dir / "classification_per_class_variants.csv", classification_per_class_rows)
    bridge_script.save_rows_csv_flexible(tbl_dir / "regression_variant_summary.csv", regression_variant_rows)
    stage_script.save_rows_csv(tbl_dir / "calibration_curve_comparison.csv", calibration_comparison_rows)
    stage_script.save_rows_csv(tbl_dir / "robustness_summary.csv", robustness_summary_rows)
    stage_script.save_rows_csv(tbl_dir / "bootstrap_primary_metrics.csv", bootstrap_rows)
    stage_script.save_rows_csv(tbl_dir / "classification_disagreement_matrix.csv", disagreement_rows)
    stage_script.save_rows_csv(tbl_dir / "uncertainty_deciles.csv", uncertainty_decile_rows)
    stage_script.save_rows_csv(tbl_dir / "classwise_regression_gain.csv", classwise_gain_rows)
    stage_script.save_rows_csv(tbl_dir / "classwise_regression_gain_vs_optlong.csv", classwise_gain_optlong_rows)
    stage_script.save_rows_csv(tbl_dir / "classwise_regression_gain_vs_dualmax.csv", classwise_gain_dual_rows)
    stage_script.save_rows_csv(tbl_dir / "classification_arbiter_class_profile.csv", arbiter_class_rows)
    stage_script.save_rows_csv(tbl_dir / "classification_arbiter_confidence_profile.csv", arbiter_conf_rows)
    stage_script.save_rows_csv(tbl_dir / "clinical_guard_bp_range_comparison.csv", clinical_guard_comparison_rows)
    stage_script.save_rows_csv(tbl_dir / "high_bias_calibration_bp_range_comparison.csv", high_bias_calibration_comparison_rows)
    stage_script.save_rows_csv(tbl_dir / "crisis_tail_fusion_bp_range_comparison.csv", crisis_tail_fusion_comparison_rows)
    stage_script.save_rows_csv(tbl_dir / "safety_class_transition_summary.csv", safety_transition_rows)
    stage_script.save_rows_csv(tbl_dir / "safety_class_fusion_class_profile.csv", safety_class_profile_rows)
    stage_script.save_rows_csv(tbl_dir / "safety_class_fusion_disagreement_profile.csv", safety_disagreement_profile_rows)
    stage_script.save_rows_csv(tbl_dir / "regression_expert_usage_by_class.csv", regression_expert_class_rows)
    stage_script.save_rows_csv(tbl_dir / "regression_expert_usage_by_uncertainty.csv", regression_expert_uncertainty_rows)
    stage_script.save_rows_csv(tbl_dir / "meta_classifier_feature_importance.csv", feature_importance_rows(test_meta_classifier["models"], test_meta_classifier["feature_names"]))
    stage_script.save_rows_csv(tbl_dir / "meta_regressor_feature_importance.csv", feature_importance_rows(test_residual_bundle["models"], test_residual_bundle["feature_names"]))
    if test_arbiter_bundle is not None:
        stage_script.save_rows_csv(tbl_dir / "classification_arbiter_feature_importance.csv", feature_importance_rows(test_arbiter_bundle["models"], test_arbiter_bundle["feature_names"]))
    if test_router_bundle is not None:
        stage_script.save_rows_csv(tbl_dir / "regression_router_feature_importance.csv", feature_importance_rows(test_router_bundle["models"], test_router_bundle["feature_names"]))
    if test_risk_guard_bundle is not None:
        stage_script.save_rows_csv(tbl_dir / "clinical_guard_feature_importance.csv", feature_importance_rows(test_risk_guard_bundle["models"], test_risk_guard_bundle["feature_names"]))

    summary_rows = [
        {"variant": "optlong_corrected", **opt_bundle["test_bundle"]["query_corr"]["metrics_reg"], **opt_bundle["test_bundle"]["query_corr"]["metrics_cls_from_reg"]},
        {"variant": "dualmax_corrected", **dual_bundle["test_bundle"]["query_corr"]["metrics_reg"], **dual_bundle["test_bundle"]["query_corr"]["metrics_cls_from_reg"]},
        {"variant": "stability_selected", **stability_selected["metrics_reg"], **stability_compat},
        {"variant": "meta_selected", **selected_reg["metrics_reg"], **selected_compat, "meta_gate_mean": float(selected_reg.get("meta_gate_mean", 0.0))},
    ]
    if bool(getattr(cfg, "ENABLE_SAFETY_CLASS_FUSION", False)):
        summary_rows.insert(
            3,
            {
                "variant": "meta_selected_prefusion_cls",
                **selected_reg["metrics_reg"],
                **selected_prefusion_compat,
                "meta_gate_mean": float(selected_reg.get("meta_gate_mean", 0.0)),
            },
        )
    if bool(test_risk_guard_bundle is not None):
        summary_rows.insert(
            3,
            {
                "variant": "meta_selected_pre_guard",
                **selected_reg_pre_guard["metrics_reg"],
                **selected_compat,
                "meta_gate_mean": float(selected_reg_pre_guard.get("meta_gate_mean", 0.0)),
            },
        )
    bridge_script.save_rows_csv_flexible(out_root / "meta_stack_summary.csv", bridge_script.sanitize_rows_for_csv(summary_rows))

    stage_script.save_regression_npz(art_dir / "test_outputs_regression_optlong_corrected.npz", opt_bundle["test_bundle"]["query_corr"])
    stage_script.save_regression_npz(art_dir / "test_outputs_regression_dualmax_corrected.npz", dual_bundle["test_bundle"]["query_corr"])
    stage_script.save_regression_npz(art_dir / "test_outputs_regression_stability_selected.npz", stability_selected)
    stage_script.save_regression_npz(art_dir / "test_outputs_regression_selected.npz", selected_reg)
    stage_script.save_regression_npz(art_dir / "test_outputs_regression_selected_base.npz", selected_reg_base)
    np.savez_compressed(
        art_dir / "test_outputs_classification_selected.npz",
        y_true_cls=np.asarray(selected_reg["y_true_cls"], dtype=np.int64),
        y_prob_cls=np.asarray(selected_cls_prob, dtype=np.float32),
        y_pred_cls=np.asarray(selected_cls_pred, dtype=np.int64),
        bp_proxy=np.asarray(guided_test["bp_pred"], dtype=np.float32),
        subject_ids=np.array(selected_reg["subject_ids"], dtype=object),
        seg_indices=np.array(selected_reg["seg_indices"], dtype=np.int64),
    )
    if bool(getattr(cfg, "ENABLE_SAFETY_CLASS_FUSION", False)):
        np.savez_compressed(
            art_dir / "test_outputs_classification_selected_prefusion.npz",
            y_true_cls=np.asarray(selected_reg["y_true_cls"], dtype=np.int64),
            y_prob_cls=np.asarray(selected_cls_prob_pre_safety, dtype=np.float32),
            y_pred_cls=np.asarray(selected_prefusion_cls_pred, dtype=np.int64),
            subject_ids=np.array(selected_reg["subject_ids"], dtype=object),
            seg_indices=np.array(selected_reg["seg_indices"], dtype=np.int64),
        )
    np.savez_compressed(
        art_dir / "test_outputs_classification_stability_selected.npz",
        y_true_cls=np.asarray(stability_selected["y_true_cls"], dtype=np.int64),
        y_prob_cls=np.asarray(test_stability_prob, dtype=np.float32),
        y_pred_cls=np.asarray(stability_cls_pred, dtype=np.int64),
        subject_ids=np.array(stability_selected["subject_ids"], dtype=object),
        seg_indices=np.array(stability_selected["seg_indices"], dtype=np.int64),
    )

    stage_script.plot_scatter_true_vs_pred(stability_selected["y_true_reg"], stability_selected["y_pred_reg"], fig_dir, filename="scatter_true_vs_pred_stability_selected.png")
    stage_script.plot_bland_altman(stability_selected["y_true_reg"], stability_selected["y_pred_reg"], fig_dir, filename="bland_altman_stability_selected.png")
    stage_script.plot_scatter_true_vs_pred(selected_reg["y_true_reg"], selected_reg["y_pred_reg"], fig_dir, filename="scatter_true_vs_pred.png")
    stage_script.plot_bland_altman(selected_reg["y_true_reg"], selected_reg["y_pred_reg"], fig_dir, filename="bland_altman.png")
    shared_plots.plot_true_pred_hexbin(
        selected_reg["y_true_reg"],
        selected_reg["y_pred_reg"],
        fig_dir,
        filename="prediction_density_hexbin.png",
    )
    stage_script.plot_confusion(
        y_true_cls_test,
        stability_cls_pred,
        list(cfg.CLASS_NAMES),
        fig_dir,
        "confusion_matrix_stability_selected.png",
        title="Confusion Matrix: Robust Operating Point",
    )
    stage_script.plot_confusion(
        y_true_cls_test,
        selected_cls_pred,
        list(cfg.CLASS_NAMES),
        fig_dir,
        "confusion_matrix_selected_final.png",
        title="Confusion Matrix: Final Operating Point",
    )
    stage_script.plot_confusion(
        y_true_cls_test,
        stability_cls_pred,
        list(cfg.CLASS_NAMES),
        fig_dir,
        "confusion_matrix_stability_selected_normalized.png",
        title="Row-Normalized Confusion Matrix: Robust Operating Point",
        normalize=True,
    )
    stage_script.plot_confusion(
        y_true_cls_test,
        selected_cls_pred,
        list(cfg.CLASS_NAMES),
        fig_dir,
        "confusion_matrix_selected_final_normalized.png",
        title="Row-Normalized Confusion Matrix: Final Operating Point",
        normalize=True,
    )
    stage_script.plot_roc_pr(y_true_cls_test, test_stability_prob, cfg, fig_dir, prefix="stability_selected")
    stage_script.plot_roc_pr(y_true_cls_test, selected_cls_prob, cfg, fig_dir, prefix="selected_final")
    stage_script.plot_calibration(calib_curve_rows, fig_dir)
    stage_script.plot_quality_conditional_coverage(cond_rows, fig_dir)
    stage_script.plot_sharpness_vs_coverage(conformal_rows, fig_dir)
    stage_script.plot_bp_range_bias(bp_range_rows, fig_dir)
    stage_script.plot_error_cdf(error_cdf_rows, fig_dir)
    stage_script.plot_noise_robustness(noise_rows, fig_dir)
    stage_script.plot_missing_modality_curve(missing_rows, fig_dir)
    stage_script.plot_split_class_distribution(split_dist_rows, fig_dir, cfg)
    if "uncertainty" in selected_reg:
        stage_script.plot_uncertainty_error_corr(
            selected_reg["uncertainty"],
            np.abs(np.asarray(selected_reg["y_pred_reg"]) - np.asarray(selected_reg["y_true_reg"])).mean(axis=1),
            fig_dir,
        )
    if "alpha" in selected_reg_base:
        stage_script.plot_router_heatmap(
            np.asarray(selected_reg_base["alpha"], dtype=np.float32),
            selected_reg_base["y_true_reg"],
            fig_dir,
            ["PPG", "ECG", "Joint", "Cross"],
        )
    stability_script.plot_head_resume_training(head_epoch_rows, fig_dir)
    stability_script.plot_subject_gain_named(subject_gain_optlong_rows, fig_dir, "subject_calibration_gain_optlong.png", "Opt-long")
    stability_script.plot_subject_gain_named(subject_gain_dual_rows, fig_dir, "subject_calibration_gain_dualmax.png", "Dualmax")
    consensus_script.plot_disagreement_heatmap(disagreement_rows, fig_dir)
    plot_bootstrap_ci(bootstrap_rows, fig_dir)
    plot_uncertainty_deciles(uncertainty_decile_rows, fig_dir)
    plot_residual_histograms(selected_reg_base, selected_reg, fig_dir)
    plot_subject_error_distribution(subject_rows, fig_dir, filename="subject_level_error_distribution.png")
    plot_uncertainty_boxplots(selected_reg, fig_dir, filename="uncertainty_decile_error_boxplot.png")
    plot_bp_range_heatmap(bp_bin_rows, fig_dir, filename="bp_range_error_heatmap.png")
    plot_variant_class_heatmap(classification_per_class_rows, fig_dir, filename="decision_system_class_profile_heatmap.png")
    plot_classwise_gain_heatmap(primary_gain_rows, fig_dir, title=primary_gain_title)
    plot_classwise_gain_heatmap(
        classwise_gain_rows,
        fig_dir,
        filename="classwise_gain_vs_selected_base_heatmap.png",
        title="Absolute Error Reduction Relative to the Calibrated Reference",
    )
    plot_classwise_gain_heatmap(
        classwise_gain_optlong_rows,
        fig_dir,
        filename="classwise_gain_vs_optlong_heatmap.png",
        title="Absolute Error Reduction Relative to the Longitudinal Reference",
    )
    plot_classwise_gain_heatmap(
        classwise_gain_dual_rows,
        fig_dir,
        filename="classwise_gain_vs_dualmax_heatmap.png",
        title="Absolute Error Reduction Relative to the Dual-Anchor Reference",
    )
    plot_regression_expert_class_heatmap(regression_expert_class_rows, fig_dir, cfg)
    plot_regression_expert_uncertainty(regression_expert_uncertainty_rows, fig_dir)
    plot_classification_arbiter_profile(arbiter_class_rows, arbiter_conf_rows, fig_dir)
    plot_classification_search_frontier(blend_rows, classification_arbiter_rows, fig_dir)
    plot_risk_guard_frontier(risk_guard_rows, fig_dir)
    plot_clinical_guard_bias_comparison(clinical_guard_comparison_rows, fig_dir)
    plot_high_bias_calibration_frontier(high_bias_calibration_rows, fig_dir)
    plot_clinical_guard_bias_comparison(
        high_bias_calibration_comparison_rows,
        fig_dir,
        filename="high_bias_calibration_bias_comparison.png",
    )
    plot_crisis_tail_fusion_frontier(crisis_tail_fusion_rows, fig_dir)
    plot_clinical_guard_bias_comparison(
        crisis_tail_fusion_comparison_rows,
        fig_dir,
        filename="crisis_tail_fusion_bias_comparison.png",
    )
    plot_safety_class_fusion_frontier(safety_class_fusion_rows, fig_dir)
    plot_safety_class_transition_heatmap(safety_transition_rows, fig_dir, cfg)
    plot_safety_class_fusion_profile(safety_class_profile_rows, safety_disagreement_profile_rows, fig_dir)

    print(f"Done. Results saved to: {out_root}")


if __name__ == "__main__":
    main()
