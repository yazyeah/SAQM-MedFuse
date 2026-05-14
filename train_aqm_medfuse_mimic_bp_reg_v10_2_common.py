from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from aqm_bp_shared_v10_2_multi_protocol import CFG, build_protocol_loaders
from aqm_bp_shared_v10_2_protocol import (
    ModelEMA,
    QMoERegressionNet,
    apply_asymmetric_modality_dropout,
    apply_training_quality_augmentation,
    build_bp_range_table,
    build_calibration_curve_table,
    build_class_weights,
    build_credibility_prior_torch,
    build_conditional_coverage_table,
    build_range_weights,
    build_router_target,
    build_subjectwise_error_table,
    collect_outputs_regression,
    conformal_from_outputs,
    ensure_out_dirs,
    heteroscedastic_loss,
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
    plot_training_curves_reg,
    plot_uncertainty_error_corr,
    reg_to_class_np,
    regression_metrics,
    regression_to_class_prob,
    regression_to_class_prob_torch,
    risk_classification_metrics,
    save_epoch_log,
    save_json,
    save_regression_npz,
    save_rows_csv,
    seed_everything,
    set_warmup_cosine_lr,
)


def maybe_apply_optuna_overrides(cfg: CFG) -> CFG:
    return cfg


def _torch_load_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resume_enabled(cfg: CFG) -> bool:
    return bool(getattr(cfg, "ENABLE_EPOCH_RESUME", False))


def _resume_state_path(out_root: Path, cfg: CFG) -> Path:
    configured = str(getattr(cfg, "RESUME_STATE_PATH", "") or "").strip()
    if configured:
        return Path(configured)
    return out_root / "latest_training_state.pt"


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _save_regression_resume_state(
    path: Path,
    cfg: CFG,
    epoch: int,
    model,
    ema,
    optimizer,
    epoch_logs: List[dict],
    best_score: float,
    best_epoch: int,
    best_state,
    patience: int,
) -> None:
    payload = {
        "protocol_id": str(getattr(cfg, "PROTOCOL_ID", "")),
        "output_name": str(getattr(cfg, "OUTPUT_NAME", "")),
        "epoch": int(epoch),
        "epochs_target": int(getattr(cfg, "EPOCHS", epoch)),
        "model_state": model.state_dict(),
        "ema_state": ema.module.state_dict() if ema is not None else None,
        "optimizer_state": optimizer.state_dict(),
        "epoch_logs": list(epoch_logs),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "best_state": best_state,
        "patience": int(patience),
    }
    _atomic_torch_save(payload, path)


def _load_regression_resume_state(path: Path, cfg: CFG, device) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = _torch_load_checkpoint(path, map_location=device)
    except Exception as exc:
        print(f"Resume checkpoint exists but could not be loaded, starting from warm-start weights: {path} ({exc})")
        return None
    if not isinstance(payload, dict):
        print(f"Resume checkpoint is not a state payload, ignoring: {path}")
        return None
    output_name = str(payload.get("output_name", ""))
    expected_output = str(getattr(cfg, "OUTPUT_NAME", ""))
    if output_name and expected_output and output_name != expected_output:
        print(f"Resume checkpoint belongs to {output_name}, expected {expected_output}; ignoring: {path}")
        return None
    return payload


def tail_underestimation_loss(pred_reg: torch.Tensor, target_reg: torch.Tensor, y_cls: torch.Tensor, cfg: CFG) -> torch.Tensor:
    delta = pred_reg - target_reg
    under = F.relu(-delta)
    over = F.relu(delta)

    class_weights = pred_reg.new_tensor(cfg.TAIL_CLASS_WEIGHTS, dtype=pred_reg.dtype)[y_cls]
    under_penalty = class_weights * (
        under[:, 0] / float(cfg.TAIL_SBP_SCALE) +
        under[:, 1] / float(cfg.TAIL_DBP_SCALE)
    )
    normal_over_penalty = (y_cls == 0).to(pred_reg.dtype) * float(cfg.NORMAL_OVERPRED_WEIGHT) * (
        over[:, 0] / float(cfg.TAIL_SBP_SCALE) +
        over[:, 1] / float(cfg.TAIL_DBP_SCALE)
    )
    return (under_penalty + normal_over_penalty).mean()


def _continuous_tail_ramp_torch(values: torch.Tensor, start: float, full: float) -> torch.Tensor:
    span = max(1.0e-6, float(full) - float(start))
    return torch.clamp((values - float(start)) / span, 0.0, 1.0)


def crisis_tail_underestimation_loss(pred_reg: torch.Tensor, target_reg: torch.Tensor, cfg: CFG) -> torch.Tensor:
    sbp_true = target_reg[:, 0]
    dbp_true = target_reg[:, 1]
    sbp_under = F.relu(target_reg[:, 0] - pred_reg[:, 0])
    dbp_under = F.relu(target_reg[:, 1] - pred_reg[:, 1])

    sbp_gate = _continuous_tail_ramp_torch(
        sbp_true,
        getattr(cfg, "CRISIS_SBP_PRE_THRESHOLD", 150.0),
        getattr(cfg, "CRISIS_SBP_THRESHOLD", 180.0),
    )
    dbp_gate = _continuous_tail_ramp_torch(
        dbp_true,
        getattr(cfg, "CRISIS_DBP_PRE_THRESHOLD", 95.0),
        getattr(cfg, "CRISIS_DBP_THRESHOLD", 120.0),
    )
    crisis_gate = torch.maximum(sbp_gate, dbp_gate)

    sbp_scale = max(1.0, float(getattr(cfg, "CRISIS_SBP_SCALE", 15.0)))
    dbp_scale = max(1.0, float(getattr(cfg, "CRISIS_DBP_SCALE", 10.0)))
    sbp_weight = float(getattr(cfg, "CRISIS_SBP_UNDER_WEIGHT", 1.0)) * sbp_gate + float(
        getattr(cfg, "CRISIS_TRUE_EXTRA_WEIGHT", 0.0)
    ) * crisis_gate
    dbp_weight = float(getattr(cfg, "CRISIS_DBP_UNDER_WEIGHT", 1.0)) * dbp_gate + 0.75 * float(
        getattr(cfg, "CRISIS_TRUE_EXTRA_WEIGHT", 0.0)
    ) * crisis_gate

    loss = sbp_weight * (sbp_under / sbp_scale) + dbp_weight * (dbp_under / dbp_scale)
    return loss.mean()


def safe_corr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def metrics_row_from_output(out: dict) -> Dict[str, float]:
    row = {
        **out["metrics_reg"],
        **out["metrics_cls_from_reg"],
        **out["metrics_cls_from_reg_hard"],
        **out["uncertainty_metrics"],
    }
    bp_range_rows = build_bp_range_table(out["y_true_reg"], out["y_pred_reg"])
    range_map = {str(item["bp_range"]): item for item in bp_range_rows}
    for bp_range in ("elevated", "high", "crisis"):
        range_row = range_map.get(bp_range, {})
        row[f"range_{bp_range}_n"] = int(range_row.get("n", 0))
        row[f"range_{bp_range}_bias_sbp"] = float(range_row.get("bias_sbp", 0.0))
        row[f"range_{bp_range}_bias_dbp"] = float(range_row.get("bias_dbp", 0.0))
        row[f"range_{bp_range}_mae_sbp"] = float(range_row.get("mae_sbp", 0.0))
        row[f"range_{bp_range}_mae_dbp"] = float(range_row.get("mae_dbp", 0.0))
    row.update(build_extreme_tail_bias_summary(out["y_true_reg"], out["y_pred_reg"]))
    return row


def build_extreme_tail_bias_summary(y_true_reg: np.ndarray, y_pred_reg: np.ndarray) -> Dict[str, float]:
    y_true_reg = np.asarray(y_true_reg, dtype=np.float32)
    y_pred_reg = np.asarray(y_pred_reg, dtype=np.float32)
    summary: Dict[str, float] = {}
    for quantile, name in ((0.90, "top10"), (0.95, "top5")):
        for dim, target_name in ((0, "sbp"), (1, "dbp")):
            threshold = float(np.quantile(y_true_reg[:, dim], quantile))
            mask = y_true_reg[:, dim] >= threshold
            if not np.any(mask):
                bias = 0.0
                mae = 0.0
                under_rate = 0.0
                n = 0
            else:
                residual = y_pred_reg[mask, dim] - y_true_reg[mask, dim]
                bias = float(np.mean(residual))
                mae = float(np.mean(np.abs(residual)))
                under_rate = float(np.mean(residual < 0.0))
                n = int(mask.sum())
            prefix = f"tail_{name}_{target_name}"
            summary[f"{prefix}_threshold"] = threshold
            summary[f"{prefix}_n"] = n
            summary[f"{prefix}_bias"] = bias
            summary[f"{prefix}_mae"] = mae
            summary[f"{prefix}_under_rate"] = under_rate
    return summary


def validation_score(row: Dict[str, float], cfg: CFG | None = None) -> float:
    score = float(
        row["mae_mean"] +
        2.10 * (1.0 - row["cls_f1_macro_from_reg"]) +
        0.65 * (1.0 - row["cls_balanced_acc_from_reg"]) +
        0.04 * abs(row["bias_sbp"]) +
        0.02 * abs(row["bias_dbp"])
    )
    if cfg is None:
        return score

    score += float(getattr(cfg, "VAL_SCORE_HIGH_BIAS_WEIGHT", 0.0)) * max(0.0, -float(row.get("range_high_bias_sbp", 0.0)))
    score += 0.60 * float(getattr(cfg, "VAL_SCORE_HIGH_BIAS_WEIGHT", 0.0)) * max(
        0.0, -float(row.get("range_high_bias_dbp", 0.0))
    )
    score += float(getattr(cfg, "VAL_SCORE_CRISIS_BIAS_WEIGHT", 0.0)) * max(
        0.0, -float(row.get("range_crisis_bias_sbp", 0.0))
    )
    score += 0.60 * float(getattr(cfg, "VAL_SCORE_CRISIS_BIAS_WEIGHT", 0.0)) * max(
        0.0, -float(row.get("range_crisis_bias_dbp", 0.0))
    )
    score += float(getattr(cfg, "VAL_SCORE_TAIL_TOP10_BIAS_WEIGHT", 0.0)) * max(
        0.0, -float(row.get("tail_top10_sbp_bias", 0.0))
    )
    score += float(getattr(cfg, "VAL_SCORE_TAIL_TOP5_BIAS_WEIGHT", 0.0)) * max(
        0.0, -float(row.get("tail_top5_sbp_bias", 0.0))
    )
    score += float(getattr(cfg, "VAL_SCORE_TAIL_TOP10_MAE_WEIGHT", 0.0)) * float(row.get("tail_top10_sbp_mae", 0.0))
    score += float(getattr(cfg, "VAL_SCORE_TAIL_TOP5_MAE_WEIGHT", 0.0)) * float(row.get("tail_top5_sbp_mae", 0.0))
    return score


def _fit_affine_1d(pred: np.ndarray, true: np.ndarray, cfg: CFG) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    true = np.asarray(true, dtype=np.float32).reshape(-1)
    if pred.size == 0:
        return 1.0, 0.0
    if cfg.SUBJECT_CALIBRATION_MODE != "affine" or pred.size < int(cfg.SUBJECT_AFFINE_MIN_SHOTS) or float(np.std(pred)) < 1e-6:
        return 1.0, float(np.mean(true - pred))
    x = pred - pred.mean()
    y = true - true.mean()
    denom = float(np.dot(x, x))
    if denom < 1e-6:
        return 1.0, float(np.mean(true - pred))
    slope = float(np.dot(x, y) / denom)
    slope = float(np.clip(slope, cfg.SUBJECT_AFFINE_SCALE_CLIP[0], cfg.SUBJECT_AFFINE_SCALE_CLIP[1]))
    intercept = float(true.mean() - slope * pred.mean())
    return slope, intercept


def fit_subject_calibration_state(calib_out: dict, cfg: CFG, n_shots: int | None = None) -> dict:
    y_true = np.asarray(calib_out["y_true_reg"], dtype=np.float32)
    y_pred = np.asarray(calib_out["y_pred_reg"], dtype=np.float32)
    subject_ids = np.asarray(calib_out["subject_ids"])
    seg_indices = np.asarray(calib_out["seg_indices"], dtype=np.int64)

    selected_rows: List[int] = []
    for sid in sorted(set(subject_ids.tolist())):
        idx = np.where(subject_ids == sid)[0]
        if idx.size == 0:
            continue
        idx = idx[np.argsort(seg_indices[idx])]
        if n_shots is not None and n_shots > 0:
            idx = idx[:min(int(n_shots), idx.size)]
        selected_rows.extend(idx.tolist())

    if not selected_rows:
        selected_rows = list(range(len(subject_ids)))

    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    global_scale = np.ones(2, dtype=np.float32)
    global_offset = np.zeros(2, dtype=np.float32)
    for dim in range(2):
        scale, offset = _fit_affine_1d(y_pred[selected_rows, dim], y_true[selected_rows, dim], cfg)
        global_scale[dim] = scale
        global_offset[dim] = offset

    subject_scale = {}
    subject_offset = {}
    for sid in sorted(set(subject_ids.tolist())):
        idx = np.where(subject_ids == sid)[0]
        if idx.size == 0:
            continue
        idx = idx[np.argsort(seg_indices[idx])]
        if n_shots is not None and n_shots > 0:
            idx = idx[:min(int(n_shots), idx.size)]
        subj_scale = np.ones(2, dtype=np.float32)
        subj_offset = np.zeros(2, dtype=np.float32)
        for dim in range(2):
            scale, offset = _fit_affine_1d(y_pred[idx, dim], y_true[idx, dim], cfg)
            subj_scale[dim] = scale
            subj_offset[dim] = offset
        shrink = float(idx.size) / float(idx.size + max(cfg.CALIBRATION_SHRINKAGE, 1e-6))
        subject_scale[str(sid)] = (shrink * subj_scale + (1.0 - shrink) * global_scale).astype(np.float32)
        subject_offset[str(sid)] = (shrink * subj_offset + (1.0 - shrink) * global_offset).astype(np.float32)

    return {
        "mode": cfg.SUBJECT_CALIBRATION_MODE,
        "global_scale": global_scale,
        "global_offset": global_offset,
        "subject_scale": subject_scale,
        "subject_offset": subject_offset,
        "n_subjects": int(len(subject_scale)),
        "shrinkage": float(cfg.CALIBRATION_SHRINKAGE),
        "n_shots": None if n_shots is None else int(n_shots),
        "n_rows_used": int(selected_rows.size),
    }


def apply_subject_calibration(out: dict, calib_state: dict, cfg: CFG) -> dict:
    y_true_reg = np.asarray(out["y_true_reg"], dtype=np.float32)
    y_true_cls = np.asarray(out["y_true_cls"], dtype=np.int64)
    y_pred_reg_raw = np.asarray(out["y_pred_reg"], dtype=np.float32)
    subject_ids = np.asarray(out["subject_ids"])
    global_scale = np.asarray(calib_state["global_scale"], dtype=np.float32)
    global_offset = np.asarray(calib_state["global_offset"], dtype=np.float32)
    subject_scale = calib_state["subject_scale"]
    subject_offset = calib_state["subject_offset"]

    scale = np.stack(
        [np.asarray(subject_scale.get(str(sid), global_scale), dtype=np.float32) for sid in subject_ids],
        axis=0,
    )
    offset = np.stack(
        [np.asarray(subject_offset.get(str(sid), global_offset), dtype=np.float32) for sid in subject_ids],
        axis=0,
    )
    y_pred_reg = y_pred_reg_raw * scale + offset
    uncertainty = np.asarray(out["uncertainty"], dtype=np.float32)
    y_prob_cls = regression_to_class_prob(y_pred_reg, uncertainty, cfg)
    y_pred_cls = y_prob_cls.argmax(axis=1).astype(np.int64)
    y_pred_cls_hard = reg_to_class_np(y_pred_reg)
    abs_err_mean = np.abs(y_pred_reg - y_true_reg).mean(axis=1)

    out_cal = dict(out)
    out_cal["y_pred_reg"] = y_pred_reg
    out_cal["y_prob_cls_from_reg"] = y_prob_cls
    out_cal["y_pred_cls_from_reg"] = y_pred_cls
    out_cal["y_pred_cls_from_reg_hard"] = y_pred_cls_hard
    out_cal["metrics_reg"] = regression_metrics(y_true_reg, y_pred_reg)
    out_cal["metrics_cls_from_reg"] = risk_classification_metrics(y_true_cls, y_pred_cls, y_prob_cls, cfg, prefix="from_reg")
    out_cal["metrics_cls_from_reg_hard"] = risk_classification_metrics(
        y_true_cls, y_pred_cls_hard, y_prob_cls, cfg, prefix="from_reg_hard"
    )
    out_cal["uncertainty_metrics"] = {
        "uncertainty_error_corr_pearson": safe_corr_np(uncertainty, abs_err_mean),
        "uncertainty_error_corr_spearman": safe_spearman_np(uncertainty, abs_err_mean),
    }
    out_cal["calibration_scale"] = scale
    out_cal["calibration_offset"] = offset
    return out_cal


def evaluate_personalized_validation(model, val_loader, calib_loader, cfg: CFG):
    val_raw = collect_outputs_regression(model, val_loader, cfg)
    raw_row = metrics_row_from_output(val_raw)
    if cfg.MODEL_SELECTION_USE_CALIBRATED_VAL and cfg.USE_SUBJECT_CALIBRATION and calib_loader is not None:
        calib_raw = collect_outputs_regression(model, calib_loader, cfg)
        calib_state = fit_subject_calibration_state(calib_raw, cfg)
        val_cal = apply_subject_calibration(val_raw, calib_state, cfg)
        cal_row = metrics_row_from_output(val_cal)
        chosen_row = cal_row
        score = (
            float(cfg.VAL_CAL_SCORE_BLEND) * validation_score(cal_row, cfg) +
            float(1.0 - cfg.VAL_CAL_SCORE_BLEND) * validation_score(raw_row, cfg)
        )
    else:
        calib_raw = None
        calib_state = None
        val_cal = val_raw
        cal_row = raw_row
        chosen_row = raw_row
        score = validation_score(raw_row, cfg)

    row = {
        **chosen_row,
        "raw_mae_mean": float(raw_row["mae_mean"]),
        "raw_mae_sbp": float(raw_row["mae_sbp"]),
        "raw_mae_dbp": float(raw_row["mae_dbp"]),
        "raw_bias_sbp": float(raw_row["bias_sbp"]),
        "raw_bias_dbp": float(raw_row["bias_dbp"]),
        "raw_cls_f1_macro_from_reg": float(raw_row["cls_f1_macro_from_reg"]),
        "raw_cls_balanced_acc_from_reg": float(raw_row["cls_balanced_acc_from_reg"]),
        "cal_mae_mean": float(cal_row["mae_mean"]),
        "cal_mae_sbp": float(cal_row["mae_sbp"]),
        "cal_mae_dbp": float(cal_row["mae_dbp"]),
        "cal_bias_sbp": float(cal_row["bias_sbp"]),
        "cal_bias_dbp": float(cal_row["bias_dbp"]),
        "cal_cls_f1_macro_from_reg": float(cal_row["cls_f1_macro_from_reg"]),
        "cal_cls_balanced_acc_from_reg": float(cal_row["cls_balanced_acc_from_reg"]),
        "val_score": float(score),
    }
    aux = {
        "val_raw": val_raw,
        "val_cal": val_cal,
        "calib_raw": calib_raw,
        "calib_state": calib_state,
    }
    return row, aux


def compute_regression_loss(model, batch, range_weights, class_weights, cfg: CFG, epoch: int):
    batch = apply_asymmetric_modality_dropout(batch, cfg, epoch=epoch)
    batch, q_targets = apply_training_quality_augmentation(batch, cfg, epoch=epoch)
    pred_reg, aux = model(
        batch["ppg"], batch["ecg"], batch["num"],
        batch["ppg_qfeat"], batch["ecg_qfeat"], batch["num_qfeat"], batch["avail"],
    )
    sample_w = range_weights[batch["y_cls"]]
    reg_l1 = F.smooth_l1_loss(pred_reg, batch["y_reg"], beta=4.0, reduction="none").mean(dim=1)
    true_map = (batch["y_reg"][:, 0] + 2.0 * batch["y_reg"][:, 1]) / 3.0
    pred_map = (pred_reg[:, 0] + 2.0 * pred_reg[:, 1]) / 3.0
    map_loss = F.smooth_l1_loss(pred_map, true_map, beta=4.0, reduction="none")
    nll_main = heteroscedastic_loss(aux["fused_mu"], aux["fused_logvar"], batch["y_reg"])
    reg_loss = (sample_w * (reg_l1 + cfg.LAMBDA_MAP * map_loss + cfg.LAMBDA_NLL * nll_main)).mean()

    q_loss = 0.5 * (
        F.mse_loss(aux["q_ppg"], q_targets["q_ppg_t"]) +
        F.mse_loss(aux["q_ecg"], q_targets["q_ecg_t"])
    )
    router_target = build_router_target(q_targets["q_ppg_t"], q_targets["q_ecg_t"], batch["avail"])
    router_loss = F.kl_div(torch.log(aux["alpha"] + 1e-8), router_target, reduction="batchmean")
    alpha_mean = aux["alpha"].mean(dim=0)
    route_bal_loss = ((alpha_mean - alpha_mean.new_full(alpha_mean.shape, 1.0 / alpha_mean.numel())) ** 2).sum()
    ppg_aux = heteroscedastic_loss(aux["ppg_mu"], aux["ppg_logvar"], batch["y_reg"]).mean()
    ecg_aux = heteroscedastic_loss(aux["ecg_mu"], aux["ecg_logvar"], batch["y_reg"]).mean()
    class_w = class_weights[batch["y_cls"]]
    prob_from_reg = regression_to_class_prob_torch(pred_reg, aux["uncertainty_proxy"], cfg)
    reg_cls_loss = (
        F.nll_loss(torch.log(prob_from_reg.clamp(min=1e-8)), batch["y_cls"], reduction="none") * class_w
    ).mean()
    branch_err = torch.stack(
        [
            (aux["ppg_mu"].detach() - batch["y_reg"]).abs().mean(dim=1),
            (aux["ecg_mu"].detach() - batch["y_reg"]).abs().mean(dim=1),
            (aux["fused_mu"].detach() - batch["y_reg"]).abs().mean(dim=1),
        ],
        dim=1,
    )
    cred_target = torch.softmax(-branch_err / max(cfg.CRED_TARGET_TEMP, 1e-3), dim=1)
    cred_prior = build_credibility_prior_torch(batch["avail"], cfg.FUSED_CREDIBILITY_PRIOR)
    both_avail = torch.minimum(batch["avail"][:, 0:1], batch["avail"][:, 1:2])
    cred_target = (1.0 - 0.35 * both_avail) * cred_target + (0.35 * both_avail) * cred_prior
    cred_target = cred_target / cred_target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    cred_loss = F.kl_div(torch.log(aux["credibility"] + 1e-8), cred_target, reduction="batchmean")
    tail_loss = tail_underestimation_loss(pred_reg, batch["y_reg"], batch["y_cls"], cfg)
    crisis_tail_loss = crisis_tail_underestimation_loss(pred_reg, batch["y_reg"], cfg)

    total = (
        reg_loss +
        cfg.LAMBDA_Q * q_loss +
        cfg.LAMBDA_ROUTER * router_loss +
        cfg.LAMBDA_BAL * route_bal_loss +
        cfg.LAMBDA_PPG_AUX * ppg_aux +
        cfg.LAMBDA_ECG_AUX * ecg_aux +
        cfg.LAMBDA_REG_CLS * reg_cls_loss +
        cfg.LAMBDA_CRED * cred_loss +
        cfg.LAMBDA_TAIL * tail_loss +
        float(getattr(cfg, "LAMBDA_CRISIS_TAIL", 0.0)) * crisis_tail_loss
    )
    logs = {
        "loss_reg": float(reg_loss.item()),
        "loss_nll": float(nll_main.mean().item()),
        "loss_q": float(q_loss.item()),
        "loss_router": float(router_loss.item()),
        "loss_bal": float(route_bal_loss.item()),
        "loss_ppg_aux": float(ppg_aux.item()),
        "loss_ecg_aux": float(ecg_aux.item()),
        "loss_reg_cls": float(reg_cls_loss.item()),
        "loss_cred": float(cred_loss.item()),
        "loss_tail": float(tail_loss.item()),
        "loss_crisis_tail": float(crisis_tail_loss.item()),
    }
    return total, logs


def bhs_grade(within_5: float, within_10: float, within_15: float) -> str:
    if within_5 >= 0.60 and within_10 >= 0.85 and within_15 >= 0.95:
        return "A"
    if within_5 >= 0.50 and within_10 >= 0.75 and within_15 >= 0.90:
        return "B"
    if within_5 >= 0.40 and within_10 >= 0.65 and within_15 >= 0.85:
        return "C"
    return "D"


def build_paper_metrics(metrics: Dict[str, float]) -> dict:
    return {
        "aami_like": {
            "sbp_mean_error": float(metrics["bias_sbp"]),
            "sbp_sd_error": float(metrics["sd_error_sbp"]),
            "dbp_mean_error": float(metrics["bias_dbp"]),
            "dbp_sd_error": float(metrics["sd_error_dbp"]),
        },
        "bhs_like": {
            "sbp_within_5": float(metrics["within_5mmhg_sbp"]),
            "sbp_within_10": float(metrics["within_10mmhg_sbp"]),
            "sbp_within_15": float(metrics["within_15mmhg_sbp"]),
            "sbp_grade": bhs_grade(metrics["within_5mmhg_sbp"], metrics["within_10mmhg_sbp"], metrics["within_15mmhg_sbp"]),
            "dbp_within_5": float(metrics["within_5mmhg_dbp"]),
            "dbp_within_10": float(metrics["within_10mmhg_dbp"]),
            "dbp_within_15": float(metrics["within_15mmhg_dbp"]),
            "dbp_grade": bhs_grade(metrics["within_5mmhg_dbp"], metrics["within_10mmhg_dbp"], metrics["within_15mmhg_dbp"]),
        },
    }


def build_error_cdf_rows(y_true_reg: np.ndarray, y_pred_reg: np.ndarray, max_threshold: int = 25) -> List[dict]:
    abs_err = np.abs(np.asarray(y_pred_reg) - np.asarray(y_true_reg))
    rows = []
    for thr in range(1, max_threshold + 1):
        rows.append({
            "threshold_mmhg": thr,
            "sbp_cdf": float(np.mean(abs_err[:, 0] <= thr)),
            "dbp_cdf": float(np.mean(abs_err[:, 1] <= thr)),
            "mean_cdf": float(np.mean(abs_err.mean(axis=1) <= thr)),
        })
    return rows


def build_split_distribution_rows(split_to_dataset: Dict[str, object], cfg: CFG) -> List[dict]:
    rows = []
    for split_name, ds in split_to_dataset.items():
        counts = np.bincount(np.asarray(ds.sample_classes, dtype=np.int64), minlength=cfg.N_CLASSES)
        total = int(counts.sum())
        for cls_idx, cls_name in enumerate(cfg.CLASS_NAMES):
            rows.append({
                "split": split_name,
                "class_idx": int(cls_idx),
                "class_name": cls_name,
                "count": int(counts[cls_idx]),
                "fraction": float(counts[cls_idx] / total) if total else 0.0,
            })
    return rows


def build_subject_gain_table(raw_out: dict, cal_out: dict) -> List[dict]:
    raw_rows = build_subjectwise_error_table(raw_out["y_true_reg"], raw_out["y_pred_reg"], raw_out["subject_ids"])
    cal_rows = build_subjectwise_error_table(cal_out["y_true_reg"], cal_out["y_pred_reg"], cal_out["subject_ids"])
    cal_map = {row["subject_id"]: row for row in cal_rows}
    rows = []
    for raw_row in raw_rows:
        sid = raw_row["subject_id"]
        cal_row = cal_map[sid]
        raw_mae_mean = 0.5 * (raw_row["mae_sbp"] + raw_row["mae_dbp"])
        cal_mae_mean = 0.5 * (cal_row["mae_sbp"] + cal_row["mae_dbp"])
        rows.append({
            "subject_id": sid,
            "n_segments": int(raw_row["n_segments"]),
            "raw_mae_sbp": float(raw_row["mae_sbp"]),
            "raw_mae_dbp": float(raw_row["mae_dbp"]),
            "cal_mae_sbp": float(cal_row["mae_sbp"]),
            "cal_mae_dbp": float(cal_row["mae_dbp"]),
            "raw_mae_mean": float(raw_mae_mean),
            "cal_mae_mean": float(cal_mae_mean),
            "mae_mean_gain": float(raw_mae_mean - cal_mae_mean),
        })
    rows.sort(key=lambda r: r["mae_mean_gain"], reverse=True)
    return rows


def build_few_shot_rows(calib_raw: dict, test_raw: dict, cfg: CFG) -> List[dict]:
    rows = []
    for shot in sorted(set(int(x) for x in cfg.FEW_SHOT_SWEEP)):
        if shot <= 0 or not cfg.USE_SUBJECT_CALIBRATION:
            out = test_raw
            meta = {"n_rows_used": 0}
        else:
            calib_state = fit_subject_calibration_state(calib_raw, cfg, n_shots=shot)
            out = apply_subject_calibration(test_raw, calib_state, cfg)
            meta = calib_state
        row = {
            "n_shots": int(shot),
            **out["metrics_reg"],
            **out["metrics_cls_from_reg"],
            **out["uncertainty_metrics"],
            "calibration_enabled": bool(shot > 0 and cfg.USE_SUBJECT_CALIBRATION),
            "n_rows_used": int(meta["n_rows_used"]),
        }
        rows.append(row)
    return rows


def plot_few_shot_curve(rows: Sequence[dict], fig_dir: Path):
    if not rows:
        return
    shots = [r["n_shots"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(shots, [r["mae_sbp"] for r in rows], marker="o", label="SBP MAE")
    axes[0].plot(shots, [r["mae_dbp"] for r in rows], marker="s", label="DBP MAE")
    axes[0].set_xlabel("Calibration shots / subject")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Few-Shot Personalized Calibration")
    axes[0].legend()
    axes[1].plot(shots, [r["cls_f1_macro_from_reg"] for r in rows], marker="o", label="Macro-F1")
    axes[1].plot(shots, [r["cls_balanced_acc_from_reg"] for r in rows], marker="s", label="Balanced Acc")
    axes[1].set_xlabel("Calibration shots / subject")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Risk Stratification vs Shots")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "few_shot_personalization_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_cdf(rows: Sequence[dict], fig_dir: Path):
    if not rows:
        return
    x = [r["threshold_mmhg"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, [r["sbp_cdf"] for r in rows], marker="o", markersize=3, label="SBP")
    ax.plot(x, [r["dbp_cdf"] for r in rows], marker="s", markersize=3, label="DBP")
    ax.set_xlabel("Absolute error threshold (mmHg)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Error CDF")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "error_cdf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bp_range_bias(rows: Sequence[dict], fig_dir: Path):
    if not rows:
        return
    labels = [r["bp_range"] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.17, [r["bias_sbp"] for r in rows], width=0.34, label="SBP bias")
    ax.bar(x + 0.17, [r["bias_dbp"] for r in rows], width=0.34, label="DBP bias")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bias (Pred - True, mmHg)")
    ax.set_title("Bias by BP Range")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "bp_range_bias.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_subject_calibration_gain(rows: Sequence[dict], fig_dir: Path):
    if not rows:
        return
    raw = np.asarray([r["raw_mae_mean"] for r in rows], dtype=np.float32)
    cal = np.asarray([r["cal_mae_mean"] for r in rows], dtype=np.float32)
    gain = np.asarray([r["mae_mean_gain"] for r in rows], dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(raw, cal, s=12, alpha=0.5)
    lo = float(min(raw.min(), cal.min()))
    hi = float(max(raw.max(), cal.max()))
    axes[0].plot([lo, hi], [lo, hi], linestyle="--")
    axes[0].set_xlabel("Raw subject MAE mean")
    axes[0].set_ylabel("Calibrated subject MAE mean")
    axes[0].set_title("Subject-Level Calibration Gain")
    axes[1].hist(gain, bins=30, alpha=0.85)
    axes[1].axvline(np.mean(gain), linestyle="--", color="black", label=f"mean={np.mean(gain):.2f}")
    axes[1].set_xlabel("MAE gain (raw - calibrated)")
    axes[1].set_ylabel("Subjects")
    axes[1].set_title("Gain Distribution")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "subject_calibration_gain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_split_class_distribution(rows: Sequence[dict], fig_dir: Path, cfg: CFG):
    if not rows:
        return
    split_names = []
    for row in rows:
        if row["split"] not in split_names:
            split_names.append(row["split"])
    class_names = list(cfg.CLASS_NAMES)
    frac_map = {(r["split"], r["class_name"]): r["fraction"] for r in rows}
    x = np.arange(len(split_names))
    bottom = np.zeros(len(split_names), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(9, 5))
    for class_name in class_names:
        vals = np.asarray([frac_map.get((split, class_name), 0.0) for split in split_names], dtype=np.float32)
        ax.bar(x, vals, bottom=bottom, label=class_name)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(split_names, rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction")
    ax.set_title("Split Class Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "split_class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_missing_modality_curve(rows: Sequence[dict], fig_dir: Path):
    if not rows:
        return
    x = [r["missing_prob"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x, [r["ppg_missing_mae_sbp"] for r in rows], marker="o", label="Drop PPG / SBP")
    axes[0].plot(x, [r["ppg_missing_mae_dbp"] for r in rows], marker="o", linestyle="--", label="Drop PPG / DBP")
    axes[0].plot(x, [r["ecg_missing_mae_sbp"] for r in rows], marker="s", label="Drop ECG / SBP")
    axes[0].plot(x, [r["ecg_missing_mae_dbp"] for r in rows], marker="s", linestyle="--", label="Drop ECG / DBP")
    axes[0].set_xlabel("Missing probability")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Missing Modality Robustness")
    axes[0].legend()
    axes[1].plot(x, [r["ppg_missing_cls_f1_macro_from_reg"] for r in rows], marker="o", label="Drop PPG")
    axes[1].plot(x, [r["ecg_missing_cls_f1_macro_from_reg"] for r in rows], marker="s", label="Drop ECG")
    axes[1].set_xlabel("Missing probability")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Missing Modality Risk Stratification")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "missing_modality_robustness_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves_reg_split(epoch_logs: Sequence[dict], fig_dir: Path):
    if not epoch_logs:
        return

    epochs = [int(row["epoch"]) for row in epoch_logs]

    def _line_plot(filename: str, title: str, ylabel: str, series: Sequence[tuple[str, Sequence[float]]]):
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, values in series:
            ax.plot(epochs, values, marker="o", markersize=3, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    _line_plot(
        "training_total_loss.png",
        "Training Total Loss",
        "Loss",
        [("Train loss", [float(row["train_loss"]) for row in epoch_logs])],
    )
    _line_plot(
        "training_main_loss_components.png",
        "Main Loss Components",
        "Loss",
        [
            ("Reg", [float(row["loss_reg"]) for row in epoch_logs]),
            ("NLL", [float(row["loss_nll"]) for row in epoch_logs]),
            ("Q", [float(row["loss_q"]) for row in epoch_logs]),
            ("Router", [float(row["loss_router"]) for row in epoch_logs]),
        ],
    )
    _line_plot(
        "training_auxiliary_grad.png",
        "Auxiliary / Grad",
        "Value",
        [
            ("Grad norm", [float(row["grad_norm"]) for row in epoch_logs]),
            ("Reg->Cls", [float(row["loss_reg_cls"]) for row in epoch_logs]),
            ("Cred", [float(row["loss_cred"]) for row in epoch_logs]),
            ("Tail", [float(row["loss_tail"]) for row in epoch_logs]),
        ]
        + (
            [("Crisis tail", [float(row.get("loss_crisis_tail", 0.0)) for row in epoch_logs])]
            if "loss_crisis_tail" in epoch_logs[0]
            else []
        ),
    )
    _line_plot(
        "validation_mae_curve.png",
        "Validation MAE",
        "MAE",
        [
            ("SBP MAE", [float(row["mae_sbp"]) for row in epoch_logs]),
            ("DBP MAE", [float(row["mae_dbp"]) for row in epoch_logs]),
            ("Mean MAE", [float(row["mae_mean"]) for row in epoch_logs]),
        ],
    )
    _line_plot(
        "validation_macro_f1_curve.png",
        "Validation Macro-F1 (from reg)",
        "Score",
        [
            ("Macro-F1", [float(row["cls_f1_macro_from_reg"]) for row in epoch_logs]),
            ("Balanced Acc", [float(row["cls_balanced_acc_from_reg"]) for row in epoch_logs]),
        ],
    )
    corr_series = [("Pearson", [float(row["uncertainty_error_corr_pearson"]) for row in epoch_logs])]
    if "uncertainty_error_corr_spearman" in epoch_logs[0]:
        corr_series.append(("Spearman", [float(row["uncertainty_error_corr_spearman"]) for row in epoch_logs]))
    _line_plot(
        "uncertainty_error_correlation_curve.png",
        "Uncertainty-Error Correlation",
        "Correlation",
        corr_series,
    )


def _finalize_regression_outputs(
    model,
    cfg: CFG,
    loaders,
    out_root: Path,
    fig_dir: Path,
    art_dir: Path,
    tbl_dir: Path,
    ds_train,
    epoch_logs: Sequence[dict],
):
    test_loader = loaders.test_query_loader
    test_calib_loader = loaders.test_calib_loader
    split_dist_rows = build_split_distribution_rows(loaders.split_datasets, cfg)

    calib_full_raw = collect_outputs_regression(model, test_calib_loader, cfg)
    test_full_raw = collect_outputs_regression(model, test_loader, cfg)
    test_drop_ppg_raw = collect_outputs_regression(model, test_loader, cfg, drop_modality="ppg", missing_prob=1.0)
    test_drop_ecg_raw = collect_outputs_regression(model, test_loader, cfg, drop_modality="ecg", missing_prob=1.0)

    if cfg.USE_SUBJECT_CALIBRATION:
        calib_state = fit_subject_calibration_state(calib_full_raw, cfg)
        calib_full = apply_subject_calibration(calib_full_raw, calib_state, cfg)
        test_full = apply_subject_calibration(test_full_raw, calib_state, cfg)
        test_drop_ppg = apply_subject_calibration(test_drop_ppg_raw, calib_state, cfg)
        test_drop_ecg = apply_subject_calibration(test_drop_ecg_raw, calib_state, cfg)
    else:
        calib_state = {
            "mode": "disabled",
            "global_scale": np.ones(2, dtype=np.float32),
            "global_offset": np.zeros(2, dtype=np.float32),
            "subject_scale": {},
            "subject_offset": {},
            "n_subjects": 0,
            "shrinkage": 0.0,
            "n_shots": 0,
            "n_rows_used": 0,
        }
        calib_full = calib_full_raw
        test_full = test_full_raw
        test_drop_ppg = test_drop_ppg_raw
        test_drop_ecg = test_drop_ecg_raw

    low, high, conformal_default = conformal_from_outputs(calib_full, test_full, alpha=cfg.CONFORMAL_ALPHA)
    conformal_rows = []
    for alpha in cfg.CONFORMAL_ALPHAS:
        _, _, met = conformal_from_outputs(calib_full, test_full, alpha=alpha)
        conformal_rows.append({"alpha": alpha, **met})

    noise_rows = []
    for noise_std in cfg.NOISE_STDS:
        out = collect_outputs_regression(model, test_loader, cfg, noise_std=noise_std)
        if cfg.USE_SUBJECT_CALIBRATION:
            out = apply_subject_calibration(out, calib_state, cfg)
        noise_rows.append({
            "noise_std": noise_std,
            **out["metrics_reg"],
            **out["metrics_cls_from_reg"],
        })

    missing_rows = []
    for missing_prob in cfg.MISSING_PROBS:
        out_ppg = collect_outputs_regression(model, test_loader, cfg, drop_modality="ppg", missing_prob=missing_prob)
        out_ecg = collect_outputs_regression(model, test_loader, cfg, drop_modality="ecg", missing_prob=missing_prob)
        if cfg.USE_SUBJECT_CALIBRATION:
            out_ppg = apply_subject_calibration(out_ppg, calib_state, cfg)
            out_ecg = apply_subject_calibration(out_ecg, calib_state, cfg)
        missing_rows.append({
            "missing_prob": missing_prob,
            "ppg_missing_mae_sbp": out_ppg["metrics_reg"]["mae_sbp"],
            "ppg_missing_mae_dbp": out_ppg["metrics_reg"]["mae_dbp"],
            "ppg_missing_cls_f1_macro_from_reg": out_ppg["metrics_cls_from_reg"]["cls_f1_macro_from_reg"],
            "ecg_missing_mae_sbp": out_ecg["metrics_reg"]["mae_sbp"],
            "ecg_missing_mae_dbp": out_ecg["metrics_reg"]["mae_dbp"],
            "ecg_missing_cls_f1_macro_from_reg": out_ecg["metrics_cls_from_reg"]["cls_f1_macro_from_reg"],
        })

    cond_rows = build_conditional_coverage_table(test_full["y_true_reg"], low, high, test_full["quality"], cfg)
    calib_curve_rows = build_calibration_curve_table(test_full["y_true_cls"], test_full["y_prob_cls_from_reg"], n_bins=cfg.ECE_BINS)
    bp_range_rows_raw = build_bp_range_table(test_full_raw["y_true_reg"], test_full_raw["y_pred_reg"])
    bp_range_rows = build_bp_range_table(test_full["y_true_reg"], test_full["y_pred_reg"])
    subject_rows_raw = build_subjectwise_error_table(test_full_raw["y_true_reg"], test_full_raw["y_pred_reg"], test_full_raw["subject_ids"])
    subject_rows = build_subjectwise_error_table(test_full["y_true_reg"], test_full["y_pred_reg"], test_full["subject_ids"])
    subject_gain_rows = build_subject_gain_table(test_full_raw, test_full)
    error_cdf_rows = build_error_cdf_rows(test_full["y_true_reg"], test_full["y_pred_reg"])
    few_shot_rows = build_few_shot_rows(calib_full_raw, test_full_raw, cfg)

    raw_metrics = metrics_row_from_output(test_full_raw)
    cal_metrics = metrics_row_from_output(test_full)
    paper_metrics_raw = build_paper_metrics(raw_metrics)
    paper_metrics = build_paper_metrics(cal_metrics)

    runtime = measure_runtime(model, next(iter(test_loader)), cfg)
    final_results = {
        "device": cfg.DEVICE,
        "protocol_id": cfg.PROTOCOL_ID,
        "protocol_rank": int(cfg.PROTOCOL_STRICTNESS_RANK),
        "split_protocol": cfg.SPLIT_PROTOCOL,
        "protocol_manifest": loaders.manifest,
        "subject_calibration": {
            "enabled": bool(cfg.USE_SUBJECT_CALIBRATION),
            "mode": calib_state["mode"],
            "shrinkage": float(calib_state["shrinkage"]),
            "n_subjects": int(calib_state["n_subjects"]),
            "global_scale": np.asarray(calib_state["global_scale"], dtype=np.float32).tolist(),
            "global_offset": np.asarray(calib_state["global_offset"], dtype=np.float32).tolist(),
        },
        "few_shot_protocol": {
            "shots": [int(r["n_shots"]) for r in few_shot_rows],
            "note": (
                "0-shot is raw inference; k-shot uses the first k support segments per subject. "
                "Calibration applies subject-level bias or affine correction depending on the protocol."
            ),
        },
        "paper_metrics_raw": paper_metrics_raw,
        "paper_metrics": paper_metrics,
        "class_counts_train": ds_train.class_counts,
        "class_weights_train": build_class_weights(ds_train, cfg, cfg.DEVICE).detach().cpu().tolist(),
        "avg_credibility_test": test_full_raw["credibility"].mean(axis=0).tolist(),
        "avg_router_test": test_full_raw["alpha"].mean(axis=0).tolist(),
        "avg_quality_test": float(test_full_raw["quality"].mean()),
        "runtime": runtime,
        "test_full_raw": {
            **raw_metrics,
        },
        "test_full": {
            **cal_metrics,
            **{f"conformal_{k}": v for k, v in conformal_default.items()},
        },
        "test_drop_ppg": {
            **test_drop_ppg["metrics_reg"],
            **test_drop_ppg["metrics_cls_from_reg"],
            **test_drop_ppg["metrics_cls_from_reg_hard"],
        },
        "test_drop_ecg": {
            **test_drop_ecg["metrics_reg"],
            **test_drop_ecg["metrics_cls_from_reg"],
            **test_drop_ecg["metrics_cls_from_reg_hard"],
        },
    }
    save_json(out_root / "final_results.json", final_results)
    save_json(out_root / "runtime_metrics.json", runtime)
    save_json(out_root / "paper_metrics_raw.json", paper_metrics_raw)
    save_json(out_root / "paper_metrics.json", paper_metrics)
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
    save_regression_npz(art_dir / "test_outputs_regression_raw.npz", test_full_raw)
    save_regression_npz(art_dir / "test_outputs_regression.npz", test_full)

    if epoch_logs:
        if bool(getattr(cfg, "PLOT_COMPOSITE_TRAINING_CURVES", True)):
            plot_training_curves_reg(epoch_logs, fig_dir)
        if bool(getattr(cfg, "SAVE_SPLIT_TRAINING_CURVES", False)):
            plot_training_curves_reg_split(epoch_logs, fig_dir)
    plot_scatter_true_vs_pred(test_full["y_true_reg"], test_full["y_pred_reg"], fig_dir, filename="scatter_true_vs_pred.png")
    plot_bland_altman(test_full["y_true_reg"], test_full["y_pred_reg"], fig_dir, filename="bland_altman.png")
    plot_confusion(test_full["y_true_cls"], test_full["y_pred_cls_from_reg"], list(cfg.CLASS_NAMES), fig_dir, "confusion_matrix_reg_to_class.png")
    plot_roc_pr(test_full["y_true_cls"], test_full["y_prob_cls_from_reg"], cfg, fig_dir, prefix="reg_to_class")
    plot_noise_robustness(noise_rows, fig_dir)
    plot_missing_modality_curve(missing_rows, fig_dir)
    plot_router_heatmap(test_full["alpha"], test_full["y_true_reg"], fig_dir, ["PPG", "ECG", "Joint", "Cross"])
    plot_quality_conditional_coverage(cond_rows, fig_dir)
    plot_sharpness_vs_coverage(conformal_rows, fig_dir)
    abs_err_mean = np.abs(test_full["y_pred_reg"] - test_full["y_true_reg"]).mean(axis=1)
    plot_uncertainty_error_corr(test_full["uncertainty"], abs_err_mean, fig_dir)
    plot_calibration(calib_curve_rows, fig_dir, filename="calibration_curve.png")
    plot_error_cdf(error_cdf_rows, fig_dir)
    plot_bp_range_bias(bp_range_rows, fig_dir)
    plot_few_shot_curve(few_shot_rows, fig_dir)
    plot_subject_calibration_gain(subject_gain_rows, fig_dir)
    plot_split_class_distribution(split_dist_rows, fig_dir, cfg)


def evaluate_saved_regression_checkpoint(cfg: CFG, checkpoint_path: str | None = None):
    cfg = maybe_apply_optuna_overrides(cfg)
    seed_everything(cfg.SEED)
    out_root, fig_dir, art_dir, tbl_dir = ensure_out_dirs(cfg)
    loaders = build_protocol_loaders(cfg, task="regression")
    save_json(out_root / "protocol_manifest.json", loaders.manifest)
    print(f"Using device: {cfg.DEVICE}")
    print(f"Protocol rank: {cfg.PROTOCOL_STRICTNESS_RANK}")
    print(f"Split protocol: {cfg.SPLIT_PROTOCOL}")
    print(f"Protocol name: {cfg.PROTOCOL_NAME}")

    ds_train = loaders.ds_train
    model = QMoERegressionNet(cfg).to(cfg.DEVICE)
    candidates = []
    if checkpoint_path:
        candidates.append(Path(checkpoint_path))
    candidates.append(out_root / "best_model.pt")
    if getattr(cfg, "INIT_CKPT_PATH", ""):
        candidates.append(Path(cfg.INIT_CKPT_PATH))
    ckpt_path = next((path for path in candidates if path.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found for evaluation-only run.")
    state = torch.load(ckpt_path, map_location=cfg.DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint for evaluation: {ckpt_path}")
    _finalize_regression_outputs(model, cfg, loaders, out_root, fig_dir, art_dir, tbl_dir, ds_train, epoch_logs=[])
    print(f"Done. Results saved to: {out_root}")


def run_regression_experiment(cfg: CFG):
    cfg = maybe_apply_optuna_overrides(cfg)
    seed_everything(cfg.SEED)
    out_root, fig_dir, art_dir, tbl_dir = ensure_out_dirs(cfg)
    loaders = build_protocol_loaders(cfg, task="regression")
    save_json(out_root / "protocol_manifest.json", loaders.manifest)
    print(f"Using device: {cfg.DEVICE}")
    print(f"Protocol rank: {cfg.PROTOCOL_STRICTNESS_RANK}")
    print(f"Split protocol: {cfg.SPLIT_PROTOCOL}")
    print(f"Protocol name: {cfg.PROTOCOL_NAME}")

    ds_train = loaders.ds_train
    train_loader = loaders.train_loader
    val_loader = loaders.val_query_loader
    val_calib_loader = loaders.val_calib_loader
    test_loader = loaders.test_query_loader
    test_calib_loader = loaders.test_calib_loader
    split_dist_rows = build_split_distribution_rows(loaders.split_datasets, cfg)
    range_weights = build_range_weights(ds_train, cfg, cfg.DEVICE)
    class_weights = build_class_weights(ds_train, cfg, cfg.DEVICE)

    model = QMoERegressionNet(cfg).to(cfg.DEVICE)
    if getattr(cfg, "INIT_CKPT_PATH", ""):
        init_path = Path(cfg.INIT_CKPT_PATH)
        if init_path.exists():
            state = torch.load(init_path, map_location=cfg.DEVICE)
            model.load_state_dict(state, strict=False)
            print(f"Warm-started model from: {init_path}")
        else:
            print(f"Warm-start checkpoint not found, training from scratch: {init_path}")
    use_ema = bool(getattr(cfg, "USE_EMA", True))
    ema = ModelEMA(model, decay=cfg.EMA_DECAY) if use_ema else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    epoch_logs: List[dict] = []
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    start_epoch = 1
    resume_path = _resume_state_path(out_root, cfg)
    if _resume_enabled(cfg):
        resume_state = _load_regression_resume_state(resume_path, cfg, cfg.DEVICE)
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
                "loss_reg",
                "loss_nll",
                "loss_q",
                "loss_router",
                "loss_bal",
                "loss_ppg_aux",
                "loss_ecg_aux",
                "loss_reg_cls",
                "loss_cred",
                "loss_tail",
                "loss_crisis_tail",
            ]
        }
        for batch in train_loader:
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
            for k, v in logs.items():
                loss_buckets.setdefault(k, []).append(v)

        eval_model = ema.module if ema is not None else model
        val_row, _ = evaluate_personalized_validation(eval_model, val_loader, val_calib_loader, cfg)
        row = {
            "epoch": epoch,
            "lr": float(lr),
            "train_loss": float(np.mean(train_loss_total)),
            "grad_norm": float(np.mean(train_grad_norms)),
            **{k: (float(np.mean(v)) if v else float("nan")) for k, v in loss_buckets.items()},
            **val_row,
        }
        epoch_logs.append(row)
        stage_name = str(getattr(cfg, "TRAIN_STAGE_NAME", "Backbone"))
        print(
            f"[{stage_name} Epoch {epoch:03d}] train_loss={row['train_loss']:.4f} | "
            f"chosen_mae_sbp={row['mae_sbp']:.3f} | chosen_mae_dbp={row['mae_dbp']:.3f} | "
            f"backbone_f1_from_reg(chosen)={row['cls_f1_macro_from_reg']:.3f} | "
            f"backbone_f1_from_reg(raw)={row.get('raw_cls_f1_macro_from_reg', row['cls_f1_macro_from_reg']):.3f} | "
            f"backbone_f1_from_reg(cal)={row.get('cal_cls_f1_macro_from_reg', row['cls_f1_macro_from_reg']):.3f} | "
            f"val_score(lower=better)={row['val_score']:.3f}"
        )
        if row["val_score"] < best_score:
            best_score = row["val_score"]
            best_epoch = int(epoch)
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
            raise RuntimeError("No best checkpoint was recorded.")
    model.load_state_dict(best_state)
    out_root.mkdir(parents=True, exist_ok=True)
    save_epoch_log(out_root / "epoch_log.csv", epoch_logs)
    _finalize_regression_outputs(model, cfg, loaders, out_root, fig_dir, art_dir, tbl_dir, ds_train, epoch_logs)

    print(f"Done. Results saved to: {out_root}")
