from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

import aqm_bp_shared_v9 as shared_v9
import train_aqm_medfuse_mimic_bp_reg_v10_13_subjectdisjoint_piso_uncertainty_moe_tailaware_protocol as v13


FINAL_OUTPUT_NAME = "mimic_bp_reg_v10_15_subjectdisjoint_piso_uncertainty_moe_exhaustive_vectorfast_proto"
FINAL_PROTOCOL_ID = "v10.15_subjectdisjoint_piso_uncertainty_moe_exhaustive_vectorfast"

# Reuse the already-trained v10.13 full-train backbones.
OPTLONG_FULLTRAIN_OUTPUT = "mimic_bp_reg_v10_13_opt_long_fulltrain_proto"
DUALMAX_FULLTRAIN_OUTPUT = "mimic_bp_reg_v10_13_optlong_stageaware_dualmax_fulltrain_proto"

_ORIG_V13_BUILD_NEXTGEN_CFG = v13.build_nextgen_cfg
_ORIG_V13_GENERATE_EXTRA_OUTPUTS = v13.generate_extra_outputs
_ORIG_V13_TAILAWARE_HIGH_BIAS_SEARCH = v13.tailaware_search_high_bias_calibration_candidates
_ORIG_V13_PREV_CRISIS_SEARCH = v13.prev_script.search_crisis_tail_debias_candidates


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _output_dir() -> Path:
    return _project_root() / "outputs" / FINAL_OUTPUT_NAME


def _artifacts_dir() -> Path:
    path = _output_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tables_dir() -> Path:
    path = _output_dir() / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _figures_dir() -> Path:
    path = _output_dir() / "figures"
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


def _to_float_array(x: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _normalize_prob(prob: np.ndarray) -> np.ndarray:
    return v13.base_script.bridge_script.normalize_prob(np.asarray(prob, dtype=np.float32))


def _signal_ramp(signal: np.ndarray, threshold: float, gamma: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    scaled = np.clip((signal - float(threshold)) / max(1.0e-6, 1.0 - float(threshold)), 0.0, 1.0)
    return np.power(scaled, float(gamma), dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))).astype(np.float32)


def _range_masks(y_true: np.ndarray) -> dict[str, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float32)
    sbp = y_true[:, 0]
    dbp = y_true[:, 1]
    return {
        "normal": (sbp < 120.0) & (dbp < 80.0),
        "elevated": ((sbp >= 120.0) | (dbp >= 80.0)) & ((sbp < 140.0) & (dbp < 90.0)) & ((sbp < 180.0) & (dbp < 120.0)),
        "high": ((sbp >= 140.0) | (dbp >= 90.0)) & ((sbp < 180.0) & (dbp < 120.0)),
        "crisis": (sbp >= 180.0) | (dbp >= 120.0),
    }


def _masked_mean_2d(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros(values.shape[0], dtype=np.float32)
    return values[:, mask].mean(axis=1).astype(np.float32)


def _batch_regression_summary(y_true: np.ndarray, y_pred_batch: np.ndarray) -> dict[str, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred_batch = np.asarray(y_pred_batch, dtype=np.float32)
    err = y_pred_batch - y_true[None, :, :]
    abs_err = np.abs(err)
    mae_sbp = abs_err[:, :, 0].mean(axis=1).astype(np.float32)
    mae_dbp = abs_err[:, :, 1].mean(axis=1).astype(np.float32)
    return {
        "mae_sbp": mae_sbp,
        "mae_dbp": mae_dbp,
        "mae_mean": (0.5 * (mae_sbp + mae_dbp)).astype(np.float32),
        "bias_sbp": err[:, :, 0].mean(axis=1).astype(np.float32),
        "bias_dbp": err[:, :, 1].mean(axis=1).astype(np.float32),
        "err_sbp": err[:, :, 0].astype(np.float32),
        "err_dbp": err[:, :, 1].astype(np.float32),
    }


def _batch_bp_bias_summary(
    err_sbp: np.ndarray,
    err_dbp: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for range_name, mask in masks.items():
        out[f"{range_name}_bias_sbp"] = _masked_mean_2d(err_sbp, mask)
        out[f"{range_name}_bias_dbp"] = _masked_mean_2d(err_dbp, mask)
    return out


def _batch_tail_bias_penalty(range_bias: dict[str, np.ndarray]) -> np.ndarray:
    penalty = np.zeros_like(next(iter(range_bias.values())), dtype=np.float32)
    penalty += 0.25 * np.maximum(0.0, range_bias.get("normal_bias_sbp", penalty))
    penalty += 0.12 * np.maximum(0.0, range_bias.get("normal_bias_dbp", penalty))
    penalty += 0.40 * np.maximum(0.0, -range_bias.get("elevated_bias_sbp", penalty))
    penalty += 0.20 * np.maximum(0.0, -range_bias.get("elevated_bias_dbp", penalty))
    penalty += 0.85 * np.maximum(0.0, -range_bias.get("high_bias_sbp", penalty))
    penalty += 0.45 * np.maximum(0.0, -range_bias.get("high_bias_dbp", penalty))
    penalty += 1.10 * np.maximum(0.0, -range_bias.get("crisis_bias_sbp", penalty))
    penalty += 0.65 * np.maximum(0.0, -range_bias.get("crisis_bias_dbp", penalty))
    return penalty.astype(np.float32)


def _batch_clinical_penalty(range_bias: dict[str, np.ndarray]) -> np.ndarray:
    penalty = np.zeros_like(next(iter(range_bias.values())), dtype=np.float32)
    penalty += 0.20 * np.maximum(0.0, -range_bias.get("elevated_bias_sbp", penalty))
    penalty += 0.10 * np.maximum(0.0, -range_bias.get("elevated_bias_dbp", penalty))
    penalty += 1.25 * np.maximum(0.0, -range_bias.get("high_bias_sbp", penalty))
    penalty += 0.70 * np.maximum(0.0, -range_bias.get("high_bias_dbp", penalty))
    penalty += 1.90 * np.maximum(0.0, -range_bias.get("crisis_bias_sbp", penalty))
    penalty += 1.10 * np.maximum(0.0, -range_bias.get("crisis_bias_dbp", penalty))
    return penalty.astype(np.float32)


def _baseline_bias_constants(cfg) -> dict[str, float]:
    baseline = v13._baseline_bp_range_map()
    return {
        "high_sbp": float(baseline.get("high", {}).get("bias_sbp", 0.0)),
        "high_dbp": float(baseline.get("high", {}).get("bias_dbp", 0.0)),
        "crisis_sbp": float(baseline.get("crisis", {}).get("bias_sbp", 0.0)),
        "crisis_dbp": float(baseline.get("crisis", {}).get("bias_dbp", 0.0)),
        "tol_sbp": float(getattr(cfg, "BASELINE_BIAS_WORSE_TOL_SBP", 0.75)),
        "tol_dbp": float(getattr(cfg, "BASELINE_BIAS_WORSE_TOL_DBP", 0.50)),
        "worse_weight": float(getattr(cfg, "BASELINE_BIAS_WORSE_WEIGHT", 2.20)),
        "high_target_sbp": float(getattr(cfg, "BASELINE_HIGH_SBP_TARGET", -5.0)),
        "high_target_dbp": float(getattr(cfg, "BASELINE_HIGH_DBP_TARGET", -3.0)),
        "crisis_target_sbp": float(getattr(cfg, "BASELINE_CRISIS_SBP_TARGET", -5.0)),
        "crisis_target_dbp": float(getattr(cfg, "BASELINE_CRISIS_DBP_TARGET", -3.0)),
        "high_target_weight": float(getattr(cfg, "BASELINE_HIGH_TARGET_WEIGHT", 1.45)),
        "crisis_target_weight": float(getattr(cfg, "BASELINE_CRISIS_TARGET_WEIGHT", 3.40)),
    }


def _batch_baseline_bias_penalty(range_bias: dict[str, np.ndarray], cfg) -> np.ndarray:
    if not bool(getattr(cfg, "BASELINE_AWARE_SELECTOR_ENABLE", True)):
        return np.zeros_like(next(iter(range_bias.values())), dtype=np.float32)
    c = _baseline_bias_constants(cfg)
    high_sbp = range_bias["high_bias_sbp"]
    high_dbp = range_bias["high_bias_dbp"]
    crisis_sbp = range_bias["crisis_bias_sbp"]
    crisis_dbp = range_bias["crisis_bias_dbp"]
    penalty = np.zeros_like(high_sbp, dtype=np.float32)
    penalty += 1.0 * c["worse_weight"] * np.maximum(0.0, (c["high_sbp"] - c["tol_sbp"]) - high_sbp)
    penalty += 1.0 * 0.65 * c["worse_weight"] * np.maximum(0.0, (c["high_dbp"] - c["tol_dbp"]) - high_dbp)
    penalty += 2.2 * c["worse_weight"] * np.maximum(0.0, (c["crisis_sbp"] - c["tol_sbp"]) - crisis_sbp)
    penalty += 2.2 * 0.65 * c["worse_weight"] * np.maximum(0.0, (c["crisis_dbp"] - c["tol_dbp"]) - crisis_dbp)
    penalty += c["high_target_weight"] * (
        np.maximum(0.0, c["high_target_sbp"] - high_sbp)
        + 0.55 * np.maximum(0.0, c["high_target_dbp"] - high_dbp)
    )
    penalty += c["crisis_target_weight"] * (
        np.maximum(0.0, c["crisis_target_sbp"] - crisis_sbp)
        + 0.65 * np.maximum(0.0, c["crisis_target_dbp"] - crisis_dbp)
    )
    return penalty.astype(np.float32)


def _batch_conformal_summary(
    calib_true: np.ndarray,
    calib_pred_batch: np.ndarray,
    calib_scale: np.ndarray,
    query_true: np.ndarray,
    query_pred_batch: np.ndarray,
    query_scale: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    target = np.float32(1.0 - float(alpha))
    scores = np.abs(calib_pred_batch - calib_true[None, :, :]) / np.clip(calib_scale[None, :, :], 1.0e-6, None)
    q = np.quantile(scores, 1.0 - float(alpha), axis=1, method="higher").astype(np.float32)
    half_width = q[:, None, :] * query_scale[None, :, :]
    cover = (
        (query_true[None, :, :] >= (query_pred_batch - half_width))
        & (query_true[None, :, :] <= (query_pred_batch + half_width))
    )
    coverage_sbp = cover[:, :, 0].mean(axis=1).astype(np.float32)
    coverage_dbp = cover[:, :, 1].mean(axis=1).astype(np.float32)
    miw_sbp = (2.0 * half_width[:, :, 0]).mean(axis=1).astype(np.float32)
    miw_dbp = (2.0 * half_width[:, :, 1]).mean(axis=1).astype(np.float32)
    return {
        "coverage_sbp": coverage_sbp,
        "coverage_dbp": coverage_dbp,
        "miw_sbp": miw_sbp,
        "miw_dbp": miw_dbp,
        "coverage_gap": (0.5 * (np.abs(coverage_sbp - target) + np.abs(coverage_dbp - target))).astype(np.float32),
        "miw_mean": (0.5 * (miw_sbp + miw_dbp)).astype(np.float32),
    }


def _candidate_key(row: dict) -> tuple:
    return (
        str(row.get("candidate", "")),
        float(row.get("score", 0.0)),
    )


def _merge_top_rows(existing: List[dict], incoming: Iterable[dict], keep: int) -> List[dict]:
    merged = list(existing) + list(incoming)
    dedup: dict[str, dict] = {}
    for row in merged:
        key = str(row.get("candidate", ""))
        current = dedup.get(key)
        if current is None or float(row.get("score", 0.0)) < float(current.get("score", 0.0)):
            dedup[key] = row
    rows = sorted(dedup.values(), key=lambda item: float(item.get("score", 0.0)))
    return rows[:keep]


def _search_cache_path(name: str) -> Path:
    return _artifacts_dir() / f"{name}_search_cache.json"


def _high_bias_signature(cfg) -> dict:
    return {
        "protocol_id": str(getattr(cfg, "PROTOCOL_ID", "")),
        "space": "high_bias",
        "scales": list(tuple(float(x) for x in cfg.RELIABILITY_BIAS_SCALES)),
        "betas": list(tuple(float(x) for x in cfg.RELIABILITY_BIAS_BETAS)),
        "floors": list(tuple(float(x) for x in cfg.RELIABILITY_BIAS_RELIABILITY_FLOORS)),
        "disagree": list(tuple(float(x) for x in cfg.RELIABILITY_BIAS_DISAGREE_GAINS)),
        "high_gain": list(tuple(float(x) for x in cfg.RELIABILITY_BIAS_HIGH_GAINS)),
        "crisis_gain": list(tuple(float(x) for x in cfg.RELIABILITY_BIAS_CRISIS_GAINS)),
        "neg_frac": list(tuple(float(x) for x in cfg.RELIABILITY_BIAS_NEGATIVE_FRACS)),
        "high_th": list(tuple(float(x) for x in cfg.RELIABILITY_BIAS_HIGH_THRESHOLDS)),
        "crisis_th": list(tuple(float(x) for x in cfg.RELIABILITY_BIAS_CRISIS_THRESHOLDS)),
        "high_floor": list(tuple(float(x) for x in cfg.RELIABILITY_BIAS_HIGH_FLOOR_SBP)),
        "crisis_floor": list(tuple(float(x) for x in cfg.RELIABILITY_BIAS_CRISIS_FLOOR_SBP)),
    }


def _crisis_signature(cfg) -> dict:
    return {
        "protocol_id": str(getattr(cfg, "PROTOCOL_ID", "")),
        "space": "crisis_tail",
        "high_th": list(tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_HIGH_THRESHOLDS)),
        "crisis_th": list(tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_CRISIS_THRESHOLDS)),
        "gamma": list(tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_GAMMAS)),
        "sbp_q": list(tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_SBP_QUANTILES)),
        "dbp_q": list(tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_DBP_QUANTILES)),
        "crisis_gain": list(tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_CRISIS_GAINS)),
        "sbp_margin": list(tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_SBP_MARGINS)),
        "dbp_margin": list(tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_DBP_MARGINS)),
        "unc_gain": list(tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_UNCERTAINTY_GAINS)),
        "model_scale": list(tuple(float(x) for x in getattr(cfg, "CRISIS_TAIL_FUSION_MODEL_SCALES", ()))),
        "expert_gain": list(tuple(float(x) for x in getattr(cfg, "CRISIS_TAIL_FUSION_EXPERT_GAINS", ()))),
    }


def _load_cached_rows(name: str, signature: dict) -> tuple[int, List[dict], bool]:
    payload = _read_json(_search_cache_path(name))
    if payload.get("signature") != signature:
        return 0, [], False
    return (
        int(payload.get("processed", 0)),
        list(payload.get("top_rows", [])),
        bool(payload.get("done", False)),
    )


def _save_cached_rows(
    name: str,
    processed: int,
    total: int,
    top_rows: List[dict],
    signature: dict,
    done: bool = False,
    extra: dict | None = None,
) -> None:
    payload = {
        "processed": int(processed),
        "total": int(total),
        "done": bool(done),
        "top_rows": top_rows,
        "signature": signature,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    _write_json(_search_cache_path(name), payload)


def _skip_generator(gen: Iterator[dict], n: int) -> Iterator[dict]:
    return itertools.islice(gen, n, None)


def _batched_rows(gen: Iterator[dict], batch_size: int) -> Iterator[List[dict]]:
    while True:
        batch = list(itertools.islice(gen, batch_size))
        if not batch:
            return
        yield batch


def _high_bias_candidate_iter(cfg) -> Iterator[dict]:
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
                                                yield {
                                                    "candidate": (
                                                        f"relbias_s{v13.base_script._tag(scale)}_b{v13.base_script._tag(beta)}"
                                                        f"_rf{v13.base_script._tag(reliability_floor)}"
                                                        f"_dg{v13.base_script._tag(disagree_gain)}"
                                                        f"_hg{v13.base_script._tag(high_gain)}"
                                                        f"_cg{v13.base_script._tag(crisis_gain)}"
                                                        f"_nf{v13.base_script._tag(negative_frac)}"
                                                        f"_ht{v13.base_script._tag(high_threshold)}"
                                                        f"_ct{v13.base_script._tag(crisis_threshold)}"
                                                        f"_hf{v13.base_script._tag(high_floor_sbp)}"
                                                        f"_cf{v13.base_script._tag(crisis_floor_sbp)}"
                                                    ),
                                                    "scale": scale,
                                                    "beta": beta,
                                                    "reliability_floor": reliability_floor,
                                                    "disagree_gain": disagree_gain,
                                                    "high_gain": high_gain,
                                                    "crisis_gain": crisis_gain,
                                                    "negative_frac": negative_frac,
                                                    "high_threshold": high_threshold,
                                                    "crisis_threshold": crisis_threshold,
                                                    "high_floor_sbp": high_floor_sbp,
                                                    "crisis_floor_sbp": crisis_floor_sbp,
                                                }


def _crisis_tail_candidate_iter(cfg) -> Iterator[dict]:
    model_scales = tuple(float(x) for x in getattr(cfg, "CRISIS_TAIL_FUSION_MODEL_SCALES", (0.85, 1.15, 1.45)))
    expert_gains = tuple(float(x) for x in getattr(cfg, "CRISIS_TAIL_FUSION_EXPERT_GAINS", (0.70, 1.00, 1.30)))
    for high_threshold in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_HIGH_THRESHOLDS):
        for crisis_threshold in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_CRISIS_THRESHOLDS):
            for gamma in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_GAMMAS):
                for sbp_quantile in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_SBP_QUANTILES):
                    for dbp_quantile in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_DBP_QUANTILES):
                        for crisis_gain in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_CRISIS_GAINS):
                            for sbp_margin in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_SBP_MARGINS):
                                for dbp_margin in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_DBP_MARGINS):
                                    for uncertainty_gain in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_UNCERTAINTY_GAINS):
                                        for model_scale in model_scales:
                                            for expert_gain in expert_gains:
                                                yield {
                                                    "candidate": (
                                                        f"crisis_debias_ht{str(high_threshold).replace('.', 'p')}"
                                                        f"_ct{str(crisis_threshold).replace('.', 'p')}"
                                                        f"_g{str(gamma).replace('.', 'p')}"
                                                        f"_sq{str(sbp_quantile).replace('.', 'p')}"
                                                        f"_dq{str(dbp_quantile).replace('.', 'p')}"
                                                        f"_cg{str(crisis_gain).replace('.', 'p')}"
                                                        f"_sm{str(sbp_margin).replace('.', 'p')}"
                                                        f"_dm{str(dbp_margin).replace('.', 'p')}"
                                                        f"_ug{str(uncertainty_gain).replace('.', 'p')}"
                                                        f"_ms{str(model_scale).replace('.', 'p')}"
                                                        f"_eg{str(expert_gain).replace('.', 'p')}"
                                                    ),
                                                    "high_threshold": high_threshold,
                                                    "crisis_threshold": crisis_threshold,
                                                    "gamma": gamma,
                                                    "sbp_quantile": sbp_quantile,
                                                    "dbp_quantile": dbp_quantile,
                                                    "crisis_gain": crisis_gain,
                                                    "sbp_margin": sbp_margin,
                                                    "dbp_margin": dbp_margin,
                                                    "uncertainty_gain": uncertainty_gain,
                                                    "model_scale": model_scale,
                                                    "expert_gain": expert_gain,
                                                }


def _product_len(values: Sequence[Sequence[float]]) -> int:
    total = 1
    for seq in values:
        total *= len(tuple(seq))
    return int(total)


def _high_bias_total(cfg) -> int:
    return _product_len(
        [
            cfg.RELIABILITY_BIAS_SCALES,
            cfg.RELIABILITY_BIAS_BETAS,
            cfg.RELIABILITY_BIAS_RELIABILITY_FLOORS,
            cfg.RELIABILITY_BIAS_DISAGREE_GAINS,
            cfg.RELIABILITY_BIAS_HIGH_GAINS,
            cfg.RELIABILITY_BIAS_CRISIS_GAINS,
            cfg.RELIABILITY_BIAS_NEGATIVE_FRACS,
            cfg.RELIABILITY_BIAS_HIGH_THRESHOLDS,
            cfg.RELIABILITY_BIAS_CRISIS_THRESHOLDS,
            cfg.RELIABILITY_BIAS_HIGH_FLOOR_SBP,
            cfg.RELIABILITY_BIAS_CRISIS_FLOOR_SBP,
        ]
    )


def _crisis_total(cfg) -> int:
    return _product_len(
        [
            cfg.CRISIS_TAIL_FUSION_HIGH_THRESHOLDS,
            cfg.CRISIS_TAIL_FUSION_CRISIS_THRESHOLDS,
            cfg.CRISIS_TAIL_FUSION_GAMMAS,
            cfg.CRISIS_TAIL_FUSION_SBP_QUANTILES,
            cfg.CRISIS_TAIL_FUSION_DBP_QUANTILES,
            cfg.CRISIS_TAIL_FUSION_CRISIS_GAINS,
            cfg.CRISIS_TAIL_FUSION_SBP_MARGINS,
            cfg.CRISIS_TAIL_FUSION_DBP_MARGINS,
            cfg.CRISIS_TAIL_FUSION_UNCERTAINTY_GAINS,
            getattr(cfg, "CRISIS_TAIL_FUSION_MODEL_SCALES", (0.85, 1.15, 1.45)),
            getattr(cfg, "CRISIS_TAIL_FUSION_EXPERT_GAINS", (0.70, 1.00, 1.30)),
        ]
    )


def _adaptive_batch_size(total_samples: int, stage: str) -> int:
    if stage == "high_bias":
        batch = max(96, min(640, int(960000 / max(1, total_samples))))
    else:
        batch = max(48, min(224, int(360000 / max(1, total_samples))))
    return int(batch)


def _class_proxy(prob: np.ndarray, cfg) -> np.ndarray:
    return v13.base_script._class_proxy(prob, cfg)


def _high_bias_precompute(reg_out: dict, cls_prob: np.ndarray, cfg) -> dict:
    pred = _to_float_array(reg_out["y_pred_reg"])
    y_true = _to_float_array(reg_out["y_true_reg"])
    base_prob = _normalize_prob(cls_prob)
    reg_prob = _normalize_prob(shared_v9.regression_to_class_prob(pred, reg_out.get("uncertainty"), cfg))
    proxy_delta = _class_proxy(base_prob, cfg) - pred
    disagreement = 0.5 * np.abs(reg_prob - base_prob).sum(axis=1).astype(np.float32)
    uncertainty = _to_float_array(reg_out.get("uncertainty", np.zeros(len(pred), dtype=np.float32))).reshape(-1)
    quality = _to_float_array(reg_out.get("quality", np.ones(len(pred), dtype=np.float32))).reshape(-1)
    uncertainty_norm = v13.base_script._normalize01(uncertainty)
    quality_norm = v13.base_script._normalize01(quality)
    reliability_base = (quality_norm * (1.0 - 0.70 * uncertainty_norm)).astype(np.float32)
    high_signal, crisis_signal = v13.base_script.meta_script.risk_guard_signals(reg_out, base_prob)
    return {
        "pred": pred,
        "y_true": y_true,
        "proxy_delta": proxy_delta.astype(np.float32),
        "disagreement": np.clip(disagreement, 1.0e-6, 1.0).astype(np.float32),
        "reliability_base": reliability_base,
        "high_signal": _to_float_array(high_signal).reshape(-1),
        "crisis_signal": _to_float_array(crisis_signal).reshape(-1),
        "unc_scale": np.sqrt(np.clip(uncertainty.reshape(-1, 1), 1.0e-6, None)).astype(np.float32),
        "masks": _range_masks(y_true),
    }


def _vectorized_high_bias_rows(
    batch_rows: List[dict],
    calib_pre: dict,
    query_pre: dict,
    base_ref: dict,
    cfg,
) -> List[dict]:
    if not batch_rows:
        return []

    b = len(batch_rows)
    scale = np.asarray([float(row["scale"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    beta = np.asarray([float(row["beta"]) for row in batch_rows], dtype=np.float32)
    reliability_floor = np.asarray([float(row["reliability_floor"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    disagree_gain = np.asarray([float(row["disagree_gain"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    high_gain = np.asarray([float(row["high_gain"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    crisis_gain = np.asarray([float(row["crisis_gain"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    negative_frac = np.asarray([float(row["negative_frac"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    high_threshold = np.asarray([float(row["high_threshold"]) for row in batch_rows], dtype=np.float32)
    crisis_threshold = np.asarray([float(row["crisis_threshold"]) for row in batch_rows], dtype=np.float32)
    high_floor_sbp = np.asarray([float(row["high_floor_sbp"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    crisis_floor_sbp = np.asarray([float(row["crisis_floor_sbp"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)

    max_shift_sbp = float(getattr(cfg, "RISK_GUARD_MAX_SHIFT_SBP", 12.0))
    max_shift_dbp = float(getattr(cfg, "RISK_GUARD_MAX_SHIFT_DBP", 8.0))

    def _apply(pre: dict) -> np.ndarray:
        pred = pre["pred"]
        proxy_delta = pre["proxy_delta"]
        reliability = reliability_floor + (1.0 - reliability_floor) * pre["reliability_base"][None, :]
        high_gate = np.stack(
            [_signal_ramp(pre["high_signal"], float(th), float(gm)) for th, gm in zip(high_threshold, beta)],
            axis=0,
        ).astype(np.float32)
        crisis_gate = np.stack(
            [_signal_ramp(pre["crisis_signal"], float(th), float(gm)) for th, gm in zip(crisis_threshold, beta)],
            axis=0,
        ).astype(np.float32)
        disagreement_scale = 1.0 + disagree_gain * np.power(pre["disagreement"][None, :], beta.reshape(b, 1), dtype=np.float32)
        risk_scale = 1.0 + high_gain * high_gate + crisis_gain * crisis_gate
        delta = scale[:, :, None] * reliability[:, :, None] * disagreement_scale[:, :, None] * risk_scale[:, :, None] * proxy_delta[None, :, :]
        delta[:, :, 0] = np.clip(delta[:, :, 0], -negative_frac * max_shift_sbp, max_shift_sbp)
        delta[:, :, 1] = np.clip(delta[:, :, 1], -negative_frac * max_shift_dbp, max_shift_dbp)
        high_floor = high_floor_sbp * high_gate
        crisis_floor = crisis_floor_sbp * crisis_gate
        delta[:, :, 0] = np.maximum(delta[:, :, 0], high_floor + crisis_floor)
        delta[:, :, 1] = np.maximum(delta[:, :, 1], 0.35 * high_floor + 0.45 * crisis_floor)
        corrected = pred[None, :, :] + delta
        corrected[:, :, 0] = np.clip(corrected[:, :, 0], 70.0, 200.0)
        corrected[:, :, 1] = np.clip(corrected[:, :, 1], 35.0, 130.0)
        return corrected.astype(np.float32)

    calib_pred = _apply(calib_pre)
    query_pred = _apply(query_pre)

    reg = _batch_regression_summary(query_pre["y_true"], query_pred)
    range_bias = _batch_bp_bias_summary(reg["err_sbp"], reg["err_dbp"], query_pre["masks"])
    conformal = _batch_conformal_summary(
        calib_pre["y_true"],
        calib_pred,
        calib_pre["unc_scale"],
        query_pre["y_true"],
        query_pred,
        query_pre["unc_scale"],
        alpha=float(cfg.CONFORMAL_ALPHA),
    )

    clinical_pen = _batch_clinical_penalty(range_bias)
    tail_pen = _batch_tail_bias_penalty(range_bias)
    baseline_pen = _batch_baseline_bias_penalty(range_bias, cfg)
    high_abs_pen = 0.85 * np.abs(range_bias["high_bias_sbp"]) + 0.45 * np.abs(range_bias["high_bias_dbp"])
    crisis_abs_pen = 0.45 * np.abs(range_bias["crisis_bias_sbp"]) + 0.25 * np.abs(range_bias["crisis_bias_dbp"])
    crisis_under_pen = 1.40 * np.maximum(0.0, -range_bias["crisis_bias_sbp"]) + 0.55 * np.maximum(0.0, -range_bias["crisis_bias_dbp"])
    mae_excess = np.maximum(0.0, reg["mae_mean"] - float(base_ref["mae_mean"]) - float(cfg.RELIABILITY_BIAS_MAX_MAE_DELTA))
    cov_excess = np.maximum(0.0, conformal["coverage_gap"] - float(base_ref["coverage_gap"]) - float(cfg.RELIABILITY_BIAS_MAX_COVERAGE_GAP_DELTA))
    score = (
        clinical_pen
        + 0.55 * tail_pen
        + high_abs_pen
        + crisis_abs_pen
        + crisis_under_pen
        + 18.0 * mae_excess
        + 8.0 * cov_excess
        + 0.012 * reg["mae_mean"]
        + 0.06 * (np.abs(reg["bias_sbp"]) + np.abs(reg["bias_dbp"]))
        + baseline_pen
    ).astype(np.float32)

    reliability_mean = (
        reliability_floor + (1.0 - reliability_floor) * query_pre["reliability_base"][None, :]
    ).mean(axis=1).astype(np.float32)
    shift = query_pred - query_pre["pred"][None, :, :]

    rows: List[dict] = []
    for idx, row in enumerate(batch_rows):
        rows.append(
            {
                **row,
                "score": float(score[idx]),
                "clinical_under_penalty": float(clinical_pen[idx]),
                "tail_bias_penalty": float(tail_pen[idx] + baseline_pen[idx]),
                "high_bias_sbp": float(range_bias["high_bias_sbp"][idx]),
                "high_bias_dbp": float(range_bias["high_bias_dbp"][idx]),
                "crisis_bias_sbp": float(range_bias["crisis_bias_sbp"][idx]),
                "crisis_bias_dbp": float(range_bias["crisis_bias_dbp"][idx]),
                "shift_mean_sbp": float(shift[idx, :, 0].mean()),
                "shift_mean_dbp": float(shift[idx, :, 1].mean()),
                "reliability_mean": float(reliability_mean[idx]),
                "mae_sbp": float(reg["mae_sbp"][idx]),
                "mae_dbp": float(reg["mae_dbp"][idx]),
                "mae_mean": float(reg["mae_mean"][idx]),
                "bias_sbp": float(reg["bias_sbp"][idx]),
                "bias_dbp": float(reg["bias_dbp"][idx]),
                "coverage_sbp": float(conformal["coverage_sbp"][idx]),
                "coverage_dbp": float(conformal["coverage_dbp"][idx]),
                "miw_sbp": float(conformal["miw_sbp"][idx]),
                "miw_dbp": float(conformal["miw_dbp"][idx]),
                "coverage_gap": float(conformal["coverage_gap"][idx]),
                "miw_mean": float(conformal["miw_mean"][idx]),
            }
        )
    return rows


def tailaware_search_high_bias_calibration_candidates(
    calib_out: dict,
    calib_cls_prob: np.ndarray,
    query_out: dict,
    query_cls_prob: np.ndarray,
    cfg,
) -> tuple[dict, List[dict]]:
    cache_name = "v10_15_high_bias"
    signature = _high_bias_signature(cfg)
    processed, top_rows, done = _load_cached_rows(cache_name, signature)
    if done and top_rows:
        print(f"[v10.15] Reusing cached exhaustive high-bias search results ({len(top_rows)} rows kept).")
        top_rows = sorted(top_rows, key=lambda item: float(item.get("score", 0.0)))
        return top_rows[0], top_rows

    start = time.time()
    calib_pre = _high_bias_precompute(calib_out, calib_cls_prob, cfg)
    query_pre = _high_bias_precompute(query_out, query_cls_prob, cfg)
    base_conformal = v13.base_script.meta_script.stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    base_ref = {
        "mae_mean": float(query_out["metrics_reg"]["mae_mean"]),
        "coverage_gap": float(base_conformal["coverage_gap"]),
    }
    total = _high_bias_total(cfg)
    batch_size = _adaptive_batch_size(max(len(calib_pre["pred"]), len(query_pre["pred"])), "high_bias")
    keep_rows = int(getattr(cfg, "VECTORFAST_HIGH_BIAS_KEEP_ROWS", 4096))
    flush_every = int(getattr(cfg, "VECTORFAST_CACHE_FLUSH_EVERY", 32))

    identity_range = v13._bp_range_map(v13.base_script.meta_script.stage_script.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"]))
    top_rows = _merge_top_rows(
        top_rows,
        [
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
                "score": float(v13.tailaware_high_bias_calibration_cost(calib_out, query_out, base_ref, cfg)[0]),
                "clinical_under_penalty": float(v13.base_script.meta_script.clinical_underestimation_penalty(list(identity_range.values()))),
                "tail_bias_penalty": float(v13.base_script._tail_bias_penalty(list(identity_range.values()))),
                "high_bias_sbp": float(identity_range.get("high", {}).get("bias_sbp", 0.0)),
                "high_bias_dbp": float(identity_range.get("high", {}).get("bias_dbp", 0.0)),
                "crisis_bias_sbp": float(identity_range.get("crisis", {}).get("bias_sbp", 0.0)),
                "crisis_bias_dbp": float(identity_range.get("crisis", {}).get("bias_dbp", 0.0)),
                "shift_mean_sbp": 0.0,
                "shift_mean_dbp": 0.0,
                "reliability_mean": 0.0,
                "mae_sbp": float(query_out["metrics_reg"].get("mae_sbp", 0.0)),
                "mae_dbp": float(query_out["metrics_reg"].get("mae_dbp", 0.0)),
                "mae_mean": float(query_out["metrics_reg"].get("mae_mean", 0.0)),
                "bias_sbp": float(query_out["metrics_reg"].get("bias_sbp", 0.0)),
                "bias_dbp": float(query_out["metrics_reg"].get("bias_dbp", 0.0)),
                "coverage_sbp": float(base_conformal.get("coverage_sbp", 0.0)),
                "coverage_dbp": float(base_conformal.get("coverage_dbp", 0.0)),
                "miw_sbp": float(base_conformal.get("miw_sbp", 0.0)),
                "miw_dbp": float(base_conformal.get("miw_dbp", 0.0)),
                "coverage_gap": float(base_conformal.get("coverage_gap", 0.0)),
                "miw_mean": float(base_conformal.get("miw_mean", 0.0)),
            }
        ],
        keep_rows,
    )

    gen = _skip_generator(_high_bias_candidate_iter(cfg), processed)
    processed_local = processed
    print(
        f"[v10.15] Exhaustive vectorized high-bias search starting from {processed_local}/{total} "
        f"candidates with batch_size={batch_size}."
    )
    for batch_idx, batch_rows in enumerate(_batched_rows(gen, batch_size), start=1):
        exact_rows = _vectorized_high_bias_rows(batch_rows, calib_pre, query_pre, base_ref, cfg)
        top_rows = _merge_top_rows(top_rows, exact_rows, keep_rows)
        processed_local += len(batch_rows)
        if batch_idx % flush_every == 0 or processed_local >= total:
            elapsed = time.time() - start
            print(
                f"[v10.15] High-bias search progress: {processed_local}/{total} "
                f"({100.0 * processed_local / max(1, total):.1f}%) | elapsed={elapsed / 60.0:.1f} min"
            )
            _save_cached_rows(
                cache_name,
                processed_local,
                total,
                top_rows,
                signature,
                done=False,
                extra={"stage": "high_bias"},
            )

    top_rows = sorted(top_rows, key=lambda item: float(item.get("score", 0.0)))
    _save_cached_rows(
        cache_name,
        total,
        total,
        top_rows,
        signature,
        done=True,
        extra={"stage": "high_bias", "runtime_sec": float(time.time() - start)},
    )
    print(f"[v10.15] Exhaustive high-bias search finished. Best candidate: {top_rows[0]['candidate']}")
    return top_rows[0], top_rows


def _tail_under_masks(y_true: np.ndarray, quantiles: Sequence[float]) -> List[tuple[int, np.ndarray, float]]:
    y_true = np.asarray(y_true, dtype=np.float32)
    rows: List[tuple[int, np.ndarray, float]] = []
    for idx, q in enumerate(tuple(float(x) for x in quantiles), start=1):
        for dim, weight in ((0, 0.90 + 0.45 * idx), (1, 0.35 + 0.18 * idx)):
            threshold = float(np.quantile(y_true[:, dim], q))
            mask = y_true[:, dim] >= threshold
            rows.append((dim, mask, float(weight)))
    return rows


def _top_tail_masks(y_true: np.ndarray, quantiles: Sequence[float]) -> List[tuple[int, np.ndarray, float]]:
    y_true = np.asarray(y_true, dtype=np.float32)
    rows: List[tuple[int, np.ndarray, float]] = []
    for q in tuple(float(x) for x in quantiles):
        for dim, weight in ((0, 1.0), (1, 0.45)):
            threshold = float(np.quantile(y_true[:, dim], q))
            mask = y_true[:, dim] >= threshold
            rows.append((dim, mask, float(weight)))
    return rows


def _batch_tail_under_penalty(err_sbp: np.ndarray, err_dbp: np.ndarray, masks: List[tuple[int, np.ndarray, float]]) -> np.ndarray:
    penalty = np.zeros(err_sbp.shape[0], dtype=np.float32)
    for dim, mask, weight in masks:
        if not np.any(mask):
            continue
        residual = err_sbp[:, mask] if dim == 0 else err_dbp[:, mask]
        mean_bias = residual.mean(axis=1).astype(np.float32)
        p90 = np.quantile(np.abs(residual), 0.90, axis=1).astype(np.float32)
        if dim == 0:
            penalty += weight * np.maximum(0.0, -mean_bias) + 0.08 * p90
        else:
            penalty += weight * np.maximum(0.0, -mean_bias) + 0.04 * p90
    return penalty.astype(np.float32)


def _batch_top_tail_under_penalty(err_sbp: np.ndarray, err_dbp: np.ndarray, masks: List[tuple[int, np.ndarray, float]]) -> np.ndarray:
    penalty = np.zeros(err_sbp.shape[0], dtype=np.float32)
    for dim, mask, weight in masks:
        if not np.any(mask):
            continue
        residual = err_sbp[:, mask] if dim == 0 else err_dbp[:, mask]
        bias = residual.mean(axis=1).astype(np.float32)
        under_rate = (residual < 0.0).mean(axis=1).astype(np.float32)
        penalty += weight * np.maximum(0.0, -bias) * (0.50 + under_rate)
    return penalty.astype(np.float32)


def _crisis_tail_precompute(
    reg_out: dict,
    cls_prob: np.ndarray,
    reg_inputs: dict,
    bundle: dict,
    cfg,
) -> dict:
    cls_prob = _normalize_prob(cls_prob)
    context = v13.base_script.meta_script.build_crisis_tail_signal_context(reg_out, cls_prob, reg_inputs)
    pred = _to_float_array(reg_out["y_pred_reg"])
    y_true = _to_float_array(reg_out["y_true_reg"])
    stage1 = cls_prob[:, 2].astype(np.float32)
    stage2 = cls_prob[:, 3].astype(np.float32)
    hypertensive = np.clip(stage1 + stage2, 0.0, 1.0).astype(np.float32)
    positive_delta = v13.prev_script.predict_crisis_debias_delta(bundle, reg_out, cls_prob, reg_inputs, cfg).astype(np.float32)
    uncertainty = _to_float_array(reg_out.get("uncertainty", np.zeros(len(pred), dtype=np.float32))).reshape(-1)
    other_experts = np.asarray(context["expert_stack"][:, 1:, :], dtype=np.float32)
    return {
        "pred": pred,
        "y_true": y_true,
        "cls_prob": cls_prob.astype(np.float32),
        "stage1": stage1,
        "stage2": stage2,
        "hypertensive": hypertensive,
        "expert_stack": np.asarray(context["expert_stack"], dtype=np.float32),
        "other_experts": other_experts,
        "expert_high_signal": np.asarray(context["expert_high_signal"], dtype=np.float32).reshape(-1),
        "expert_crisis_signal": np.asarray(context["expert_crisis_signal"], dtype=np.float32).reshape(-1),
        "spread_signal": np.asarray(context["spread_signal"], dtype=np.float32).reshape(-1),
        "uncertainty_signal": np.asarray(context["uncertainty_signal"], dtype=np.float32).reshape(-1),
        "positive_delta": positive_delta,
        "unc_scale": np.sqrt(np.clip(uncertainty.reshape(-1, 1), 1.0e-6, None)).astype(np.float32),
        "masks": _range_masks(y_true),
    }


def _build_base_crisis_tail_batch(
    batch_rows: List[dict],
    pre: dict,
    cfg,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    b = len(batch_rows)
    high_threshold = np.asarray([float(row["high_threshold"]) for row in batch_rows], dtype=np.float32)
    crisis_threshold = np.asarray([float(row["crisis_threshold"]) for row in batch_rows], dtype=np.float32)
    gamma = np.asarray([float(row["gamma"]) for row in batch_rows], dtype=np.float32)
    crisis_gain = np.asarray([float(row["crisis_gain"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    sbp_margin = np.asarray([float(row["sbp_margin"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    dbp_margin = np.asarray([float(row["dbp_margin"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    uncertainty_gain = np.asarray([float(row["uncertainty_gain"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    model_scale = np.asarray([float(row["model_scale"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    expert_gain = np.asarray([float(row["expert_gain"]) for row in batch_rows], dtype=np.float32).reshape(b, 1)
    sbp_quantile = [float(row["sbp_quantile"]) for row in batch_rows]
    dbp_quantile = [float(row["dbp_quantile"]) for row in batch_rows]

    high_gate = np.stack(
        [_signal_ramp(pre["expert_high_signal"], float(th), float(gm)) for th, gm in zip(high_threshold, gamma)],
        axis=0,
    ).astype(np.float32)
    crisis_gate = np.stack(
        [_signal_ramp(pre["expert_crisis_signal"], float(th), float(gm)) for th, gm in zip(crisis_threshold, gamma)],
        axis=0,
    ).astype(np.float32)
    gate_scale = (1.0 + uncertainty_gain * pre["uncertainty_signal"][None, :] + 0.35 * pre["spread_signal"][None, :]).astype(np.float32)

    sbp_anchor = np.stack(
        [np.quantile(pre["expert_stack"][:, :, 0], q, axis=1) for q in sbp_quantile],
        axis=0,
    ).astype(np.float32)
    dbp_anchor = np.stack(
        [np.quantile(pre["expert_stack"][:, :, 1], q, axis=1) for q in dbp_quantile],
        axis=0,
    ).astype(np.float32)

    pred = pre["pred"][None, :, :]
    sbp_expert_gap = np.clip(sbp_anchor - pred[:, :, 0], 0.0, None).astype(np.float32)
    dbp_expert_gap = np.clip(dbp_anchor - pred[:, :, 1], 0.0, None).astype(np.float32)
    disagreement_signal = np.clip(sbp_expert_gap / np.maximum(sbp_margin, 1.0), 0.0, 2.5).astype(np.float32)
    hazard_gate = np.clip(
        float(getattr(cfg, "CRISIS_TAIL_CLASS_STAGE1_GAIN", 0.85)) * pre["stage1"][None, :]
        + float(getattr(cfg, "CRISIS_TAIL_CLASS_STAGE2_GAIN", 1.85)) * pre["stage2"][None, :]
        + float(getattr(cfg, "CRISIS_TAIL_DISAGREEMENT_GAIN", 0.85)) * np.clip(disagreement_signal, 0.0, 1.5)
        + 0.35 * high_gate
        + 0.85 * crisis_gate
        + 0.22 * pre["uncertainty_signal"][None, :],
        0.0,
        3.0,
    ).astype(np.float32)

    sbp_gate = np.clip((0.32 * high_gate + crisis_gain * crisis_gate) * gate_scale, 0.0, 1.5).astype(np.float32)
    dbp_gate = np.clip((0.26 * high_gate + 0.72 * crisis_gain * crisis_gate) * gate_scale, 0.0, 1.35).astype(np.float32)
    sbp_anchor_delta = (expert_gain * sbp_expert_gap + sbp_margin * crisis_gate).astype(np.float32)
    dbp_anchor_delta = (expert_gain * dbp_expert_gap + dbp_margin * crisis_gate).astype(np.float32)
    sbp_hazard_delta = (
        np.clip(0.25 + 0.40 * hazard_gate + 0.30 * pre["hypertensive"][None, :], 0.0, 2.20)
        * (0.55 * sbp_expert_gap + sbp_margin)
    ).astype(np.float32)
    dbp_hazard_delta = (
        np.clip(0.18 + 0.28 * hazard_gate + 0.22 * pre["stage2"][None, :], 0.0, 1.60)
        * (0.45 * dbp_expert_gap + dbp_margin)
    ).astype(np.float32)

    sbp_delta = np.maximum(
        sbp_gate * model_scale * pre["positive_delta"][None, :, 0],
        np.clip(0.22 + 0.58 * high_gate + 0.95 * crisis_gate, 0.0, 1.4) * sbp_anchor_delta,
    )
    dbp_delta = np.maximum(
        dbp_gate * model_scale * pre["positive_delta"][None, :, 1],
        np.clip(0.18 + 0.42 * high_gate + 0.72 * crisis_gate, 0.0, 1.25) * dbp_anchor_delta,
    )
    sbp_delta = np.maximum(sbp_delta, sbp_hazard_delta)
    dbp_delta = np.maximum(dbp_delta, dbp_hazard_delta)

    hard_crisis_mask = (
        (crisis_gate >= 0.80)
        | (pre["stage2"][None, :] >= 0.55)
        | ((pre["hypertensive"][None, :] >= 0.78) & (sbp_expert_gap >= 6.5))
        | ((pre["expert_high_signal"][None, :] >= 0.92) & (sbp_expert_gap >= 8.0))
    )
    sbp_delta = np.where(
        hard_crisis_mask,
        np.maximum(sbp_delta, sbp_expert_gap + 0.5 * sbp_margin),
        sbp_delta,
    )
    dbp_delta = np.where(
        hard_crisis_mask,
        np.maximum(dbp_delta, 0.70 * dbp_expert_gap + 0.5 * dbp_margin),
        dbp_delta,
    )
    sbp_hard_floor = np.where(
        hard_crisis_mask,
        np.maximum(
            float(getattr(cfg, "CRISIS_TAIL_HARD_FLOOR_SBP", 13.0)),
            0.55 * sbp_expert_gap + 0.65 * sbp_margin,
        ),
        0.0,
    ).astype(np.float32)
    dbp_hard_floor = np.where(
        hard_crisis_mask,
        np.maximum(
            float(getattr(cfg, "CRISIS_TAIL_HARD_FLOOR_DBP", 4.5)),
            0.45 * dbp_expert_gap + 0.55 * dbp_margin,
        ),
        0.0,
    ).astype(np.float32)
    sbp_delta = np.maximum(sbp_delta, sbp_hard_floor)
    dbp_delta = np.maximum(dbp_delta, dbp_hard_floor)
    delta = np.stack([sbp_delta, dbp_delta], axis=2).astype(np.float32)
    delta[:, :, 0] = np.clip(delta[:, :, 0], 0.0, float(getattr(cfg, "CRISIS_TAIL_MAX_SHIFT_SBP", cfg.RISK_GUARD_MAX_SHIFT_SBP)))
    delta[:, :, 1] = np.clip(delta[:, :, 1], 0.0, float(getattr(cfg, "CRISIS_TAIL_MAX_SHIFT_DBP", cfg.RISK_GUARD_MAX_SHIFT_DBP)))
    corrected = pre["pred"][None, :, :] + delta
    corrected[:, :, 0] = np.clip(corrected[:, :, 0], 70.0, 200.0)
    corrected[:, :, 1] = np.clip(corrected[:, :, 1], 35.0, 130.0)

    diag = {
        "fusion_gate_mean": (0.5 * (np.maximum(sbp_gate, hazard_gate).mean(axis=1) + np.maximum(dbp_gate, 0.80 * hazard_gate).mean(axis=1))).astype(np.float32),
        "shift_mean_sbp": delta[:, :, 0].mean(axis=1).astype(np.float32),
        "shift_mean_dbp": delta[:, :, 1].mean(axis=1).astype(np.float32),
        "activation_rate": np.mean((crisis_gate >= 0.15) | (high_gate >= 0.20) | (hazard_gate >= 0.70), axis=1).astype(np.float32),
    }
    return corrected.astype(np.float32), diag


def _apply_guard_batch(pred_batch: np.ndarray, pre: dict, cfg) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    b = pred_batch.shape[0]
    cls_prob = pre["cls_prob"]
    stage1 = np.broadcast_to(pre["stage1"][None, :], (b, pre["stage1"].shape[0])).astype(np.float32)
    stage2 = np.broadcast_to(pre["stage2"][None, :], (b, pre["stage2"].shape[0])).astype(np.float32)
    hypertensive = np.broadcast_to(pre["hypertensive"][None, :], (b, pre["hypertensive"].shape[0])).astype(np.float32)
    other = pre["other_experts"][None, :, :, :]
    pred = np.asarray(pred_batch, dtype=np.float32)

    expert_stack = np.concatenate([pred[:, :, None, :], np.repeat(other, b, axis=0)], axis=2).astype(np.float32)
    expert_peak = np.max(expert_stack, axis=2).astype(np.float32)
    expert_q = np.quantile(expert_stack, float(getattr(cfg, "CRISIS_SBP_GUARD_QUANTILE", 0.98)), axis=2).astype(np.float32)

    cls_high = np.broadcast_to(
        np.clip(cls_prob[None, :, 2] + 0.85 * cls_prob[None, :, 3], 0.0, 1.5),
        (b, cls_prob.shape[0]),
    ).astype(np.float32)
    cls_crisis = np.broadcast_to(
        np.clip(cls_prob[None, :, 3], 0.0, 1.0),
        (b, cls_prob.shape[0]),
    ).astype(np.float32)

    expert_high_signal = np.maximum.reduce(
        [
            cls_high,
            _sigmoid((pred[:, :, 0] - 138.0) / 7.5),
            _sigmoid((pred[:, :, 1] - 88.0) / 5.5),
            _sigmoid((expert_q[:, :, 0] - 138.0) / 7.0),
            _sigmoid((expert_q[:, :, 1] - 88.0) / 5.0),
        ]
    ).astype(np.float32)
    expert_crisis_signal = np.maximum.reduce(
        [
            cls_crisis,
            _sigmoid((pred[:, :, 0] - 170.0) / 5.5),
            _sigmoid((pred[:, :, 1] - 110.0) / 4.5),
            _sigmoid((expert_q[:, :, 0] - 170.0) / 5.0),
            _sigmoid((expert_q[:, :, 1] - 108.0) / 4.5),
        ]
    ).astype(np.float32)

    risk_signal = np.maximum.reduce(
        [
            expert_crisis_signal,
            0.55 * expert_high_signal + 0.45 * stage2,
            _sigmoid((expert_peak[:, :, 0] - 170.0) / 7.5),
            _sigmoid((expert_peak[:, :, 1] - 108.0) / 5.5),
            0.35 * hypertensive + 0.65 * stage2,
        ]
    ).astype(np.float32)
    gate = np.stack(
        [_signal_ramp(risk_signal[idx], float(getattr(cfg, "CRISIS_SBP_GUARD_TRIGGER", 0.52)), 1.0) for idx in range(b)],
        axis=0,
    ).astype(np.float32)
    expert_gate = (expert_peak[:, :, 0] >= float(getattr(cfg, "CRISIS_SBP_GUARD_MIN_EXPERT_SBP", 165.0))).astype(np.float32)
    cls_gate = ((stage2 >= 0.28) | (hypertensive >= 0.72)).astype(np.float32)
    gate = np.clip(gate * np.maximum(expert_gate, cls_gate), 0.0, 1.0).astype(np.float32)

    sbp_target = np.maximum(
        expert_q[:, :, 0] + float(getattr(cfg, "CRISIS_SBP_GUARD_MARGIN", 3.5)) * gate,
        np.where(
            (gate >= 0.65) & (stage2 >= 0.35),
            float(getattr(cfg, "CRISIS_SBP_GUARD_ABSOLUTE_FLOOR", 172.0)),
            pred[:, :, 0],
        ),
    ).astype(np.float32)
    dbp_target = np.maximum(
        expert_q[:, :, 1] + 0.35 * float(getattr(cfg, "CRISIS_SBP_GUARD_MARGIN", 3.5)) * gate,
        np.where(
            (gate >= 0.75) & (stage2 >= 0.45),
            float(getattr(cfg, "CRISIS_DBP_GUARD_ABSOLUTE_FLOOR", 106.0)),
            pred[:, :, 1],
        ),
    ).astype(np.float32)
    sbp_gap = np.clip(expert_q[:, :, 0] - pred[:, :, 0], 0.0, None).astype(np.float32)
    dbp_gap = np.clip(expert_q[:, :, 1] - pred[:, :, 1], 0.0, None).astype(np.float32)
    sbp_delta = (
        float(getattr(cfg, "CRISIS_SBP_GUARD_GAIN", 0.88))
        * gate
        * np.maximum(sbp_target - pred[:, :, 0], 0.0)
    ).astype(np.float32)
    dbp_delta = (
        float(getattr(cfg, "CRISIS_DBP_GUARD_GAIN", 0.45))
        * gate
        * np.maximum(dbp_target - pred[:, :, 1], 0.0)
    ).astype(np.float32)
    sbp_delta = np.maximum(sbp_delta, gate * np.minimum(sbp_gap, float(getattr(cfg, "CRISIS_SBP_GUARD_MAX_EXTRA_SHIFT", 18.0))))
    dbp_delta = np.maximum(
        dbp_delta,
        0.55 * gate * np.minimum(dbp_gap, float(getattr(cfg, "CRISIS_DBP_GUARD_MAX_EXTRA_SHIFT", 8.0))),
    )
    sbp_delta = np.clip(sbp_delta, 0.0, float(getattr(cfg, "CRISIS_SBP_GUARD_MAX_EXTRA_SHIFT", 18.0))).astype(np.float32)
    dbp_delta = np.clip(dbp_delta, 0.0, float(getattr(cfg, "CRISIS_DBP_GUARD_MAX_EXTRA_SHIFT", 8.0))).astype(np.float32)
    corrected = pred.copy()
    corrected[:, :, 0] = np.clip(corrected[:, :, 0] + sbp_delta, 70.0, 200.0)
    corrected[:, :, 1] = np.clip(corrected[:, :, 1] + dbp_delta, 35.0, 130.0)
    diag = {
        "guard_shift_mean_sbp": sbp_delta.mean(axis=1).astype(np.float32),
        "guard_shift_mean_dbp": dbp_delta.mean(axis=1).astype(np.float32),
        "guard_activation_rate": np.mean(gate >= 0.20, axis=1).astype(np.float32),
    }
    return corrected.astype(np.float32), diag


def _vectorized_crisis_tail_rows(
    batch_rows: List[dict],
    calib_pre: dict,
    query_pre: dict,
    base_ref: dict,
    cfg,
) -> List[dict]:
    if not batch_rows:
        return []

    calib_pred_pre_guard, _ = _build_base_crisis_tail_batch(batch_rows, calib_pre, cfg)
    query_pred_pre_guard, diag = _build_base_crisis_tail_batch(batch_rows, query_pre, cfg)
    calib_pred, _ = _apply_guard_batch(calib_pred_pre_guard, calib_pre, cfg)
    query_pred, guard_diag = _apply_guard_batch(query_pred_pre_guard, query_pre, cfg)

    reg = _batch_regression_summary(query_pre["y_true"], query_pred)
    range_bias = _batch_bp_bias_summary(reg["err_sbp"], reg["err_dbp"], query_pre["masks"])
    conformal = _batch_conformal_summary(
        calib_pre["y_true"],
        calib_pred,
        calib_pre["unc_scale"],
        query_pre["y_true"],
        query_pred,
        query_pre["unc_scale"],
        alpha=float(cfg.CONFORMAL_ALPHA),
    )
    clinical_pen = _batch_clinical_penalty(range_bias)
    tail_pen = _batch_tail_bias_penalty(range_bias)
    tail_masks = _tail_under_masks(query_pre["y_true"], getattr(cfg, "CRISIS_TAIL_SURROGATE_QUANTILES", (0.90, 0.95, 0.98)))
    top_tail_masks = _top_tail_masks(query_pre["y_true"], getattr(cfg, "CRISIS_TAIL_SURROGATE_QUANTILES", (0.88, 0.92, 0.95, 0.98)))
    surrogate_pen = _batch_tail_under_penalty(reg["err_sbp"], reg["err_dbp"], tail_masks)
    top_tail_pen = _batch_top_tail_under_penalty(reg["err_sbp"], reg["err_dbp"], top_tail_masks)
    baseline_pen = _batch_baseline_bias_penalty(range_bias, cfg)

    crisis_under_pen = (
        float(getattr(cfg, "CRISIS_TAIL_UNDEREST_WEIGHT_SBP", 8.50)) * np.maximum(0.0, -range_bias["crisis_bias_sbp"])
        + float(getattr(cfg, "CRISIS_TAIL_UNDEREST_WEIGHT_DBP", 3.10)) * np.maximum(0.0, -range_bias["crisis_bias_dbp"])
    ).astype(np.float32)
    crisis_abs_pen = (1.35 * np.abs(range_bias["crisis_bias_sbp"]) + 0.70 * np.abs(range_bias["crisis_bias_dbp"])).astype(np.float32)
    high_under_pen = (1.55 * np.maximum(0.0, -range_bias["high_bias_sbp"]) + 0.82 * np.maximum(0.0, -range_bias["high_bias_dbp"])).astype(np.float32)
    mae_excess = np.maximum(0.0, reg["mae_mean"] - float(base_ref["mae_mean"]) - float(cfg.CRISIS_TAIL_FUSION_MAX_MAE_DELTA))
    cov_excess = np.maximum(0.0, conformal["coverage_gap"] - float(base_ref["coverage_gap"]) - float(cfg.CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA))

    base_score = (
        1.90 * surrogate_pen
        + crisis_under_pen
        + 0.55 * crisis_abs_pen
        + 0.75 * high_under_pen
        + 0.42 * clinical_pen
        + 0.20 * tail_pen
        + 18.0 * mae_excess
        + 8.0 * cov_excess
        + 0.008 * reg["mae_mean"]
    ).astype(np.float32)
    crisis_sbp_bias = range_bias["crisis_bias_sbp"]
    crisis_dbp_bias = range_bias["crisis_bias_dbp"]
    high_sbp_bias = range_bias["high_bias_sbp"]
    guard_shift = guard_diag["guard_shift_mean_sbp"]
    guard_activation = guard_diag["guard_activation_rate"]
    guard_overuse_pen = np.maximum(0.0, guard_activation - 0.18) * np.maximum(0.0, guard_shift - 1.2)
    score = (
        base_score
        + 4.25 * np.maximum(0.0, -crisis_sbp_bias)
        + 1.35 * np.maximum(0.0, -crisis_dbp_bias)
        + 1.15 * np.maximum(0.0, -high_sbp_bias)
        + 2.10 * top_tail_pen
        + baseline_pen
        + 2.50 * guard_overuse_pen
    ).astype(np.float32)

    rows: List[dict] = []
    for idx, row in enumerate(batch_rows):
        rows.append(
            {
                **row,
                "score": float(score[idx]),
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
                "coverage_sbp": float(conformal["coverage_sbp"][idx]),
                "coverage_dbp": float(conformal["coverage_dbp"][idx]),
                "miw_sbp": float(conformal["miw_sbp"][idx]),
                "miw_dbp": float(conformal["miw_dbp"][idx]),
                "coverage_gap": float(conformal["coverage_gap"][idx]),
                "miw_mean": float(conformal["miw_mean"][idx]),
            }
        )
    return rows


def search_crisis_tail_debias_candidates(
    calib_out: dict,
    calib_cls_prob: np.ndarray,
    calib_reg_inputs: dict,
    query_out: dict,
    query_cls_prob: np.ndarray,
    query_reg_inputs: dict,
    cfg,
) -> tuple[dict, List[dict]]:
    cache_name = "v10_15_crisis_tail"
    signature = _crisis_signature(cfg)
    processed, top_rows, done = _load_cached_rows(cache_name, signature)
    if done and top_rows:
        print(f"[v10.15] Reusing cached exhaustive crisis-tail search results ({len(top_rows)} rows kept).")
        top_rows = sorted(top_rows, key=lambda item: float(item.get("score", 0.0)))
        return top_rows[0], top_rows

    start = time.time()
    bundle_key = v13.prev_script._crisis_debias_registry_key(cfg)
    v13.prev_script._CRISIS_DEBIAS_REGISTRY[bundle_key] = v13.prev_script.fit_crisis_debias_bundle(
        calib_out,
        calib_cls_prob,
        calib_reg_inputs,
        cfg,
        seed=int(cfg.SEED) + 4701,
    )
    bundle = v13.prev_script._CRISIS_DEBIAS_REGISTRY[bundle_key]
    calib_pre = _crisis_tail_precompute(calib_out, calib_cls_prob, calib_reg_inputs, bundle, cfg)
    query_pre = _crisis_tail_precompute(query_out, query_cls_prob, query_reg_inputs, bundle, cfg)
    base_conformal = v13.base_script.meta_script.stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    base_ref = {
        "mae_mean": float(query_out["metrics_reg"]["mae_mean"]),
        "coverage_gap": float(base_conformal["coverage_gap"]),
    }
    total = _crisis_total(cfg)
    batch_size = _adaptive_batch_size(max(len(calib_pre["pred"]), len(query_pre["pred"])), "crisis_tail")
    keep_rows = int(getattr(cfg, "VECTORFAST_CRISIS_KEEP_ROWS", 4096))
    flush_every = int(getattr(cfg, "VECTORFAST_CACHE_FLUSH_EVERY", 24))

    identity_range = v13._bp_range_map(v13.base_script.meta_script.stage_script.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"]))
    top_rows = _merge_top_rows(
        top_rows,
        [
            {
                "candidate": "identity",
                "bundle_key": bundle_key,
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
                "score": float(v13.crisis_tail_debias_cost(calib_out, query_out, base_ref, cfg)[0]),
                "clinical_under_penalty": float(v13.base_script.meta_script.clinical_underestimation_penalty(list(identity_range.values()))),
                "tail_bias_penalty": float(v13.base_script._tail_bias_penalty(list(identity_range.values()))),
                "high_bias_sbp": float(identity_range.get("high", {}).get("bias_sbp", 0.0)),
                "high_bias_dbp": float(identity_range.get("high", {}).get("bias_dbp", 0.0)),
                "crisis_bias_sbp": float(identity_range.get("crisis", {}).get("bias_sbp", 0.0)),
                "crisis_bias_dbp": float(identity_range.get("crisis", {}).get("bias_dbp", 0.0)),
                "fusion_gate_mean": 0.0,
                "shift_mean_sbp": 0.0,
                "shift_mean_dbp": 0.0,
                "activation_rate": 0.0,
                "guard_shift_mean_sbp": 0.0,
                "guard_shift_mean_dbp": 0.0,
                "guard_activation_rate": 0.0,
                "mae_sbp": float(query_out["metrics_reg"].get("mae_sbp", 0.0)),
                "mae_dbp": float(query_out["metrics_reg"].get("mae_dbp", 0.0)),
                "mae_mean": float(query_out["metrics_reg"].get("mae_mean", 0.0)),
                "bias_sbp": float(query_out["metrics_reg"].get("bias_sbp", 0.0)),
                "bias_dbp": float(query_out["metrics_reg"].get("bias_dbp", 0.0)),
                "coverage_sbp": float(base_conformal.get("coverage_sbp", 0.0)),
                "coverage_dbp": float(base_conformal.get("coverage_dbp", 0.0)),
                "miw_sbp": float(base_conformal.get("miw_sbp", 0.0)),
                "miw_dbp": float(base_conformal.get("miw_dbp", 0.0)),
                "coverage_gap": float(base_conformal.get("coverage_gap", 0.0)),
                "miw_mean": float(base_conformal.get("miw_mean", 0.0)),
            }
        ],
        keep_rows,
    )

    gen = _skip_generator(_crisis_tail_candidate_iter(cfg), processed)
    processed_local = processed
    print(
        f"[v10.15] Exhaustive vectorized crisis-tail search starting from {processed_local}/{total} "
        f"candidates with batch_size={batch_size}."
    )
    for batch_idx, batch_rows in enumerate(_batched_rows(gen, batch_size), start=1):
        for row in batch_rows:
            row["bundle_key"] = bundle_key
        exact_rows = _vectorized_crisis_tail_rows(batch_rows, calib_pre, query_pre, base_ref, cfg)
        top_rows = _merge_top_rows(top_rows, exact_rows, keep_rows)
        processed_local += len(batch_rows)
        if batch_idx % flush_every == 0 or processed_local >= total:
            elapsed = time.time() - start
            print(
                f"[v10.15] Crisis-tail search progress: {processed_local}/{total} "
                f"({100.0 * processed_local / max(1, total):.1f}%) | elapsed={elapsed / 60.0:.1f} min"
            )
            _save_cached_rows(
                cache_name,
                processed_local,
                total,
                top_rows,
                signature,
                done=False,
                extra={"stage": "crisis_tail"},
            )

    top_rows = sorted(top_rows, key=lambda item: float(item.get("score", 0.0)))
    _save_cached_rows(
        cache_name,
        total,
        total,
        top_rows,
        signature,
        done=True,
        extra={"stage": "crisis_tail", "runtime_sec": float(time.time() - start)},
    )
    print(f"[v10.15] Exhaustive crisis-tail search finished. Best candidate: {top_rows[0]['candidate']}")
    return top_rows[0], top_rows


def build_nextgen_cfg():
    cfg = _ORIG_V13_BUILD_NEXTGEN_CFG()
    cfg.OUTPUT_NAME = FINAL_OUTPUT_NAME
    cfg.PROTOCOL_ID = FINAL_PROTOCOL_ID
    cfg.PROTOCOL_NAME = (
        "v10.15 subject-disjoint PiSO-inspired uncertainty-MoE exhaustive vector-fast protocol "
        "(reuse v10.13 backbones + exhaustive batched high-bias/crisis-tail search + stronger crisis guard)"
    )
    cfg.FULLTRAIN_OPTLONG_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
    cfg.FULLTRAIN_DUALMAX_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT
    cfg.WARMSTART_CANDIDATES = tuple(
        dict.fromkeys(
            tuple(getattr(cfg, "WARMSTART_CANDIDATES", ()))
            + (
                "mimic_bp_reg_v10_13_subjectdisjoint_piso_uncertainty_moe_tailaware_proto",
                "mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_proto",
            )
        )
    )

    # Keep the broad candidate spaces from v10.13 but make the late search exact and batched.
    cfg.BASELINE_AWARE_SELECTOR_ENABLE = True
    cfg.BASELINE_ACC_SHORTFALL_WEIGHT = max(float(getattr(cfg, "BASELINE_ACC_SHORTFALL_WEIGHT", 18.0)), 24.0)
    cfg.BASELINE_F1_SHORTFALL_WEIGHT = max(float(getattr(cfg, "BASELINE_F1_SHORTFALL_WEIGHT", 22.0)), 28.0)
    cfg.BASELINE_ASPIRATIONAL_ACC = max(float(getattr(cfg, "BASELINE_ASPIRATIONAL_ACC", 0.86)), 0.86)
    cfg.BASELINE_ASPIRATIONAL_F1 = max(float(getattr(cfg, "BASELINE_ASPIRATIONAL_F1", 0.80)), 0.80)
    cfg.BASELINE_ASPIRATIONAL_ACC_WEIGHT = max(float(getattr(cfg, "BASELINE_ASPIRATIONAL_ACC_WEIGHT", 2.40)), 3.0)
    cfg.BASELINE_ASPIRATIONAL_F1_WEIGHT = max(float(getattr(cfg, "BASELINE_ASPIRATIONAL_F1_WEIGHT", 2.80)), 3.3)
    cfg.BASELINE_CRISIS_SBP_TARGET = -5.0
    cfg.BASELINE_CRISIS_DBP_TARGET = -2.5
    cfg.BASELINE_HIGH_SBP_TARGET = -4.5
    cfg.BASELINE_HIGH_DBP_TARGET = -2.5
    cfg.BASELINE_CRISIS_TARGET_WEIGHT = max(float(getattr(cfg, "BASELINE_CRISIS_TARGET_WEIGHT", 3.40)), 4.10)
    cfg.BASELINE_HIGH_TARGET_WEIGHT = max(float(getattr(cfg, "BASELINE_HIGH_TARGET_WEIGHT", 1.45)), 1.70)

    # Slightly stronger conservative crisis guard than v10.13 without changing the candidate space.
    cfg.CRISIS_SBP_GUARD_ENABLE = True
    cfg.CRISIS_SBP_GUARD_TRIGGER = min(float(getattr(cfg, "CRISIS_SBP_GUARD_TRIGGER", 0.52)), 0.48)
    cfg.CRISIS_SBP_GUARD_QUANTILE = max(float(getattr(cfg, "CRISIS_SBP_GUARD_QUANTILE", 0.98)), 0.985)
    cfg.CRISIS_SBP_GUARD_ABSOLUTE_FLOOR = max(float(getattr(cfg, "CRISIS_SBP_GUARD_ABSOLUTE_FLOOR", 172.0)), 173.0)
    cfg.CRISIS_SBP_GUARD_GAIN = max(float(getattr(cfg, "CRISIS_SBP_GUARD_GAIN", 0.88)), 0.96)
    cfg.CRISIS_SBP_GUARD_MAX_EXTRA_SHIFT = max(float(getattr(cfg, "CRISIS_SBP_GUARD_MAX_EXTRA_SHIFT", 18.0)), 20.0)
    cfg.CRISIS_DBP_GUARD_ABSOLUTE_FLOOR = max(float(getattr(cfg, "CRISIS_DBP_GUARD_ABSOLUTE_FLOOR", 106.0)), 107.0)
    cfg.CRISIS_DBP_GUARD_GAIN = max(float(getattr(cfg, "CRISIS_DBP_GUARD_GAIN", 0.45)), 0.52)
    cfg.CRISIS_DBP_GUARD_MAX_EXTRA_SHIFT = max(float(getattr(cfg, "CRISIS_DBP_GUARD_MAX_EXTRA_SHIFT", 8.0)), 9.0)

    cfg.VECTORFAST_HIGH_BIAS_KEEP_ROWS = 4096
    cfg.VECTORFAST_CRISIS_KEEP_ROWS = 4096
    cfg.VECTORFAST_CACHE_FLUSH_EVERY = 24

    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    _tables_dir()
    _figures_dir()
    _artifacts_dir()
    return cfg


def generate_extra_outputs(output_dir: Path) -> None:
    _ORIG_V13_GENERATE_EXTRA_OUTPUTS(output_dir)
    runtime_audit = {
        "protocol_id": FINAL_PROTOCOL_ID,
        "search_strategy": {
            "high_bias": {
                "candidate_space": _high_bias_total(build_nextgen_cfg()),
                "mode": "exact exhaustive search with batched vectorized scoring and resume cache",
                "cache_file": str(_search_cache_path("v10_15_high_bias")),
            },
            "crisis_tail": {
                "candidate_space": _crisis_total(build_nextgen_cfg()),
                "mode": "exact exhaustive search with batched vectorized scoring and resume cache",
                "cache_file": str(_search_cache_path("v10_15_crisis_tail")),
            },
        },
        "backbone_reuse": {
            "optlong": str(_project_root() / "outputs" / OPTLONG_FULLTRAIN_OUTPUT / "best_model.pt"),
            "dualmax": str(_project_root() / "outputs" / DUALMAX_FULLTRAIN_OUTPUT / "best_model.pt"),
        },
        "notes": [
            "v10.15 keeps the v10.13 reliability-bias and crisis-tail candidate spaces instead of shrinking the grids.",
            "The slow late-stage search is replaced by vectorized batch evaluation over the full candidate space, with progress caches so reruns resume from the last processed block.",
            "The output artifact schema stays aligned with the v10.11/v10.13 pipeline while adding this runtime audit.",
        ],
    }
    _write_json(Path(output_dir) / "v10_15_vectorfast_runtime_audit.json", runtime_audit)


def main() -> None:
    originals = {
        "final_output": v13.FINAL_OUTPUT_NAME,
        "protocol_id": v13.FINAL_PROTOCOL_ID,
        "optlong": v13.OPTLONG_FULLTRAIN_OUTPUT,
        "dualmax": v13.DUALMAX_FULLTRAIN_OUTPUT,
        "build_nextgen": v13.build_nextgen_cfg,
        "generate_extra": v13.generate_extra_outputs,
        "high_bias_search": v13.tailaware_search_high_bias_calibration_candidates,
        "crisis_search": v13.prev_script.search_crisis_tail_debias_candidates,
    }
    try:
        v13.FINAL_OUTPUT_NAME = FINAL_OUTPUT_NAME
        v13.FINAL_PROTOCOL_ID = FINAL_PROTOCOL_ID
        v13.OPTLONG_FULLTRAIN_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
        v13.DUALMAX_FULLTRAIN_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT
        v13.build_nextgen_cfg = build_nextgen_cfg
        v13.generate_extra_outputs = generate_extra_outputs
        v13.tailaware_search_high_bias_calibration_candidates = tailaware_search_high_bias_calibration_candidates
        v13.prev_script.search_crisis_tail_debias_candidates = search_crisis_tail_debias_candidates
        v13.main()
    finally:
        v13.FINAL_OUTPUT_NAME = originals["final_output"]
        v13.FINAL_PROTOCOL_ID = originals["protocol_id"]
        v13.OPTLONG_FULLTRAIN_OUTPUT = originals["optlong"]
        v13.DUALMAX_FULLTRAIN_OUTPUT = originals["dualmax"]
        v13.build_nextgen_cfg = originals["build_nextgen"]
        v13.generate_extra_outputs = originals["generate_extra"]
        v13.tailaware_search_high_bias_calibration_candidates = originals["high_bias_search"]
        v13.prev_script.search_crisis_tail_debias_candidates = originals["crisis_search"]


if __name__ == "__main__":
    main()
