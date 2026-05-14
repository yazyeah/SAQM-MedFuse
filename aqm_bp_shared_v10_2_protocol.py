from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import aqm_bp_shared_v11_protocol as v11
from aqm_bp_shared_v11_protocol import *  # noqa: F401,F403


@dataclass
class CFG(v11.CFG):
    DATA_ROOT: str = os.environ.get("AQM_MIMIC_BP_ROOT", v11.CFG.DATA_ROOT)
    OUTPUT_NAME: str = "mimic_bp_reg_v10_2_proto"
    EPOCHS: int = 30
    EARLY_STOPPING_PATIENCE: int = 10

    # Reuse the v11 personalized loader string so we can inherit the same split code.
    SPLIT_PROTOCOL: str = "subject_overlap_stratified_calibrated"
    SEGMENTWISE_SPLIT_COUNTS: Tuple[int, int, int, int] = (16, 4, 5, 5)
    PROTOCOL_NAME: str = "subject-overlap stratified calibrated personalized protocol (v10.2)"

    # Architecture defaults lean on the best v10 Optuna architecture, but keep
    # the v11-style personalized calibration pipeline.
    BATCH_SIZE: int = 32
    LR: float = 1.645331778178685e-4
    WEIGHT_DECAY: float = 1.249870217168137e-4
    GRAD_CLIP_NORM: float = 2.6738035892876946
    WARMUP_EPOCHS: int = 8
    WARMUP_START_FACTOR: float = 0.4558245317068895
    MIN_LR_RATIO: float = 0.08308980248526492
    EMA_DECAY: float = 0.9952669450712013
    EMBED_DIM: int = 160
    EXPERT_DIM: int = 192
    PATCH_SIZE: int = 25
    PATCH_STRIDE: int = 16
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    TOPK_EXPERTS: int = 2
    ROUTER_DENSE_BLEND: float = 0.043558677744173324

    # Personalized / few-shot oriented regularization.
    USE_SUBJECT_CALIBRATION: bool = True
    CALIBRATION_SHRINKAGE: float = 4.0
    FEW_SHOT_SWEEP: Tuple[int, ...] = (0, 1, 2, 3, 5)
    MODEL_SELECTION_USE_CALIBRATED_VAL: bool = True
    VAL_CAL_SCORE_BLEND: float = 0.65

    MODALITY_DROPOUT_PPG: float = 0.08
    MODALITY_DROPOUT_ECG: float = 0.03
    P_DEGRADE: float = 0.12
    AUG_RAMP_EPOCHS: int = 10
    AUG_WARMUP_FACTOR: float = 0.45

    LAMBDA_Q: float = 0.015
    LAMBDA_ROUTER: float = 0.015
    LAMBDA_BAL: float = 0.010
    LAMBDA_PPG_AUX: float = 0.020
    LAMBDA_ECG_AUX: float = 0.040
    LAMBDA_CRED: float = 0.080
    LAMBDA_REG_CLS: float = 0.380
    LAMBDA_TAIL: float = 0.220
    LAMBDA_MAP: float = 0.260
    LAMBDA_NLL: float = 0.100

    REG_USE_WEIGHTED_SAMPLER: bool = True
    REG_SAMPLER_POWER: float = 0.72
    CREDIBILITY_SMOOTHING: float = 0.05
    FUSED_CREDIBILITY_PRIOR: float = 0.62
    FUSED_CREDIBILITY_FLOOR: float = 0.18


build_loaders = v11.build_loaders


def protocol_manifest(cfg: CFG) -> dict:
    if cfg.SPLIT_PROTOCOL == "official_subjectwise":
        return {
            "split_protocol": cfg.SPLIT_PROTOCOL,
            "protocol_name": "official MIMIC-BP calibration-free subject-wise benchmark (via v10.2)",
            "protocol_seed": int(cfg.PROTOCOL_SEED),
            "segmentwise_split_counts": None,
            "subject_pool": cfg.SUBJECT_POOL,
            "warning": (
                "This protocol is the official subject-wise calibration-free benchmark. "
                "It is directly comparable to the official MIMIC-BP dataset paper."
            ),
        }
    return v11.protocol_manifest(cfg)
