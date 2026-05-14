from __future__ import annotations

import json
from pathlib import Path

import train_aqm_medfuse_mimic_bp_reg_v10_13_subjectdisjoint_piso_uncertainty_moe_tailaware_protocol as v13


FINAL_OUTPUT_NAME = "mimic_bp_reg_v10_14_subjectdisjoint_piso_uncertainty_moe_fast_baselineaware_proto"
FINAL_PROTOCOL_ID = "v10.14_subjectdisjoint_piso_uncertainty_moe_fast_baselineaware"

# v10.14 intentionally reuses the finished v10.13 full-train backbones.
OPTLONG_FULLTRAIN_OUTPUT = "mimic_bp_reg_v10_13_opt_long_fulltrain_proto"
DUALMAX_FULLTRAIN_OUTPUT = "mimic_bp_reg_v10_13_optlong_stageaware_dualmax_fulltrain_proto"

_ORIG_V13_BUILD_NEXTGEN_CFG = v13.build_nextgen_cfg
_ORIG_V13_GENERATE_EXTRA_OUTPUTS = v13.generate_extra_outputs


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _output_dir() -> Path:
    return _project_root() / "outputs" / FINAL_OUTPUT_NAME


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_nextgen_cfg():
    cfg = _ORIG_V13_BUILD_NEXTGEN_CFG()
    cfg.OUTPUT_NAME = FINAL_OUTPUT_NAME
    cfg.PROTOCOL_ID = FINAL_PROTOCOL_ID
    cfg.PROTOCOL_NAME = (
        "v10.14 subject-disjoint PiSO-inspired uncertainty-MoE fast baseline-aware protocol "
        "(v10.13 backbone reuse + budgeted reliability calibration + fast crisis-tail fusion)"
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

    # Keep the head trainable, but reduce long late-stage wall time.
    cfg.HEAD_EPOCHS = min(int(getattr(cfg, "HEAD_EPOCHS", 176)), 96)
    cfg.HEAD_MIN_EPOCHS = min(int(getattr(cfg, "HEAD_MIN_EPOCHS", 64)), 40)
    cfg.HEAD_PATIENCE = min(int(getattr(cfg, "HEAD_PATIENCE", 56)), 18)
    cfg.HEAD_SELECTION_RANK_MODE = "clean_acc_f1_then_score"
    cfg.HEAD_CLEAN_ACC_WEIGHT = 2.10
    cfg.HEAD_CLEAN_F1_WEIGHT = 1.55
    cfg.HEAD_CLEAN_BALANCED_WEIGHT = 0.45
    cfg.HEAD_CLEAN_ROBUST_WEIGHT = 0.08
    cfg.HEAD_CLASS_WEIGHT_POWER = 0.60
    cfg.HEAD_ELEVATED_REPEAT = 2
    cfg.HEAD_STAGE1_REPEAT = 3
    cfg.HEAD_STAGE2_REPEAT = 4

    # Baseline-aware objectives: do not select candidates below v10.11 unless no
    # feasible candidate exists, while still pushing crisis SBP toward -5 mmHg.
    cfg.BASELINE_AWARE_SELECTOR_ENABLE = True
    cfg.BASELINE_ACC_SHORTFALL_WEIGHT = 26.0
    cfg.BASELINE_F1_SHORTFALL_WEIGHT = 30.0
    cfg.BASELINE_ASPIRATIONAL_ACC = 0.86
    cfg.BASELINE_ASPIRATIONAL_F1 = 0.80
    cfg.BASELINE_ASPIRATIONAL_ACC_WEIGHT = 3.20
    cfg.BASELINE_ASPIRATIONAL_F1_WEIGHT = 3.60
    cfg.BASELINE_CRISIS_SBP_TARGET = -5.0
    cfg.BASELINE_CRISIS_DBP_TARGET = -2.5
    cfg.BASELINE_HIGH_SBP_TARGET = -4.5
    cfg.BASELINE_HIGH_DBP_TARGET = -2.5
    cfg.BASELINE_CRISIS_TARGET_WEIGHT = 4.20
    cfg.BASELINE_HIGH_TARGET_WEIGHT = 1.70

    # Budgeted reliability/high-bias search. This cuts the v10.13 grid from
    # hundreds of thousands of candidates to a few hundred.
    cfg.RELIABILITY_BIAS_SCALES = (0.55, 0.72, 0.90)
    cfg.RELIABILITY_BIAS_BETAS = (0.85,)
    cfg.RELIABILITY_BIAS_RELIABILITY_FLOORS = (0.15, 0.28)
    cfg.RELIABILITY_BIAS_DISAGREE_GAINS = (0.35, 0.70)
    cfg.RELIABILITY_BIAS_HIGH_GAINS = (0.85,)
    cfg.RELIABILITY_BIAS_CRISIS_GAINS = (1.45, 2.10)
    cfg.RELIABILITY_BIAS_NEGATIVE_FRACS = (0.06, 0.12)
    cfg.RELIABILITY_BIAS_HIGH_THRESHOLDS = (0.16, 0.24)
    cfg.RELIABILITY_BIAS_CRISIS_THRESHOLDS = (0.03, 0.07)
    cfg.RELIABILITY_BIAS_HIGH_FLOOR_SBP = (1.5, 3.0)
    cfg.RELIABILITY_BIAS_CRISIS_FLOOR_SBP = (4.5, 7.0)
    cfg.RELIABILITY_BIAS_MAX_MAE_DELTA = 0.26
    cfg.RELIABILITY_BIAS_MAX_COVERAGE_GAP_DELTA = 0.055

    # Fast crisis-tail fusion. v10.13 used a very large Cartesian grid; this
    # keeps only clinically meaningful operating points.
    cfg.CRISIS_TAIL_FUSION_HIGH_THRESHOLDS = (0.10, 0.16)
    cfg.CRISIS_TAIL_FUSION_CRISIS_THRESHOLDS = (0.03, 0.07)
    cfg.CRISIS_TAIL_FUSION_GAMMAS = (0.90,)
    cfg.CRISIS_TAIL_FUSION_SBP_QUANTILES = (0.95, 1.00)
    cfg.CRISIS_TAIL_FUSION_DBP_QUANTILES = (0.88, 1.00)
    cfg.CRISIS_TAIL_FUSION_CRISIS_GAINS = (2.50, 3.40)
    cfg.CRISIS_TAIL_FUSION_SBP_MARGINS = (5.5, 8.5)
    cfg.CRISIS_TAIL_FUSION_DBP_MARGINS = (2.0, 3.2)
    cfg.CRISIS_TAIL_FUSION_UNCERTAINTY_GAINS = (0.35,)
    cfg.CRISIS_TAIL_FUSION_MODEL_SCALES = (1.15,)
    cfg.CRISIS_TAIL_FUSION_EXPERT_GAINS = (1.00,)
    cfg.CRISIS_TAIL_FUSION_MAX_MAE_DELTA = 0.36
    cfg.CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA = 0.06
    cfg.CRISIS_TAIL_MAX_SHIFT_SBP = 24.0
    cfg.CRISIS_TAIL_MAX_SHIFT_DBP = 11.0
    cfg.CRISIS_TAIL_HARD_FLOOR_SBP = 11.0
    cfg.CRISIS_TAIL_HARD_FLOOR_DBP = 4.0
    cfg.CRISIS_TAIL_UNDEREST_WEIGHT_SBP = 9.00
    cfg.CRISIS_TAIL_UNDEREST_WEIGHT_DBP = 3.30
    cfg.CRISIS_TAIL_SURROGATE_QUANTILES = (0.90, 0.95, 0.98)

    # Keep safety fusion but make the grid small.
    cfg.SAFETY_EVIDENTIAL_SCALES = (0.16, 0.32, 0.50)
    cfg.SAFETY_EVIDENTIAL_BETAS = (0.85,)
    cfg.SAFETY_EVIDENTIAL_DISAGREE_GAINS = (0.35, 0.75)
    cfg.SAFETY_EVIDENTIAL_HIGH_GAINS = (0.55, 0.95)
    cfg.SAFETY_EVIDENTIAL_CRISIS_GAINS = (0.95, 1.50)
    cfg.SAFETY_EVIDENTIAL_RELIABILITY_FLOORS = (0.12,)
    cfg.SAFETY_EVIDENTIAL_STAGE1_BIASES = (0.00, 0.06)
    cfg.SAFETY_EVIDENTIAL_STAGE2_BIASES = (0.08, 0.16)
    cfg.SAFETY_CLASS_FUSION_SCALES = (0.18, 0.36, 0.58)
    cfg.SAFETY_CLASS_FUSION_BETAS = (0.85,)
    cfg.SAFETY_CLASS_FUSION_DISAGREE_GAINS = (1.20, 1.70)
    cfg.SAFETY_CLASS_FUSION_HIGH_GAINS = (1.20, 1.70)
    cfg.SAFETY_CLASS_FUSION_CRISIS_GAINS = (1.35, 1.95)
    cfg.SAFETY_CLASS_FUSION_MAX_WEIGHT = 0.76

    # Reduce expensive late robustness loops while preserving the same output
    # file types and figures.
    cfg.NOISE_STDS = (0.00, 0.10, 0.30)
    cfg.MISSING_PROBS = (0.00, 0.50, 1.00)
    cfg.CONFORMAL_ALPHAS = (0.05, 0.10, 0.20)
    cfg.BOOTSTRAP_SAMPLES = min(int(getattr(cfg, "BOOTSTRAP_SAMPLES", 200)), 80)

    cfg.META_BLEND_WEIGHTS = (0.25, 0.50, 0.70, 0.85, 1.00)
    cfg.CLS_ARBITER_SCALES = (0.45, 0.65, 0.85)
    cfg.CLS_ARBITER_BETAS = (0.95, 1.35)
    cfg.CLS_ARBITER_FLOORS = (0.00, 0.03)
    cfg.CLS_ARBITER_AGREE_SHRINKS = (0.25, 0.55)
    cfg.REG_ROUTER_BLEND_SCALES = (0.40, 0.70, 1.00)
    cfg.REG_ROUTER_TEMPS = (0.70, 1.10, 1.40)
    cfg.REG_ROUTER_GAMMAS = (0.60, 1.00)
    cfg.REG_ROUTER_FLOORS = (0.00, 0.05)

    cfg.CRISIS_SBP_GUARD_ENABLE = True
    cfg.CRISIS_SBP_GUARD_TRIGGER = 0.48
    cfg.CRISIS_SBP_GUARD_QUANTILE = 0.98
    cfg.CRISIS_SBP_GUARD_ABSOLUTE_FLOOR = 171.0
    cfg.CRISIS_SBP_GUARD_GAIN = 0.82
    cfg.CRISIS_SBP_GUARD_MAX_EXTRA_SHIFT = 16.0
    cfg.CRISIS_DBP_GUARD_ABSOLUTE_FLOOR = 105.0
    cfg.CRISIS_DBP_GUARD_GAIN = 0.42
    cfg.CRISIS_DBP_GUARD_MAX_EXTRA_SHIFT = 7.0

    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    return cfg


def generate_extra_outputs(output_dir: Path) -> None:
    _ORIG_V13_GENERATE_EXTRA_OUTPUTS(output_dir)
    budget = {
        "protocol_id": FINAL_PROTOCOL_ID,
        "runtime_design": {
            "backbones": "reuse v10.13 opt-long and dualmax best_model.pt checkpoints",
            "high_bias_grid_candidates": 768,
            "crisis_tail_grid_candidates": 128,
            "safety_grid": "budgeted reliability/evidential fusion",
            "robustness_rows_reduced": True,
        },
        "algorithmic_notes": [
            "Baseline-aware final selection penalizes candidates below v10.11 Acc/F1 and candidates worsening high/crisis bias.",
            "Fast crisis-tail fusion keeps only clinically meaningful quantile/gain/margin operating points instead of an exhaustive Cartesian grid.",
            "The uncertainty/reliability calibration follows recent reliability-aware fusion and conformal-calibration practice while preserving the v10.11 output artifact schema.",
        ],
    }
    _write_json(Path(output_dir) / "v10_14_fast_runtime_audit.json", budget)


def main() -> None:
    originals = {
        "final_output": v13.FINAL_OUTPUT_NAME,
        "protocol_id": v13.FINAL_PROTOCOL_ID,
        "optlong": v13.OPTLONG_FULLTRAIN_OUTPUT,
        "dualmax": v13.DUALMAX_FULLTRAIN_OUTPUT,
        "build_nextgen": v13.build_nextgen_cfg,
        "generate_extra": v13.generate_extra_outputs,
    }
    try:
        v13.FINAL_OUTPUT_NAME = FINAL_OUTPUT_NAME
        v13.FINAL_PROTOCOL_ID = FINAL_PROTOCOL_ID
        v13.OPTLONG_FULLTRAIN_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
        v13.DUALMAX_FULLTRAIN_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT
        v13.build_nextgen_cfg = build_nextgen_cfg
        v13.generate_extra_outputs = generate_extra_outputs
        v13.main()
    finally:
        v13.FINAL_OUTPUT_NAME = originals["final_output"]
        v13.FINAL_PROTOCOL_ID = originals["protocol_id"]
        v13.OPTLONG_FULLTRAIN_OUTPUT = originals["optlong"]
        v13.DUALMAX_FULLTRAIN_OUTPUT = originals["dualmax"]
        v13.build_nextgen_cfg = originals["build_nextgen"]
        v13.generate_extra_outputs = originals["generate_extra"]


if __name__ == "__main__":
    main()
