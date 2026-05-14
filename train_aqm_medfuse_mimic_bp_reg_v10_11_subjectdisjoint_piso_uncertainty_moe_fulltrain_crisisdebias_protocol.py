from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List

import numpy as np
import torch
import gc
import random

import aqm_bp_shared_v9 as shared_v9
import aqm_bp_shared_v10_2_multi_protocol as multi_protocol
import train_aqm_medfuse_mimic_bp_reg_v10_2_common as common_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_dualmax_bridgeguided_refinement_protocol as guided_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_dualanchor_adaptive_gate_meta_protocol as adaptive_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_dualanchor_meta_stack_protocol as meta_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_dualanchor_resume_tailcal_protocol as resume_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_dualbackbone_bridge_protocol as bridge_script
import train_aqm_medfuse_mimic_bp_reg_v10_2_optlong_stageaware_dualmax_protocol as stage_dual_script
import train_aqm_medfuse_mimic_bp_reg_v10_10_subjectdisjoint_piso_uncertainty_moe_highbiascal_protocol as v1010_script
import train_aqm_medfuse_mimic_bp_reg_v10_3_subjectdisjoint_piso_uncertainty_moe_protocol as nextgen_script


FINAL_OUTPUT_NAME = "mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_proto"
FINAL_PROTOCOL_ID = "v10.11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias"
OPTLONG_FULLTRAIN_OUTPUT = "mimic_bp_reg_v10_11_opt_long_fulltrain_proto"
DUALMAX_FULLTRAIN_OUTPUT = "mimic_bp_reg_v10_11_optlong_stageaware_dualmax_fulltrain_proto"
TARGET_ACC = 0.90
TARGET_MACRO_F1 = 0.80
TARGET_ELEVATED_F1 = 0.68
TARGET_STAGE1_F1 = 0.74
TARGET_STAGE2_F1 = 0.84
TARGET_MIN_ROBUST_F1 = 0.56
FULLTRAIN_SAFE_BATCH_SIZE = 8
DUALMAX_SAFE_BATCH_SIZE = 4
MIN_FULLTRAIN_EPOCH_LOG_ROWS = 8
RESUME_INCOMPLETE_FULLTRAIN = False

_CRISIS_DEBIAS_REGISTRY: Dict[str, dict] = {}


def _base_optlong_ckpt(project_root: Path) -> Path:
    return project_root / "outputs" / "mimic_bp_reg_v10_2_opt_long_proto" / "best_model.pt"


def _base_dualmax_ckpt(project_root: Path) -> Path:
    return project_root / "outputs" / "mimic_bp_reg_v10_2_optlong_stageaware_dualmax_proto" / "best_model.pt"


def _fulltrain_optlong_ckpt(project_root: Path) -> Path:
    return project_root / "outputs" / OPTLONG_FULLTRAIN_OUTPUT / "best_model.pt"


def _fulltrain_dualmax_ckpt(project_root: Path) -> Path:
    return project_root / "outputs" / DUALMAX_FULLTRAIN_OUTPUT / "best_model.pt"


def _preferred_optlong_ckpt(project_root: Path) -> Path:
    fulltrain_ckpt = _fulltrain_optlong_ckpt(project_root)
    return _select_operational_checkpoint(fulltrain_ckpt, _base_optlong_ckpt(project_root), "opt-long")


def _preferred_dualmax_ckpt(project_root: Path) -> Path:
    fulltrain_ckpt = _fulltrain_dualmax_ckpt(project_root)
    return _select_operational_checkpoint(fulltrain_ckpt, _base_dualmax_ckpt(project_root), "dualmax")


def _cfg_project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_data_root(project_root: Path, raw_root: str) -> Path:
    return nextgen_script.resolve_data_root(project_root, raw_root)


def _checkpoint_artifact_complete(ckpt_path: Path) -> bool:
    if not ckpt_path.exists():
        return False
    out_dir = ckpt_path.parent
    epoch_log = out_dir / "epoch_log.csv"
    if not epoch_log.exists():
        return False
    try:
        with epoch_log.open("r", encoding="utf-8") as f:
            row_count = max(0, sum(1 for _ in f) - 1)
        return row_count >= int(MIN_FULLTRAIN_EPOCH_LOG_ROWS)
    except OSError:
        return False


def _select_operational_checkpoint(candidate_ckpt: Path, fallback_ckpt: Path, label: str) -> Path:
    if _checkpoint_artifact_complete(candidate_ckpt):
        return candidate_ckpt
    if candidate_ckpt != fallback_ckpt and candidate_ckpt.exists():
        print(
            f"Using base {label} checkpoint for the main v10.11 protocol because the cached "
            f"full-train artifact set is incomplete under {candidate_ckpt.parent} "
            "(expected companion epoch_log.csv)."
        )
    return fallback_ckpt


def _should_continue_incomplete_fulltrain(ckpt_path: Path) -> bool:
    if not bool(globals().get("RESUME_INCOMPLETE_FULLTRAIN", False)):
        return False
    if not ckpt_path.exists():
        return False
    epoch_log = ckpt_path.parent / "epoch_log.csv"
    if not epoch_log.exists():
        return True
    try:
        with epoch_log.open("r", encoding="utf-8") as f:
            row_count = max(0, sum(1 for _ in f) - 1)
        return row_count < int(MIN_FULLTRAIN_EPOCH_LOG_ROWS)
    except OSError:
        return True


def build_nextgen_cfg():
    cfg = v1010_script.build_nextgen_cfg()
    project_root = Path(cfg.PROJECT_ROOT)

    cfg.OUTPUT_NAME = FINAL_OUTPUT_NAME
    cfg.PROTOCOL_ID = FINAL_PROTOCOL_ID
    cfg.PROTOCOL_NAME = (
        "v10.11 subject-disjoint PiSO-inspired uncertainty-MoE full-train crisis-debias protocol "
        "(v10.10 high-bias/crisis-tail stack + warm-start backbone retraining + "
        "Acc/F1-targeted head search + extreme-tail debias fusion)"
    )

    cfg.WARMSTART_REG_PATH = str(_preferred_optlong_ckpt(project_root))
    cfg.WARMSTART_HEAD_PATH = str(cfg.HEAD_RESUME_PATH)
    cfg.WARMSTART_CANDIDATES = (
        _preferred_optlong_ckpt(project_root).parent.name,
        _preferred_dualmax_ckpt(project_root).parent.name,
        "mimic_bp_reg_v10_10_subjectdisjoint_piso_uncertainty_moe_highbiascal_crisistailfusion_proto",
    )

    cfg.FULLTRAIN_OPTLONG_OUTPUT = OPTLONG_FULLTRAIN_OUTPUT
    cfg.FULLTRAIN_DUALMAX_OUTPUT = DUALMAX_FULLTRAIN_OUTPUT
    cfg.TARGET_ACC = TARGET_ACC
    cfg.TARGET_MACRO_F1 = TARGET_MACRO_F1
    cfg.TARGET_ELEVATED_F1 = TARGET_ELEVATED_F1
    cfg.TARGET_STAGE1_F1 = TARGET_STAGE1_F1
    cfg.TARGET_STAGE2_F1 = TARGET_STAGE2_F1

    cfg.HEAD_EPOCHS = 144
    cfg.HEAD_PATIENCE = 40
    cfg.HEAD_MIN_EPOCHS = 48
    cfg.HEAD_LR = 3.2e-5
    cfg.HEAD_MIN_LR = 5.0e-7
    cfg.HEAD_SELECTION_MODE = "target_accf1_then_score"
    cfg.HEAD_MISSING_ECG = max(float(cfg.HEAD_MISSING_ECG), 0.42)
    cfg.HEAD_MISSING_PPG = max(float(cfg.HEAD_MISSING_PPG), 0.34)
    cfg.HEAD_NOISE_STD = max(float(cfg.HEAD_NOISE_STD), 0.06)
    cfg.HEAD_CLASS_WEIGHT_POWER = 0.68
    cfg.HEAD_ELEVATED_REPEAT = 2
    cfg.HEAD_STAGE1_REPEAT = 2
    cfg.HEAD_STAGE2_REPEAT = 3
    cfg.HEAD_TARGET_RARE_MIN_WEIGHT = 0.46
    cfg.HEAD_TARGET_STAGE2_WEIGHT = 0.24
    cfg.HEAD_TARGET_GAP_WEIGHT = 1.25
    cfg.HEAD_TARGET_ROBUST_GAP_WEIGHT = 0.55
    cfg.HEAD_ROBUST_NOISE_WEIGHT = 0.10
    cfg.HEAD_ROBUST_ECG_WEIGHT = 0.15
    cfg.HEAD_ROBUST_PPG_WEIGHT = 0.22
    cfg.HEAD_ROBUST_MIN_WEIGHT = 0.18
    cfg.HEAD_KD_WEIGHT = max(float(getattr(cfg, "HEAD_KD_WEIGHT", 0.25)), 0.42)
    cfg.HEAD_BP_WEIGHT = min(float(getattr(cfg, "HEAD_BP_WEIGHT", 0.20)), 0.16)

    cfg.META_BLEND_WEIGHTS = (0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90, 1.00)
    cfg.CLS_ARBITER_SCALES = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
    cfg.CLS_ARBITER_BETAS = (0.55, 0.75, 0.95, 1.10, 1.30, 1.55, 1.80)
    cfg.CLS_ARBITER_FLOORS = (0.00, 0.01, 0.02, 0.04, 0.06)
    cfg.CLS_ARBITER_AGREE_SHRINKS = (0.12, 0.25, 0.40, 0.55, 0.70, 0.85)

    cfg.REG_ROUTER_BLEND_SCALES = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
    cfg.REG_ROUTER_TEMPS = (0.35, 0.50, 0.70, 0.90, 1.10, 1.30, 1.50)
    cfg.REG_ROUTER_GAMMAS = (0.45, 0.60, 0.80, 1.00, 1.20, 1.40)
    cfg.REG_ROUTER_FLOORS = (0.00, 0.01, 0.03, 0.06)

    cfg.SAFETY_CLASS_FUSION_SCALES = (0.10, 0.18, 0.28, 0.40, 0.55, 0.70, 0.82)
    cfg.SAFETY_CLASS_FUSION_BETAS = (0.45, 0.65, 0.85, 1.05, 1.25)
    cfg.SAFETY_CLASS_FUSION_DISAGREE_GAINS = (1.00, 1.20, 1.45, 1.70, 1.95, 2.20)
    cfg.SAFETY_CLASS_FUSION_HIGH_GAINS = (1.00, 1.25, 1.50, 1.80, 2.10)
    cfg.SAFETY_CLASS_FUSION_CRISIS_GAINS = (1.00, 1.35, 1.70, 2.05, 2.40)
    cfg.SAFETY_CLASS_FUSION_MAX_WEIGHT = 0.72
    cfg.SAFETY_CLASS_FUSION_STAGE1_RECALL_WEIGHT = 0.14
    cfg.SAFETY_CLASS_FUSION_STAGE2_RECALL_WEIGHT = 0.30
    cfg.SAFETY_CLASS_FUSION_STAGE2_F1_WEIGHT = 0.12
    cfg.CLS_SCORE_NOISE_WEIGHT = 0.10
    cfg.CLS_SCORE_ECG_WEIGHT = 0.15
    cfg.CLS_SCORE_PPG_WEIGHT = 0.22
    cfg.CLS_SCORE_MINROBUST_WEIGHT = 0.18

    cfg.HIGH_BIAS_CAL_HIGH_THRESHOLDS = (0.22, 0.32, 0.42)
    cfg.HIGH_BIAS_CAL_CRISIS_THRESHOLDS = (0.08, 0.14, 0.20)
    cfg.HIGH_BIAS_CAL_GAMMAS = (0.55, 0.75, 0.95, 1.15)
    cfg.HIGH_BIAS_CAL_SBP_HIGH_SHIFTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    cfg.HIGH_BIAS_CAL_SBP_CRISIS_SHIFTS = (2.5, 4.0, 5.5, 7.0, 8.5)
    cfg.HIGH_BIAS_CAL_DBP_HIGH_SHIFTS = (0.25, 0.5, 0.75, 1.0, 1.25)
    cfg.HIGH_BIAS_CAL_DBP_CRISIS_SHIFTS = (1.0, 1.5, 2.0, 2.75, 3.5, 4.25)
    cfg.HIGH_BIAS_CAL_MAX_MAE_DELTA = 0.24
    cfg.HIGH_BIAS_CAL_MAX_COVERAGE_GAP_DELTA = 0.05

    cfg.CRISIS_TAIL_FUSION_HIGH_THRESHOLDS = (0.12, 0.20)
    cfg.CRISIS_TAIL_FUSION_CRISIS_THRESHOLDS = (0.03, 0.07, 0.12)
    cfg.CRISIS_TAIL_FUSION_GAMMAS = (0.80, 1.10)
    cfg.CRISIS_TAIL_FUSION_SBP_QUANTILES = (0.90, 1.00)
    cfg.CRISIS_TAIL_FUSION_DBP_QUANTILES = (0.85, 1.00)
    cfg.CRISIS_TAIL_FUSION_CRISIS_GAINS = (2.20, 3.00, 3.80)
    cfg.CRISIS_TAIL_FUSION_SBP_MARGINS = (3.0, 5.0, 7.5, 10.0)
    cfg.CRISIS_TAIL_FUSION_DBP_MARGINS = (1.5, 2.5, 3.5)
    cfg.CRISIS_TAIL_FUSION_UNCERTAINTY_GAINS = (0.25, 0.60)
    cfg.CRISIS_TAIL_FUSION_MODEL_SCALES = (0.85, 1.15, 1.45)
    cfg.CRISIS_TAIL_FUSION_EXPERT_GAINS = (0.70, 1.00, 1.30)
    cfg.CRISIS_TAIL_FUSION_MAX_MAE_DELTA = 0.30
    cfg.CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA = 0.06
    cfg.CRISIS_TAIL_MAX_SHIFT_SBP = 22.0
    cfg.CRISIS_TAIL_MAX_SHIFT_DBP = 12.0
    cfg.CRISIS_TAIL_SURROGATE_QUANTILES = (0.90, 0.95, 0.98)
    cfg.CRISIS_TAIL_CLASS_STAGE1_GAIN = 0.85
    cfg.CRISIS_TAIL_CLASS_STAGE2_GAIN = 1.85
    cfg.CRISIS_TAIL_DISAGREEMENT_GAIN = 0.85
    cfg.CRISIS_TAIL_HARD_FLOOR_SBP = 8.5
    cfg.CRISIS_TAIL_HARD_FLOOR_DBP = 3.5
    cfg.CRISIS_TAIL_UNDEREST_WEIGHT_SBP = 5.10
    cfg.CRISIS_TAIL_UNDEREST_WEIGHT_DBP = 2.20
    return cfg


def build_optlong_fulltrain_cfg():
    cfg = multi_protocol.make_v10_2_optimized_long_cfg()
    try:
        cfg.DATA_ROOT = str(_resolve_data_root(Path(cfg.PROJECT_ROOT), str(cfg.DATA_ROOT)))
    except FileNotFoundError:
        cfg.DATA_ROOT = str(cfg.DATA_ROOT)
    cfg.OUTPUT_NAME = OPTLONG_FULLTRAIN_OUTPUT
    cfg.PROTOCOL_ID = "v10.11_optlong_fulltrain"
    cfg.PROTOCOL_NAME = "v10.11 opt-long warm-start full retraining"
    cfg.EPOCHS = 96
    cfg.EARLY_STOPPING_PATIENCE = 32
    cfg.LR = 3.2e-5
    cfg.MIN_LR_RATIO = 0.01
    cfg.BATCH_SIZE = min(int(getattr(cfg, "BATCH_SIZE", 32)), FULLTRAIN_SAFE_BATCH_SIZE)
    cfg.NUM_WORKERS = 0
    cfg.USE_EMA = False
    cfg.TRAIN_STAGE_NAME = "OptLong Backbone"
    cfg.INIT_CKPT_PATH = str(_base_optlong_ckpt(Path(cfg.PROJECT_ROOT)))
    cfg.PLOT_COMPOSITE_TRAINING_CURVES = False
    cfg.SAVE_SPLIT_TRAINING_CURVES = True
    return cfg


def build_dualmax_fulltrain_cfg(base_builder=None):
    if base_builder is None:
        base_builder = stage_dual_script.build_stageaware_cfg
    cfg = base_builder()
    try:
        cfg.DATA_ROOT = str(_resolve_data_root(Path(cfg.PROJECT_ROOT), str(cfg.DATA_ROOT)))
    except FileNotFoundError:
        cfg.DATA_ROOT = str(cfg.DATA_ROOT)
    cfg.OUTPUT_NAME = DUALMAX_FULLTRAIN_OUTPUT
    cfg.PROTOCOL_ID = "v10.11_optlong_stageaware_dualmax_fulltrain"
    cfg.PROTOCOL_NAME = "v10.11 stage-aware dualmax warm-start full retraining"
    cfg.EPOCHS = 128
    cfg.EARLY_STOPPING_PATIENCE = max(int(cfg.EPOCHS), 48)
    cfg.LR = min(float(cfg.LR), 1.6e-5)
    cfg.BATCH_SIZE = min(int(getattr(cfg, "BATCH_SIZE", 32)), DUALMAX_SAFE_BATCH_SIZE)
    cfg.NUM_WORKERS = 0
    cfg.USE_EMA = False
    cfg.TRAIN_STAGE_NAME = "DualMax Backbone"
    cfg.PLOT_COMPOSITE_TRAINING_CURVES = False
    cfg.SAVE_SPLIT_TRAINING_CURVES = True
    cfg.INIT_CKPT_PATH = str(_fulltrain_optlong_ckpt(Path(cfg.PROJECT_ROOT)))
    return cfg


def _is_cuda_runtime_failure(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "cudnn_status_execution_failed",
            "cuda out of memory",
            "out of memory",
            "cudaerrormemoryallocation",
            "cublas_status",
            "cuda error",
            "acceleratorerror",
        )
    )


def _retry_batch_sizes(initial_batch_size: int) -> List[int]:
    ordered = [initial_batch_size, 6, 4, 2, 1]
    out: List[int] = []
    for item in ordered:
        item = max(1, int(item))
        if item not in out:
            out.append(item)
    return out


def _safe_cuda_cleanup():
    gc.collect()
    if not torch.cuda.is_available():
        return
    for fn in (torch.cuda.synchronize, torch.cuda.empty_cache, torch.cuda.ipc_collect):
        try:
            fn()
        except Exception:
            pass


def _safe_seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        return
    except Exception as exc:
        if not _is_cuda_runtime_failure(exc):
            raise
        print(
            "CUDA RNG reseed failed after an earlier GPU memory fault. "
            "Cleaning CUDA state and falling back to a CPU-only RNG reseed for this stage."
        )
        _safe_cuda_cleanup()
        default_gen = getattr(torch, "default_generator", None)
        if default_gen is not None:
            default_gen.manual_seed(seed)
        else:
            torch.random.default_generator.manual_seed(seed)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(seed)
            except Exception:
                print("Warning: continuing without refreshing CUDA RNG state because the device remains memory-constrained.")


def _run_self_subprocess(args: List[str]) -> int:
    default_data_root = str(multi_protocol.CFG().DATA_ROOT)
    try:
        resolved_data_root = str(_resolve_data_root(_cfg_project_root(), default_data_root))
    except FileNotFoundError:
        resolved_data_root = default_data_root
    env = os.environ.copy()
    env["AQM_MIMIC_BP_ROOT"] = resolved_data_root
    env["AQM_MIMIC_BP_DATA_ROOT"] = resolved_data_root
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        cwd=str(_cfg_project_root()),
        env=env,
        check=False,
    )
    return int(proc.returncode)


def _subprocess_optlong_fulltrain_entry(batch_size: int, cudnn_enabled: bool) -> int:
    original_common_seed_everything = common_script.seed_everything
    original_cudnn_enabled = torch.backends.cudnn.enabled
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        cfg = build_optlong_fulltrain_cfg()
        cfg.BATCH_SIZE = max(1, int(batch_size))
        common_script.seed_everything = _safe_seed_everything
        torch.backends.cudnn.enabled = bool(cudnn_enabled)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        _safe_cuda_cleanup()
        common_script.run_regression_experiment(cfg)
        return 0
    except Exception as exc:
        if _is_cuda_runtime_failure(exc):
            print(f"Opt-long full-training subprocess hit a CUDA runtime issue: {exc}")
            return 32
        print(f"Opt-long full-training subprocess failed: {exc}")
        return 1
    finally:
        common_script.seed_everything = original_common_seed_everything
        torch.backends.cudnn.enabled = original_cudnn_enabled
        torch.backends.cudnn.benchmark = original_cudnn_benchmark
        torch.backends.cudnn.deterministic = original_cudnn_deterministic
        _safe_cuda_cleanup()


def _subprocess_stageaware_fulltrain_entry(optlong_ckpt: Path, batch_size: int, cudnn_enabled: bool) -> int:
    original_build_stageaware_cfg = stage_dual_script.build_stageaware_cfg
    original_pick_optlong_checkpoint = stage_dual_script.pick_optlong_checkpoint
    original_stage_seed_everything = stage_dual_script.seed_everything
    original_common_seed_everything = common_script.seed_everything
    original_cudnn_enabled = torch.backends.cudnn.enabled
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        def _build_stageaware_cfg():
            cfg = build_dualmax_fulltrain_cfg(original_build_stageaware_cfg)
            cfg.BATCH_SIZE = max(1, int(batch_size))
            return cfg

        def _pick_optlong_checkpoint(_cfg):
            return Path(optlong_ckpt)

        stage_dual_script.build_stageaware_cfg = _build_stageaware_cfg
        stage_dual_script.pick_optlong_checkpoint = _pick_optlong_checkpoint
        stage_dual_script.seed_everything = _safe_seed_everything
        common_script.seed_everything = _safe_seed_everything
        torch.backends.cudnn.enabled = bool(cudnn_enabled)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        _safe_cuda_cleanup()
        stage_dual_script.main()
        return 0
    except Exception as exc:
        if _is_cuda_runtime_failure(exc):
            print(f"Stage-aware dualmax full-training subprocess hit a CUDA runtime issue: {exc}")
            return 32
        print(f"Stage-aware dualmax full-training subprocess failed: {exc}")
        return 1
    finally:
        stage_dual_script.build_stageaware_cfg = original_build_stageaware_cfg
        stage_dual_script.pick_optlong_checkpoint = original_pick_optlong_checkpoint
        stage_dual_script.seed_everything = original_stage_seed_everything
        common_script.seed_everything = original_common_seed_everything
        torch.backends.cudnn.enabled = original_cudnn_enabled
        torch.backends.cudnn.benchmark = original_cudnn_benchmark
        torch.backends.cudnn.deterministic = original_cudnn_deterministic
        _safe_cuda_cleanup()


def _run_regression_experiment_with_cuda_fallback(cfg_builder, label: str):
    initial_cfg = cfg_builder()
    batch_sizes = _retry_batch_sizes(int(getattr(initial_cfg, "BATCH_SIZE", FULLTRAIN_SAFE_BATCH_SIZE)))
    original_cudnn_enabled = torch.backends.cudnn.enabled
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    last_error_message: str | None = None

    try:
        for attempt_idx, batch_size in enumerate(batch_sizes):
            if attempt_idx >= 2:
                torch.backends.cudnn.enabled = False
            else:
                torch.backends.cudnn.enabled = original_cudnn_enabled
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False

            if attempt_idx > 0:
                print(
                    f"Retrying {label} with safer CUDA settings: "
                    f"batch_size={batch_size}, cudnn_enabled={torch.backends.cudnn.enabled}"
                )
            return_code = _run_self_subprocess(
                [
                    "--run-optlong-fulltrain-subprocess",
                    str(batch_size),
                    "1" if torch.backends.cudnn.enabled else "0",
                ]
            )
            if return_code == 0:
                return True
            if return_code != 32:
                raise RuntimeError(f"{label} subprocess failed with exit code {return_code}")
            last_error_message = f"CUDA runtime failure exit code {return_code}"
            print(
                f"{label} hit a CUDA/cuDNN backward failure. "
                f"Cleaning up CUDA state and retrying with a smaller batch."
            )
            _safe_cuda_cleanup()
    finally:
        torch.backends.cudnn.enabled = original_cudnn_enabled
        torch.backends.cudnn.benchmark = original_cudnn_benchmark
        torch.backends.cudnn.deterministic = original_cudnn_deterministic

    if last_error_message is not None:
        print(f"{label} could not be completed under current GPU memory budget.")
    return False


def _run_stageaware_fulltrain_with_cuda_fallback(optlong_ckpt: Path):
    initial_cfg = build_dualmax_fulltrain_cfg(stage_dual_script.build_stageaware_cfg)
    batch_sizes = _retry_batch_sizes(int(getattr(initial_cfg, "BATCH_SIZE", FULLTRAIN_SAFE_BATCH_SIZE)))
    original_cudnn_enabled = torch.backends.cudnn.enabled
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    last_error_message: str | None = None

    try:
        for attempt_idx, batch_size in enumerate(batch_sizes):
            if attempt_idx >= 2:
                torch.backends.cudnn.enabled = False
            else:
                torch.backends.cudnn.enabled = original_cudnn_enabled
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False

            if attempt_idx > 0:
                print(
                    "Retrying stage-aware dualmax full training with safer CUDA settings: "
                    f"batch_size={batch_size}, cudnn_enabled={torch.backends.cudnn.enabled}"
                )
            return_code = _run_self_subprocess(
                [
                    "--run-stageaware-fulltrain-subprocess",
                    str(optlong_ckpt),
                    str(batch_size),
                    "1" if torch.backends.cudnn.enabled else "0",
                ]
            )
            if return_code == 0:
                return True
            if return_code != 32:
                raise RuntimeError(
                    "Stage-aware dualmax full-training subprocess failed with "
                    f"exit code {return_code}"
                )
            last_error_message = f"CUDA runtime failure exit code {return_code}"
            print(
                "Stage-aware dualmax full training hit a CUDA/cuDNN backward failure. "
                "Cleaning up CUDA state and retrying with a smaller batch."
            )
            _safe_cuda_cleanup()
    finally:
        torch.backends.cudnn.enabled = original_cudnn_enabled
        torch.backends.cudnn.benchmark = original_cudnn_benchmark
        torch.backends.cudnn.deterministic = original_cudnn_deterministic

    if last_error_message is not None:
        print("Stage-aware dualmax full training could not be completed under current GPU memory budget.")
    return False


def train_or_load_full_backbones() -> tuple[Path, Path]:
    project_root = _cfg_project_root()
    fulltrain_optlong_ckpt = _fulltrain_optlong_ckpt(project_root)
    fulltrain_dualmax_ckpt = _fulltrain_dualmax_ckpt(project_root)
    base_optlong_ckpt = _base_optlong_ckpt(project_root)
    base_dualmax_ckpt = _base_dualmax_ckpt(project_root)
    optlong_ckpt = fulltrain_optlong_ckpt
    dualmax_ckpt = fulltrain_dualmax_ckpt

    optlong_needs_training = (not optlong_ckpt.exists()) or _should_continue_incomplete_fulltrain(optlong_ckpt)
    if optlong_needs_training:
        if optlong_ckpt.exists():
            print("Continuing incomplete opt-long backbone training from the latest resume state or cached best checkpoint...")
        else:
            print("Retraining opt-long backbone from warm-start checkpoint...")
        print(
            "Metric note: upcoming backbone logs report regression-derived class metrics "
            "`backbone_f1_from_reg(...)`, comparable only to prior backbone training logs, "
            "not to v10.10 `Resume Head` val_f1."
        )
        optlong_success = _run_regression_experiment_with_cuda_fallback(build_optlong_fulltrain_cfg, "opt-long full training")
        if not optlong_success:
            fallback_optlong = base_optlong_ckpt
            if not fallback_optlong.exists():
                raise FileNotFoundError(f"Fallback opt-long checkpoint not found: {fallback_optlong}")
            print(
                "Falling back to the original opt-long checkpoint because full retraining "
                "exceeded the available GPU memory budget."
            )
            optlong_ckpt = fallback_optlong
    else:
        if _checkpoint_artifact_complete(optlong_ckpt):
            print(f"Using cached full-train opt-long checkpoint: {optlong_ckpt}")
        else:
            print(
                "Cached full-train opt-long checkpoint exists but its artifact set is incomplete; "
                "the main v10.11 protocol will fall back to the validated base checkpoint."
            )

    if not optlong_ckpt.exists():
        raise FileNotFoundError(f"Full-train opt-long checkpoint not found after training: {optlong_ckpt}")
    optlong_ckpt = _select_operational_checkpoint(optlong_ckpt, base_optlong_ckpt, "opt-long")

    dualmax_needs_training = (not dualmax_ckpt.exists()) or _should_continue_incomplete_fulltrain(dualmax_ckpt)
    if dualmax_needs_training:
        if dualmax_ckpt.exists():
            print("Continuing incomplete stage-aware dualmax backbone training from the latest resume state or cached best checkpoint...")
        else:
            print("Retraining stage-aware dualmax backbone from the new opt-long checkpoint...")
        print(
            "Metric note: upcoming dualmax backbone logs still use `backbone_f1_from_reg(...)`; "
            "compare them with backbone-stage logs, not with head-stage classification F1."
        )
        dualmax_success = _run_stageaware_fulltrain_with_cuda_fallback(optlong_ckpt)
        if not dualmax_success:
            fallback_dualmax = base_dualmax_ckpt
            if not fallback_dualmax.exists():
                raise FileNotFoundError(f"Fallback dualmax checkpoint not found: {fallback_dualmax}")
            print(
                "Falling back to the original stage-aware dualmax checkpoint because full retraining "
                "exceeded the available GPU memory budget."
            )
            dualmax_ckpt = fallback_dualmax
    else:
        if _checkpoint_artifact_complete(dualmax_ckpt):
            print(f"Using cached full-train dualmax checkpoint: {dualmax_ckpt}")
        else:
            print(
                "Cached full-train dualmax checkpoint exists but its artifact set is incomplete; "
                "the main v10.11 protocol will fall back to the validated base checkpoint."
            )

    if not dualmax_ckpt.exists():
        raise FileNotFoundError(f"Full-train dualmax checkpoint not found after training: {dualmax_ckpt}")
    dualmax_ckpt = _select_operational_checkpoint(dualmax_ckpt, base_dualmax_ckpt, "dualmax")

    _safe_cuda_cleanup()
    return optlong_ckpt, dualmax_ckpt


def _target_class_summary(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    elevated = float(metrics.get(f"cls_f1_{prefix}_Elevated", 0.0))
    stage1 = float(metrics.get(f"cls_f1_{prefix}_Stage1", 0.0))
    stage2 = float(metrics.get(f"cls_f1_{prefix}_Stage2", 0.0))
    return {
        "acc": float(metrics[f"cls_acc_{prefix}"]),
        "macro_f1": float(metrics[f"cls_f1_macro_{prefix}"]),
        "balanced_acc": float(metrics[f"cls_balanced_acc_{prefix}"]),
        "ece": float(metrics.get(f"cls_ece_{prefix}", 0.0)),
        "elevated_f1": elevated,
        "stage1_f1": stage1,
        "stage2_f1": stage2,
        "rare_f1_mean": float(np.mean([elevated, stage1, stage2])),
        "rare_f1_min": float(np.min([elevated, stage1, stage2])),
    }


def _target_gap_components(summary: Dict[str, float]) -> Dict[str, float]:
    rare_min = min(summary["elevated_f1"], summary["stage1_f1"], summary["stage2_f1"])
    return {
        "acc_gap": max(0.0, float(TARGET_ACC) - summary["acc"]),
        "f1_gap": max(0.0, float(TARGET_MACRO_F1) - summary["macro_f1"]),
        "elevated_gap": max(0.0, float(TARGET_ELEVATED_F1) - summary["elevated_f1"]),
        "stage1_gap": max(0.0, float(TARGET_STAGE1_F1) - summary["stage1_f1"]),
        "stage2_gap": max(0.0, float(TARGET_STAGE2_F1) - summary["stage2_f1"]),
        "robust_gap": max(0.0, float(TARGET_MIN_ROBUST_F1) - rare_min),
    }


def targeted_classification_selection_score(metrics: Dict[str, float], prefix: str) -> float:
    summary = _target_class_summary(metrics, prefix)
    gaps = _target_gap_components(summary)
    return float(
        1.48 * summary["macro_f1"]
        + 1.30 * summary["acc"]
        + 0.90 * summary["balanced_acc"]
        + 0.72 * summary["rare_f1_mean"]
        + 0.40 * summary["rare_f1_min"]
        + 0.16 * summary["stage2_f1"]
        - 0.10 * summary["ece"]
        - 1.28 * gaps["acc_gap"]
        - 1.58 * gaps["f1_gap"]
        - 0.72 * gaps["elevated_gap"]
        - 0.76 * gaps["stage1_gap"]
        - 0.62 * gaps["stage2_gap"]
        - 0.30 * gaps["robust_gap"]
    )


def targeted_classification_candidate_score(metrics: Dict[str, float], prefix: str) -> float:
    return float(targeted_classification_selection_score(metrics, prefix))


def targeted_robust_classification_score(
    clean_metrics: Dict[str, float],
    noise_metrics: Dict[str, float],
    ecg_metrics: Dict[str, float],
    ppg_metrics: Dict[str, float],
    cfg,
) -> float:
    noise_f1 = float(noise_metrics["cls_f1_macro_selected_noise_val"])
    ecg_f1 = float(ecg_metrics["cls_f1_macro_selected_ecg_val"])
    ppg_f1 = float(ppg_metrics["cls_f1_macro_selected_ppg_val"])
    robust_min = min(noise_f1, ecg_f1, ppg_f1)
    return float(
        targeted_classification_candidate_score(clean_metrics, "selected_val")
        + float(getattr(cfg, "CLS_SCORE_NOISE_WEIGHT", 0.0)) * noise_f1
        + float(getattr(cfg, "CLS_SCORE_ECG_WEIGHT", 0.0)) * ecg_f1
        + float(getattr(cfg, "CLS_SCORE_PPG_WEIGHT", 0.0)) * ppg_f1
        + float(getattr(cfg, "CLS_SCORE_MINROBUST_WEIGHT", 0.0)) * robust_min
    )


def _index_feature_bank(bank: dict, indices: np.ndarray) -> dict:
    idx_np = np.asarray(indices, dtype=np.int64)
    idx_t = torch.as_tensor(idx_np, dtype=torch.long)
    n = int(bank["y"].shape[0])
    out = {}
    for key, value in bank.items():
        if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == n:
            out[key] = value.index_select(0, idx_t.to(value.device))
        elif isinstance(value, np.ndarray) and value.ndim >= 1 and int(value.shape[0]) == n:
            out[key] = value[idx_np]
        else:
            out[key] = value
    return out


def _repeat_feature_bank(bank: dict, repeat_by_class: Dict[int, int]) -> dict:
    y = np.asarray(bank["y"].detach().cpu().numpy(), dtype=np.int64)
    parts = [bank]
    for class_id, repeat in repeat_by_class.items():
        extra_repeats = max(0, int(repeat) - 1)
        if extra_repeats <= 0:
            continue
        cls_idx = np.where(y == int(class_id))[0]
        if cls_idx.size == 0:
            continue
        for _ in range(extra_repeats):
            parts.append(_index_feature_bank(bank, cls_idx))
    if len(parts) == 1:
        return bank
    return guided_script.concat_feature_banks(parts)


def run_feature_head_warmstart_targeted(
    resume_path: Path,
    train_banks: List[dict],
    val_clean_bank: dict,
    val_noise_bank: dict,
    val_ecg_bank: dict,
    val_ppg_bank: dict,
    cfg,
):
    model, state = resume_script.load_feature_head_checkpoint(resume_path, int(train_banks[0]["x"].shape[1]), cfg)

    train_bank_norm = guided_script.concat_feature_banks(
        [guided_script.normalize_bank(bank, state) for bank in train_banks]
    )
    train_bank_norm = _repeat_feature_bank(
        train_bank_norm,
        {
            1: int(getattr(cfg, "HEAD_ELEVATED_REPEAT", 1)),
            2: int(getattr(cfg, "HEAD_STAGE1_REPEAT", 1)),
            3: int(getattr(cfg, "HEAD_STAGE2_REPEAT", 1)),
        },
    )
    val_clean_bank = guided_script.normalize_bank(val_clean_bank, state)
    val_noise_bank = guided_script.normalize_bank(val_noise_bank, state)
    val_ecg_bank = guided_script.normalize_bank(val_ecg_bank, state)
    val_ppg_bank = guided_script.normalize_bank(val_ppg_bank, state)

    train_loader = guided_script.build_train_loader(train_bank_norm, cfg)
    class_counts = torch.bincount(train_bank_norm["y"], minlength=cfg.N_CLASSES).float().clamp_min(1.0)
    class_weight_power = float(getattr(cfg, "HEAD_CLASS_WEIGHT_POWER", 0.5))
    class_weights = (1.0 / torch.pow(class_counts, class_weight_power)).to(cfg.DEVICE)
    class_weights = class_weights / class_weights.mean()

    ord_target_train = guided_script.ordinal_targets(train_bank_norm["y"], cfg.N_CLASSES)
    pos_count = ord_target_train.sum(dim=0).clamp_min(1.0)
    neg_count = (ord_target_train.shape[0] - pos_count).clamp_min(1.0)
    ord_pos_weight = (neg_count / pos_count).to(cfg.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.HEAD_LR),
        weight_decay=float(cfg.HEAD_WEIGHT_DECAY),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(cfg.HEAD_EPOCHS), 1),
        eta_min=float(getattr(cfg, "HEAD_MIN_LR", 1.0e-6)),
    )

    best_state = None
    best_rank = None
    best_epoch = 0
    best_metrics = None
    patience = 0
    epoch_rows: List[dict] = []
    reg_mu = guided_script.state_tensor_to_device(state["reg_mu"], cfg.DEVICE)
    reg_sigma = guided_script.state_tensor_to_device(state["reg_sigma"], cfg.DEVICE)
    min_epochs = int(getattr(cfg, "HEAD_MIN_EPOCHS", 0))

    for epoch in range(1, int(cfg.HEAD_EPOCHS) + 1):
        model.train()
        losses: List[float] = []
        for xb, yb, y_reg_b, seed_prob_b in train_loader:
            xb = xb.to(cfg.DEVICE)
            yb = yb.to(cfg.DEVICE)
            y_reg_b = y_reg_b.to(cfg.DEVICE)
            seed_prob_b = seed_prob_b.to(cfg.DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits, ord_logits, bp_proxy = model(xb)
            loss_cls = guided_script.focal_ce_loss(
                logits,
                yb,
                class_weights,
                gamma=float(cfg.HEAD_FOCAL_GAMMA),
                label_smoothing=float(cfg.HEAD_LABEL_SMOOTHING),
            )
            loss_kd = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(logits / 1.25, dim=1),
                seed_prob_b,
                reduction="batchmean",
            )
            ord_target = guided_script.ordinal_targets(yb, cfg.N_CLASSES).to(cfg.DEVICE)
            loss_ord = torch.nn.functional.binary_cross_entropy_with_logits(
                ord_logits,
                ord_target,
                pos_weight=ord_pos_weight,
            )
            y_reg_norm = (y_reg_b - reg_mu) / reg_sigma
            loss_bp = torch.nn.functional.smooth_l1_loss(bp_proxy, y_reg_norm)
            loss = (
                loss_cls
                + float(cfg.HEAD_KD_WEIGHT) * loss_kd
                + float(cfg.HEAD_ORD_WEIGHT) * loss_ord
                + float(cfg.HEAD_BP_WEIGHT) * loss_bp
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5)
            optimizer.step()
            losses.append(float(loss.item()))

        _, _, clean_metrics, _ = guided_script.evaluate_head(model, val_clean_bank, state, cfg, prefix="guided_val")
        _, _, noise_metrics, _ = guided_script.evaluate_head(model, val_noise_bank, state, cfg, prefix="guided_noise")
        _, _, ecg_metrics, _ = guided_script.evaluate_head(model, val_ecg_bank, state, cfg, prefix="guided_missing_ecg")
        _, _, ppg_metrics, _ = guided_script.evaluate_head(model, val_ppg_bank, state, cfg, prefix="guided_missing_ppg")

        noise_f1 = float(noise_metrics["cls_f1_macro_guided_noise"])
        ecg_f1 = float(ecg_metrics["cls_f1_macro_guided_missing_ecg"])
        ppg_f1 = float(ppg_metrics["cls_f1_macro_guided_missing_ppg"])
        robust_min = min(noise_f1, ecg_f1, ppg_f1)
        clean_summary = _target_class_summary(clean_metrics, "guided_val")
        clean_gaps = _target_gap_components(clean_summary)
        score = float(
            targeted_classification_candidate_score(clean_metrics, "guided_val")
            + float(getattr(cfg, "HEAD_ROBUST_NOISE_WEIGHT", 0.0)) * noise_f1
            + float(getattr(cfg, "HEAD_ROBUST_ECG_WEIGHT", 0.0)) * ecg_f1
            + float(getattr(cfg, "HEAD_ROBUST_PPG_WEIGHT", 0.0)) * ppg_f1
            + float(getattr(cfg, "HEAD_ROBUST_MIN_WEIGHT", 0.0)) * robust_min
            + float(getattr(cfg, "HEAD_TARGET_RARE_MIN_WEIGHT", 0.0)) * clean_summary["rare_f1_min"]
            + float(getattr(cfg, "HEAD_TARGET_STAGE2_WEIGHT", 0.0)) * clean_summary["stage2_f1"]
            - float(getattr(cfg, "HEAD_TARGET_ROBUST_GAP_WEIGHT", 0.0)) * clean_gaps["robust_gap"]
        )

        clean_acc = float(clean_metrics["cls_acc_guided_val"])
        clean_f1 = float(clean_metrics["cls_f1_macro_guided_val"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        target_gap = (
            clean_gaps["acc_gap"]
            + clean_gaps["f1_gap"]
            + 0.60 * clean_gaps["elevated_gap"]
            + 0.75 * clean_gaps["stage1_gap"]
            + 0.90 * clean_gaps["stage2_gap"]
            + float(getattr(cfg, "HEAD_TARGET_GAP_WEIGHT", 1.0)) * clean_gaps["robust_gap"]
        )

        epoch_row = {
            "epoch": int(epoch),
            "lr": current_lr,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "score": float(score),
            "target_gap": float(target_gap),
            "target_rare_f1_min": float(clean_summary["rare_f1_min"]),
            "target_stage2_f1": float(clean_summary["stage2_f1"]),
            "target_robust_min": float(robust_min),
            **clean_metrics,
            **noise_metrics,
            **ecg_metrics,
            **ppg_metrics,
        }
        epoch_rows.append(epoch_row)
        print(
            f"[Warmstart Head Epoch {epoch:03d}] "
            f"lr={current_lr:.6g} | "
            f"train_loss={epoch_row['train_loss']:.4f} | "
            f"head_val_acc={clean_acc:.3f} | "
            f"head_val_f1(guided)={clean_f1:.3f} | "
            f"head_noise_f1={noise_f1:.3f} | "
            f"head_ecgmiss_f1={ecg_f1:.3f} | "
            f"head_ppgmiss_f1={ppg_f1:.3f} | "
            f"head_robust_min={robust_min:.3f} | "
            f"head_score(higher=better)={score:.4f}"
        )

        rank_mode = str(getattr(cfg, "HEAD_SELECTION_RANK_MODE", "target_gap_first")).lower()
        if rank_mode == "clean_acc_f1_then_score":
            clean_priority = (
                float(getattr(cfg, "HEAD_CLEAN_ACC_WEIGHT", 1.25)) * clean_acc
                + float(getattr(cfg, "HEAD_CLEAN_F1_WEIGHT", 1.00)) * clean_f1
                + float(getattr(cfg, "HEAD_CLEAN_BALANCED_WEIGHT", 0.35)) * clean_summary["balanced_acc"]
                + float(getattr(cfg, "HEAD_CLEAN_ROBUST_WEIGHT", 0.18)) * robust_min
                + float(getattr(cfg, "HEAD_CLEAN_STAGE2_WEIGHT", 0.12)) * clean_summary["stage2_f1"]
            )
            current_rank = (
                float(clean_priority),
                clean_acc,
                clean_f1,
                float(score),
                robust_min,
                -float(target_gap),
                clean_summary["stage2_f1"],
                noise_f1,
                ecg_f1,
                ppg_f1,
            )
        else:
            current_rank = (
                -float(target_gap),
                float(score),
                clean_f1,
                clean_acc,
                robust_min,
                clean_summary["stage2_f1"],
                noise_f1,
                ecg_f1,
                ppg_f1,
            )
        if best_rank is None or current_rank > best_rank:
            best_rank = current_rank
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "clean": clean_metrics,
                "noise": noise_metrics,
                "ecg": ecg_metrics,
                "ppg": ppg_metrics,
            }
            patience = 0
        else:
            patience += 1

        scheduler.step()
        if epoch >= min_epochs and patience >= int(cfg.HEAD_PATIENCE):
            print(f"Early stopping at epoch {epoch} after min_epochs={min_epochs}.")
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("No valid checkpoint recorded while warm-start training the guided feature head.")

    model.load_state_dict(best_state)
    model.to(cfg.DEVICE)
    model.eval()
    print(f"Selected best warmstart-head checkpoint from epoch {best_epoch:03d}.")
    return model, state, best_metrics, epoch_rows


def _crisis_debias_registry_key(cfg) -> str:
    return f"{cfg.PROTOCOL_ID}_crisis_debias"


def _tail_underestimation_penalty(y_true_reg: np.ndarray, y_pred_reg: np.ndarray, quantiles: tuple[float, ...]) -> float:
    y_true_reg = np.asarray(y_true_reg, dtype=np.float32)
    y_pred_reg = np.asarray(y_pred_reg, dtype=np.float32)
    err = y_pred_reg - y_true_reg
    sbp_true = y_true_reg[:, 0]
    dbp_true = y_true_reg[:, 1]

    penalty = 0.0
    for idx, q in enumerate(tuple(float(x) for x in quantiles), start=1):
        sbp_thr = float(np.quantile(sbp_true, q))
        dbp_thr = float(np.quantile(dbp_true, q))
        sbp_mask = sbp_true >= sbp_thr
        dbp_mask = dbp_true >= dbp_thr
        if np.any(sbp_mask):
            sbp_under = max(0.0, -float(err[sbp_mask, 0].mean()))
            sbp_p90 = float(np.quantile(np.abs(err[sbp_mask, 0]), 0.90))
            penalty += (0.90 + 0.45 * idx) * sbp_under + 0.08 * sbp_p90
        if np.any(dbp_mask):
            dbp_under = max(0.0, -float(err[dbp_mask, 1].mean()))
            dbp_p90 = float(np.quantile(np.abs(err[dbp_mask, 1]), 0.90))
            penalty += (0.35 + 0.18 * idx) * dbp_under + 0.04 * dbp_p90
    return float(penalty)


def build_crisis_debias_features(reg_out: dict, cls_prob: np.ndarray, reg_inputs: dict, cfg) -> np.ndarray:
    pred = np.asarray(reg_out["y_pred_reg"], dtype=np.float32)
    cls_prob = bridge_script.normalize_prob(np.asarray(cls_prob, dtype=np.float32))
    context = meta_script.build_crisis_tail_signal_context(reg_out, cls_prob, reg_inputs)
    expert_stack = np.asarray(context["expert_stack"], dtype=np.float32)
    expert_peak = np.max(expert_stack, axis=1).astype(np.float32)
    expert_q90 = np.quantile(expert_stack, 0.90, axis=1).astype(np.float32)
    expert_q75 = np.quantile(expert_stack, 0.75, axis=1).astype(np.float32)
    expert_mean = np.mean(expert_stack, axis=1).astype(np.float32)
    uncertainty = np.asarray(reg_out.get("uncertainty", np.zeros(len(pred), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
    quality = np.asarray(reg_out.get("quality", np.ones(len(pred), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)

    return np.concatenate(
        [
            pred,
            cls_prob,
            context["expert_high_signal"].reshape(-1, 1),
            context["expert_crisis_signal"].reshape(-1, 1),
            context["spread_signal"].reshape(-1, 1),
            context["uncertainty_signal"].reshape(-1, 1),
            uncertainty,
            quality,
            expert_peak,
            expert_q90,
            expert_q75,
            expert_mean,
            np.clip(expert_peak - pred, 0.0, None),
            np.clip(expert_q90 - pred, 0.0, None),
            np.clip(expert_q75 - pred, 0.0, None),
            expert_stack.reshape(len(pred), -1),
        ],
        axis=1,
    ).astype(np.float32)


def _crisis_debias_sample_weight(calib_out: dict, cls_prob: np.ndarray, reg_inputs: dict, cfg) -> np.ndarray:
    pred = np.asarray(calib_out["y_pred_reg"], dtype=np.float32)
    y_true = np.asarray(calib_out["y_true_reg"], dtype=np.float32)
    context = meta_script.build_crisis_tail_signal_context(calib_out, cls_prob, reg_inputs)
    under = np.clip(y_true - pred, 0.0, None)
    true_high = 0.55 * meta_script._sigmoid((y_true[:, 0] - 140.0) / 8.0) + 0.45 * meta_script._sigmoid((y_true[:, 1] - 90.0) / 6.0)
    true_crisis = 0.60 * meta_script._sigmoid((y_true[:, 0] - 170.0) / 5.5) + 0.40 * meta_script._sigmoid((y_true[:, 1] - 110.0) / 4.5)
    under_signal = np.clip(under[:, 0] / 12.0 + 0.65 * under[:, 1] / 8.0, 0.0, 3.0)
    weight = (
        1.0
        + 1.25 * context["expert_high_signal"]
        + 2.75 * context["expert_crisis_signal"]
        + 1.30 * true_high
        + 2.60 * true_crisis
        + 1.80 * under_signal
        + 0.40 * context["spread_signal"]
        + 0.35 * context["uncertainty_signal"]
    )
    return np.clip(weight.astype(np.float32), 1.0, 60.0)


def fit_crisis_debias_bundle(calib_out: dict, calib_cls_prob: np.ndarray, calib_reg_inputs: dict, cfg, seed: int) -> dict:
    x = build_crisis_debias_features(calib_out, calib_cls_prob, calib_reg_inputs, cfg)
    y_true = np.asarray(calib_out["y_true_reg"], dtype=np.float32)
    pred = np.asarray(calib_out["y_pred_reg"], dtype=np.float32)
    target = np.clip(y_true - pred, 0.0, None).astype(np.float32)
    sample_weight = _crisis_debias_sample_weight(calib_out, calib_cls_prob, calib_reg_inputs, cfg)
    models = meta_script.fit_weighted_regressor_ensemble_safe(x, target, sample_weight, seed=seed)
    return {"models": models}


def predict_crisis_debias_delta(bundle: dict, reg_out: dict, cls_prob: np.ndarray, reg_inputs: dict, cfg) -> np.ndarray:
    x = build_crisis_debias_features(reg_out, cls_prob, reg_inputs, cfg)
    pred = np.asarray(meta_script.stage_script.predict_regressor_ensemble(bundle["models"], x), dtype=np.float32)
    return np.clip(pred.reshape(len(x), 2), 0.0, None)


def apply_crisis_tail_debias_fusion(
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

    bundle_key = str(row.get("bundle_key", _crisis_debias_registry_key(cfg)))
    bundle = _CRISIS_DEBIAS_REGISTRY.get(bundle_key)
    if bundle is None:
        raise KeyError(f"Crisis debias bundle not found: {bundle_key}")

    pred = np.asarray(reg_out["y_pred_reg"], dtype=np.float32)
    cls_prob = bridge_script.normalize_prob(np.asarray(cls_prob, dtype=np.float32))
    context = meta_script.build_crisis_tail_signal_context(reg_out, cls_prob, reg_inputs)
    expert_stack = np.asarray(context["expert_stack"], dtype=np.float32)
    positive_delta = predict_crisis_debias_delta(bundle, reg_out, cls_prob, reg_inputs, cfg)
    stage1_prob = cls_prob[:, 2].astype(np.float32)
    stage2_prob = cls_prob[:, 3].astype(np.float32)
    hypertensive_prob = np.clip(stage1_prob + stage2_prob, 0.0, 1.0).astype(np.float32)

    high_gate = meta_script._signal_ramp(context["expert_high_signal"], float(row["high_threshold"]), float(row["gamma"]))
    crisis_gate = meta_script._signal_ramp(context["expert_crisis_signal"], float(row["crisis_threshold"]), float(row["gamma"]))
    gate_scale = (
        1.0
        + float(row["uncertainty_gain"]) * context["uncertainty_signal"]
        + 0.35 * context["spread_signal"]
    ).astype(np.float32)

    sbp_anchor = np.quantile(expert_stack[:, :, 0], float(row["sbp_quantile"]), axis=1).astype(np.float32)
    dbp_anchor = np.quantile(expert_stack[:, :, 1], float(row["dbp_quantile"]), axis=1).astype(np.float32)
    sbp_expert_gap = np.clip(sbp_anchor - pred[:, 0], 0.0, None)
    dbp_expert_gap = np.clip(dbp_anchor - pred[:, 1], 0.0, None)
    disagreement_signal = np.clip(
        sbp_expert_gap / max(float(row["sbp_margin"]), 1.0),
        0.0,
        2.5,
    ).astype(np.float32)
    hazard_gate = np.clip(
        float(getattr(cfg, "CRISIS_TAIL_CLASS_STAGE1_GAIN", 0.85)) * stage1_prob
        + float(getattr(cfg, "CRISIS_TAIL_CLASS_STAGE2_GAIN", 1.85)) * stage2_prob
        + float(getattr(cfg, "CRISIS_TAIL_DISAGREEMENT_GAIN", 0.85)) * np.clip(disagreement_signal, 0.0, 1.5)
        + 0.35 * high_gate
        + 0.85 * crisis_gate
        + 0.22 * context["uncertainty_signal"],
        0.0,
        3.0,
    ).astype(np.float32)

    sbp_gate = np.clip((0.32 * high_gate + float(row["crisis_gain"]) * crisis_gate) * gate_scale, 0.0, 1.5).astype(np.float32)
    dbp_gate = np.clip((0.26 * high_gate + 0.72 * float(row["crisis_gain"]) * crisis_gate) * gate_scale, 0.0, 1.35).astype(np.float32)

    sbp_anchor_delta = (float(row["expert_gain"]) * sbp_expert_gap + float(row["sbp_margin"]) * crisis_gate).astype(np.float32)
    dbp_anchor_delta = (float(row["expert_gain"]) * dbp_expert_gap + float(row["dbp_margin"]) * crisis_gate).astype(np.float32)
    sbp_hazard_delta = (
        np.clip(0.25 + 0.40 * hazard_gate + 0.30 * hypertensive_prob, 0.0, 2.20)
        * (0.55 * sbp_expert_gap + float(row["sbp_margin"]))
    ).astype(np.float32)
    dbp_hazard_delta = (
        np.clip(0.18 + 0.28 * hazard_gate + 0.22 * stage2_prob, 0.0, 1.60)
        * (0.45 * dbp_expert_gap + float(row["dbp_margin"]))
    ).astype(np.float32)

    sbp_delta = np.maximum(
        sbp_gate * float(row["model_scale"]) * positive_delta[:, 0],
        np.clip(0.22 + 0.58 * high_gate + 0.95 * crisis_gate, 0.0, 1.4) * sbp_anchor_delta,
    )
    dbp_delta = np.maximum(
        dbp_gate * float(row["model_scale"]) * positive_delta[:, 1],
        np.clip(0.18 + 0.42 * high_gate + 0.72 * crisis_gate, 0.0, 1.25) * dbp_anchor_delta,
    )
    sbp_delta = np.maximum(sbp_delta, sbp_hazard_delta)
    dbp_delta = np.maximum(dbp_delta, dbp_hazard_delta)

    hard_crisis_mask = (
        (crisis_gate >= 0.80)
        | (stage2_prob >= 0.55)
        | ((hypertensive_prob >= 0.78) & (sbp_expert_gap >= 6.5))
        | ((context["expert_high_signal"] >= 0.92) & (sbp_expert_gap >= 8.0))
    )
    sbp_delta[hard_crisis_mask] = np.maximum(
        sbp_delta[hard_crisis_mask],
        sbp_expert_gap[hard_crisis_mask] + 0.5 * float(row["sbp_margin"]),
    )
    dbp_delta[hard_crisis_mask] = np.maximum(
        dbp_delta[hard_crisis_mask],
        0.70 * dbp_expert_gap[hard_crisis_mask] + 0.5 * float(row["dbp_margin"]),
    )
    sbp_hard_floor = np.where(
        hard_crisis_mask,
        np.maximum(
            float(getattr(cfg, "CRISIS_TAIL_HARD_FLOOR_SBP", 8.5)),
            0.55 * sbp_expert_gap + 0.65 * float(row["sbp_margin"]),
        ),
        0.0,
    ).astype(np.float32)
    dbp_hard_floor = np.where(
        hard_crisis_mask,
        np.maximum(
            float(getattr(cfg, "CRISIS_TAIL_HARD_FLOOR_DBP", 3.5)),
            0.45 * dbp_expert_gap + 0.55 * float(row["dbp_margin"]),
        ),
        0.0,
    ).astype(np.float32)
    sbp_delta = np.maximum(sbp_delta, sbp_hard_floor)
    dbp_delta = np.maximum(dbp_delta, dbp_hard_floor)

    delta = np.stack([sbp_delta, dbp_delta], axis=1).astype(np.float32)
    delta[:, 0] = np.clip(delta[:, 0], 0.0, float(getattr(cfg, "CRISIS_TAIL_MAX_SHIFT_SBP", cfg.RISK_GUARD_MAX_SHIFT_SBP)))
    delta[:, 1] = np.clip(delta[:, 1], 0.0, float(getattr(cfg, "CRISIS_TAIL_MAX_SHIFT_DBP", cfg.RISK_GUARD_MAX_SHIFT_DBP)))

    corrected = meta_script.stage_script.clipped_regression_prediction(pred + delta)
    out = meta_script.stage_script.clone_regression_output(reg_out, corrected, cfg)
    out["crisis_tail_fusion_gate_mean"] = float(0.5 * (np.maximum(sbp_gate, hazard_gate).mean() + np.maximum(dbp_gate, 0.80 * hazard_gate).mean()))
    out["crisis_tail_fusion_shift_mean_sbp"] = float(delta[:, 0].mean())
    out["crisis_tail_fusion_shift_mean_dbp"] = float(delta[:, 1].mean())
    out["crisis_tail_fusion_activation_rate"] = float(np.mean((crisis_gate >= 0.15) | (high_gate >= 0.20) | (hazard_gate >= 0.70)))
    return out


def crisis_tail_debias_cost(calib_out: dict, query_out: dict, base_ref: dict, cfg):
    conformal = meta_script.stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    bp_range_rows = meta_script.stage_script.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"])
    range_map = {str(row["bp_range"]): row for row in bp_range_rows}
    high_row = range_map.get("high", {})
    crisis_row = range_map.get("crisis", {})
    clinical_pen = meta_script.clinical_underestimation_penalty(bp_range_rows)
    tail_pen = meta_script.prev_script.tail_bias_penalty(bp_range_rows)
    surrogate_pen = _tail_underestimation_penalty(
        np.asarray(query_out["y_true_reg"], dtype=np.float32),
        np.asarray(query_out["y_pred_reg"], dtype=np.float32),
        tuple(getattr(cfg, "CRISIS_TAIL_SURROGATE_QUANTILES", (0.90, 0.95, 0.98))),
    )
    crisis_under_pen = (
        float(getattr(cfg, "CRISIS_TAIL_UNDEREST_WEIGHT_SBP", 5.10)) * max(0.0, -float(crisis_row.get("bias_sbp", 0.0)))
        + float(getattr(cfg, "CRISIS_TAIL_UNDEREST_WEIGHT_DBP", 2.20)) * max(0.0, -float(crisis_row.get("bias_dbp", 0.0)))
    )
    crisis_abs_pen = 1.35 * abs(float(crisis_row.get("bias_sbp", 0.0))) + 0.70 * abs(float(crisis_row.get("bias_dbp", 0.0)))
    high_under_pen = 1.55 * max(0.0, -float(high_row.get("bias_sbp", 0.0))) + 0.82 * max(0.0, -float(high_row.get("bias_dbp", 0.0)))
    reg = query_out["metrics_reg"]
    mae_excess = max(0.0, float(reg["mae_mean"]) - float(base_ref["mae_mean"]) - float(cfg.CRISIS_TAIL_FUSION_MAX_MAE_DELTA))
    cov_excess = max(0.0, float(conformal["coverage_gap"]) - float(base_ref["coverage_gap"]) - float(cfg.CRISIS_TAIL_FUSION_MAX_COVERAGE_GAP_DELTA))
    score = float(
        1.90 * surrogate_pen
        + crisis_under_pen
        + 0.55 * crisis_abs_pen
        + 0.75 * high_under_pen
        + 0.42 * clinical_pen
        + 0.20 * tail_pen
        + 18.0 * mae_excess
        + 8.0 * cov_excess
        + 0.008 * float(reg["mae_mean"])
    )
    return float(score), conformal, bp_range_rows, float(clinical_pen), float(tail_pen)


def search_crisis_tail_debias_candidates(
    calib_out: dict,
    calib_cls_prob: np.ndarray,
    calib_reg_inputs: dict,
    query_out: dict,
    query_cls_prob: np.ndarray,
    query_reg_inputs: dict,
    cfg,
) -> tuple[dict, List[dict]]:
    bundle_key = _crisis_debias_registry_key(cfg)
    _CRISIS_DEBIAS_REGISTRY[bundle_key] = fit_crisis_debias_bundle(calib_out, calib_cls_prob, calib_reg_inputs, cfg, seed=int(cfg.SEED) + 4701)

    base_conformal = meta_script.stage_script.summarize_conformal_tradeoff(calib_out, query_out, cfg)
    base_bp_range_rows = meta_script.stage_script.build_bp_range_table(query_out["y_true_reg"], query_out["y_pred_reg"])
    base_clinical_pen = meta_script.clinical_underestimation_penalty(base_bp_range_rows)
    base_tail_pen = meta_script.prev_script.tail_bias_penalty(base_bp_range_rows)
    base_ref = {"mae_mean": float(query_out["metrics_reg"]["mae_mean"]), "coverage_gap": float(base_conformal["coverage_gap"])}
    range_map = {str(row["bp_range"]): row for row in base_bp_range_rows}
    base_score, _, _, _, _ = crisis_tail_debias_cost(calib_out, query_out, base_ref, cfg)

    rows: List[dict] = [{
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
        "score": float(base_score),
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
    }]

    for high_threshold in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_HIGH_THRESHOLDS):
        for crisis_threshold in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_CRISIS_THRESHOLDS):
            for gamma in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_GAMMAS):
                for sbp_quantile in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_SBP_QUANTILES):
                    for dbp_quantile in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_DBP_QUANTILES):
                        for crisis_gain in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_CRISIS_GAINS):
                            for sbp_margin in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_SBP_MARGINS):
                                for dbp_margin in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_DBP_MARGINS):
                                    for uncertainty_gain in tuple(float(x) for x in cfg.CRISIS_TAIL_FUSION_UNCERTAINTY_GAINS):
                                        for model_scale in tuple(float(x) for x in getattr(cfg, "CRISIS_TAIL_FUSION_MODEL_SCALES", (0.60, 0.85, 1.10, 1.35))):
                                            for expert_gain in tuple(float(x) for x in getattr(cfg, "CRISIS_TAIL_FUSION_EXPERT_GAINS", (0.35, 0.70, 1.00))):
                                                row = {
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
                                                    "bundle_key": bundle_key,
                                                    "high_threshold": float(high_threshold),
                                                    "crisis_threshold": float(crisis_threshold),
                                                    "gamma": float(gamma),
                                                    "sbp_quantile": float(sbp_quantile),
                                                    "dbp_quantile": float(dbp_quantile),
                                                    "crisis_gain": float(crisis_gain),
                                                    "sbp_margin": float(sbp_margin),
                                                    "dbp_margin": float(dbp_margin),
                                                    "uncertainty_gain": float(uncertainty_gain),
                                                    "model_scale": float(model_scale),
                                                    "expert_gain": float(expert_gain),
                                                }
                                                calib_adj = apply_crisis_tail_debias_fusion(row, calib_out, calib_cls_prob, calib_reg_inputs, cfg)
                                                query_adj = apply_crisis_tail_debias_fusion(row, query_out, query_cls_prob, query_reg_inputs, cfg)
                                                score, conformal, bp_range_rows, clinical_pen, tail_pen = crisis_tail_debias_cost(calib_adj, query_adj, base_ref, cfg)
                                                row_map = {str(item["bp_range"]): item for item in bp_range_rows}
                                                rows.append({
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
                                                })

    rows.sort(key=lambda row: float(row["score"]))
    return rows[0], rows


def main():
    optlong_ckpt, dualmax_ckpt = train_or_load_full_backbones()

    original_nextgen_build_cfg = nextgen_script.build_nextgen_cfg
    original_guided_score = guided_script.classification_candidate_score
    original_bridge_score = bridge_script.classification_selection_score
    original_resume_robust_score = resume_script.robust_classification_score
    original_resume_long = adaptive_script.run_feature_head_resume_long
    original_bridge_pick_optlong = bridge_script.pick_optlong_best_checkpoint
    original_bridge_pick_dualmax = bridge_script.pick_dualmax_best_checkpoint
    original_meta_apply_crisis_tail = meta_script.apply_crisis_tail_fusion
    original_meta_crisis_cost = meta_script.crisis_tail_fusion_cost
    original_meta_search_crisis_tail = meta_script.search_crisis_tail_fusion_candidates
    original_stage_seed_everything = stage_dual_script.seed_everything
    original_meta_stage_seed_everything = meta_script.stage_script.seed_everything
    original_common_seed_everything = common_script.seed_everything
    original_shared_v9_seed_everything = shared_v9.seed_everything

    def _pick_optlong_best_checkpoint(_cfg):
        return optlong_ckpt

    def _pick_dualmax_best_checkpoint(_cfg):
        return dualmax_ckpt

    try:
        print(
            "Metric note: after backbone full-training, upcoming `Warmstart Head Epoch` logs report "
            "`head_val_f1(guided)`, which is the directly comparable quantity to v10.10 `Resume Head` val_f1."
        )
        nextgen_script.build_nextgen_cfg = build_nextgen_cfg
        guided_script.classification_candidate_score = targeted_classification_candidate_score
        bridge_script.classification_selection_score = targeted_classification_selection_score
        resume_script.robust_classification_score = targeted_robust_classification_score
        adaptive_script.run_feature_head_resume_long = run_feature_head_warmstart_targeted
        bridge_script.pick_optlong_best_checkpoint = _pick_optlong_best_checkpoint
        bridge_script.pick_dualmax_best_checkpoint = _pick_dualmax_best_checkpoint
        meta_script.apply_crisis_tail_fusion = apply_crisis_tail_debias_fusion
        meta_script.crisis_tail_fusion_cost = crisis_tail_debias_cost
        meta_script.search_crisis_tail_fusion_candidates = search_crisis_tail_debias_candidates
        stage_dual_script.seed_everything = _safe_seed_everything
        meta_script.stage_script.seed_everything = _safe_seed_everything
        common_script.seed_everything = _safe_seed_everything
        shared_v9.seed_everything = _safe_seed_everything
        _safe_cuda_cleanup()
        nextgen_script.main()
    finally:
        nextgen_script.build_nextgen_cfg = original_nextgen_build_cfg
        guided_script.classification_candidate_score = original_guided_score
        bridge_script.classification_selection_score = original_bridge_score
        resume_script.robust_classification_score = original_resume_robust_score
        adaptive_script.run_feature_head_resume_long = original_resume_long
        bridge_script.pick_optlong_best_checkpoint = original_bridge_pick_optlong
        bridge_script.pick_dualmax_best_checkpoint = original_bridge_pick_dualmax
        meta_script.apply_crisis_tail_fusion = original_meta_apply_crisis_tail
        meta_script.crisis_tail_fusion_cost = original_meta_crisis_cost
        meta_script.search_crisis_tail_fusion_candidates = original_meta_search_crisis_tail
        stage_dual_script.seed_everything = original_stage_seed_everything
        meta_script.stage_script.seed_everything = original_meta_stage_seed_everything
        common_script.seed_everything = original_common_seed_everything
        shared_v9.seed_everything = original_shared_v9_seed_everything


def _dispatch_subprocess_mode(argv: List[str]) -> int | None:
    if not argv:
        return None
    mode = str(argv[0]).strip()
    if mode == "--run-optlong-fulltrain-subprocess":
        if len(argv) != 3:
            raise ValueError("Expected arguments: --run-optlong-fulltrain-subprocess <batch_size> <cudnn_enabled>")
        return _subprocess_optlong_fulltrain_entry(int(argv[1]), bool(int(argv[2])))
    if mode == "--run-stageaware-fulltrain-subprocess":
        if len(argv) != 4:
            raise ValueError(
                "Expected arguments: --run-stageaware-fulltrain-subprocess "
                "<optlong_ckpt> <batch_size> <cudnn_enabled>"
            )
        return _subprocess_stageaware_fulltrain_entry(Path(argv[1]), int(argv[2]), bool(int(argv[3])))
    return None


if __name__ == "__main__":
    dispatch_code = _dispatch_subprocess_mode(sys.argv[1:])
    if dispatch_code is None:
        main()
    else:
        raise SystemExit(dispatch_code)
