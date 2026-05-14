from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List


MAIN_RUNS = (
    (
        "v10.11",
        "mimic_bp_reg_v10_11_subjectdisjoint_piso_uncertainty_moe_fulltrain_crisisdebias_proto",
    ),
    (
        "v10.17",
        "mimic_bp_reg_v10_17_subjectdisjoint_paretoordinal_crisiscal_proto",
    ),
    (
        "v10.18",
        "mimic_bp_reg_v10_18_subjectdisjoint_baselineguard_crisisrepair_proto",
    ),
)

COMPARISON_METHODS = (
    (
        "ann_lstm_ecg_ppg",
        "Cuffless blood pressure estimation from ECG and PPG using waveform based ANN-LSTM network",
        "https://doi.org/10.1016/j.bspc.2019.02.028",
    ),
    (
        "ppg_bilstm_attention",
        "Deep learning models for cuffless blood pressure monitoring from PPG signals using attention mechanism",
        "https://doi.org/10.1016/j.bspc.2020.102301",
    ),
    (
        "bpnet_cnn",
        "BP-Net: Cuff-less and non-invasive blood pressure estimation via a generic deep convolutional architecture",
        "https://doi.org/10.1016/j.bspc.2022.103850",
    ),
    (
        "mlp_bp_mixer",
        "MLP-BP: cuffless BP measurement with PPG and ECG based on MLP-Mixer neural networks",
        "https://doi.org/10.1016/j.bspc.2021.103404",
    ),
    (
        "piso_transformer",
        "PiSO model-selection branch for ABP estimation on MIMIC-IV",
        "https://doi.org/10.1109/ACCESS.2026.3665255",
    ),
    (
        "mufubp_dual_feature_pfe",
        "MuFuBP-Net multimodal fusion with dual-feature pipeline and probabilistic feature encoder",
        "https://doi.org/10.1109/JBHI.2025.3563852",
    ),
)

ABLATIONS = (
    (
        "full",
        "none",
        "mimic_bp_reg_v10_18_subjectdisjoint_baselineguard_crisisrepair_proto",
    ),
    (
        "no_tail_reweighting",
        "tail-aware training loss/sampler weights",
        "mimic_bp_ablation_v10_18_no_tail_reweighting_proto",
    ),
    (
        "no_reliability_bias_calibration",
        "high-range reliability bias calibration",
        "mimic_bp_ablation_v10_18_no_reliability_bias_calibration_proto",
    ),
    (
        "no_crisis_tail_debias",
        "crisis-tail debias/fusion search",
        "mimic_bp_ablation_v10_18_no_crisis_tail_debias_proto",
    ),
    (
        "no_safety_evidential_fusion",
        "safety-aware classification fusion",
        "mimic_bp_ablation_v10_18_no_safety_evidential_fusion_proto",
    ),
    (
        "no_accf1_targeted_head",
        "accuracy/F1-targeted feature-head ranking",
        "mimic_bp_ablation_v10_18_no_accf1_targeted_head_proto",
    ),
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


def _to_float(value, default=""):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _range_row(output: Path, label: str) -> dict:
    for row in _read_csv_rows(output / "tables" / "bp_range_metrics.csv"):
        if str(row.get("bp_range", "")).lower() == label:
            return row
    return {}


def _metric_from_results(results: dict, *keys: str):
    for section in ("test_selected", "test", "validation_selected", "validation"):
        metrics = results.get(section, {})
        if not isinstance(metrics, dict):
            continue
        for key in keys:
            if key in metrics:
                return metrics.get(key)
    return ""


def _summary_row(name: str, output_name: str, **extra) -> dict:
    output = _project_root() / "outputs" / output_name
    summary = _read_json(output / "protocol_summary.json")
    results = _read_json(output / "final_results.json")
    paper = _read_json(output / "paper_metrics.json")
    aami = paper.get("aami_like", {})
    bhs = paper.get("bhs_like", {})
    high = _range_row(output, "high")
    crisis = _range_row(output, "crisis")
    selected = _read_json(output / "selected_strategy.json")
    return {
        "name": name,
        "output": output_name,
        "exists": output.exists(),
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
        "selected_regression_candidate": summary.get("selected_regression_candidate", ""),
        "selected_classification_candidate": summary.get("selected_classification_candidate", ""),
        "classification_source": summary.get(
            "classification_source",
            selected.get("classification_source", results.get("classification_source", "")),
        ),
        "support_calibration": summary.get(
            "support_calibration",
            selected.get("support_calibration", results.get("support_calibration", "")),
        ),
        "support_shrinkage": summary.get(
            "support_shrinkage",
            selected.get("support_shrinkage", results.get("support_shrinkage", "")),
        ),
        "selected_tail_correction_candidate": summary.get("selected_tail_correction_candidate", ""),
        "crisis_tail_fusion_candidate": summary.get(
            "crisis_tail_fusion_candidate",
            selected.get("crisis_tail_fusion_candidate", ""),
        ),
        **extra,
    }


def main() -> None:
    out_dir = _project_root() / "outputs" / "v10_18_paper_summary"
    main_rows = [_summary_row(name, output, kind="main_version") for name, output in MAIN_RUNS]
    comparison_rows = [
        _summary_row(
            method,
            f"mimic_bp_compare_v10_18_{method}_proto",
            kind="comparison_baseline",
            paper_title=title,
            paper_url=url,
        )
        for method, title, url in COMPARISON_METHODS
    ]
    lite_ablation_rows = _read_csv_rows(out_dir / "lite_train_ablation_summary.csv")
    major_ablation_rows = _read_csv_rows(out_dir / "major_module_ablation_summary.csv")
    if lite_ablation_rows or major_ablation_rows:
        ablation_rows = []
        for row in lite_ablation_rows:
            normalized = dict(row)
            normalized.setdefault("kind", "lite_train_ablation")
            normalized.setdefault("module_removed", "")
            ablation_rows.append(normalized)
        for row in major_ablation_rows:
            normalized = dict(row)
            normalized.setdefault("kind", "major_module_ablation")
            normalized.setdefault("module_removed", "")
            ablation_rows.append(normalized)
    else:
        ablation_rows = [
            _summary_row(
                variant,
                output,
                kind="ablation",
                module_removed=module_removed,
            )
            for variant, module_removed, output in ABLATIONS
        ]
    _write_csv(out_dir / "main_version_summary.csv", main_rows)
    _write_csv(out_dir / "comparison_summary.csv", comparison_rows)
    _write_csv(out_dir / "ablation_summary.csv", ablation_rows)
    _write_csv(
        out_dir / "all_paper_runs_summary.csv",
        main_rows + comparison_rows + ablation_rows,
    )
    print(f"[v10.18 summary] Wrote summary tables to: {out_dir}")


if __name__ == "__main__":
    main()
