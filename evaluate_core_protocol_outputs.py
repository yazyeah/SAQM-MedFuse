from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
AUDIT_ROOT = OUTPUT_ROOT / "core_protocol_audit"


PROTOCOL_SPECS = [
    {
        "label": "opt_long",
        "dir": "mimic_bp_reg_v10_2_opt_long_proto",
        "kind": "opt_long",
    },
    {
        "label": "stageaware_dualmax",
        "dir": "mimic_bp_reg_v10_2_optlong_stageaware_dualmax_proto",
        "kind": "stageaware",
    },
    {
        "label": "dualanchor_conservative",
        "dir": "mimic_bp_reg_v10_2_optlong_dualanchor_conservative_proto",
        "kind": "selected",
    },
    {
        "label": "dualanchor_resume_tailcal",
        "dir": "mimic_bp_reg_v10_2_optlong_dualanchor_resume_tailcal_proto",
        "kind": "selected",
    },
    {
        "label": "dualanchor_stability",
        "dir": "mimic_bp_reg_v10_2_optlong_dualanchor_stability_ensemble_proto",
        "kind": "selected",
    },
    {
        "label": "dualanchor_consensus_sparse_tail",
        "dir": "mimic_bp_reg_v10_2_optlong_dualanchor_consensus_sparse_tail_proto",
        "kind": "selected",
    },
    {
        "label": "dualanchor_meta_stack",
        "dir": "mimic_bp_reg_v10_2_optlong_dualanchor_meta_stack_proto",
        "kind": "meta_stack",
    },
    {
        "label": "adaptive_gate_meta",
        "dir": "mimic_bp_reg_v10_2_optlong_dualanchor_adaptive_gate_meta_proto",
        "kind": "meta_stack",
    },
    {
        "label": "uncertainty_moe_paper",
        "dir": "mimic_bp_reg_v10_2_optlong_dualanchor_uncertainty_moe_paper_proto",
        "kind": "meta_stack",
    },
]


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def first_present(mapping: dict | None, keys: Iterable[str], default=None):
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def extract_kind_metrics(kind: str, final_results: dict | None, protocol_summary: dict | None) -> dict:
    row = {
        "mae_mean": None,
        "acc": None,
        "balanced_acc": None,
        "macro_f1": None,
        "selection_regression": None,
        "selection_classification": None,
        "selection_tail": None,
    }
    if kind == "opt_long" and isinstance(final_results, dict):
        test_full = first_present(final_results, ["test_full"], {})
        row["mae_mean"] = first_present(test_full, ["mae_mean"])
        row["acc"] = first_present(test_full, ["cls_acc_from_reg"])
        row["balanced_acc"] = first_present(test_full, ["cls_balanced_acc_from_reg"])
        row["macro_f1"] = first_present(test_full, ["cls_f1_macro_from_reg"])
    elif kind == "stageaware" and isinstance(final_results, dict):
        reg = first_present(final_results, ["test_regression_corrected"], {})
        cls = first_present(final_results, ["test_hybrid_classification"], {})
        row["mae_mean"] = first_present(reg, ["mae_mean"])
        row["acc"] = first_present(cls, ["cls_acc_stageaware_dualmax"])
        row["balanced_acc"] = first_present(cls, ["cls_balanced_acc_stageaware_dualmax"])
        row["macro_f1"] = first_present(cls, ["cls_f1_macro_stageaware_dualmax"])
        row["selection_regression"] = "regression_corrected"
        row["selection_classification"] = "stageaware_dualmax"
    elif kind in {"selected", "meta_stack"}:
        if isinstance(protocol_summary, dict):
            row["mae_mean"] = first_present(protocol_summary, ["selected_mae_mean"])
            row["acc"] = first_present(protocol_summary, ["selected_acc"])
            row["balanced_acc"] = first_present(protocol_summary, ["selected_balanced_acc"])
            row["macro_f1"] = first_present(protocol_summary, ["selected_macro_f1"])
            row["selection_regression"] = first_present(protocol_summary, ["selected_regression_candidate"])
            row["selection_classification"] = first_present(protocol_summary, ["selected_classification_candidate"])
            row["selection_tail"] = first_present(protocol_summary, ["selected_tail_correction_candidate"])
        if kind == "meta_stack" and isinstance(final_results, dict):
            test_selected = first_present(final_results, ["test_selected"], {})
            row["mae_mean"] = first_present(test_selected, ["mae_mean"], row["mae_mean"])
            row["acc"] = first_present(test_selected, ["cls_acc_from_reg"], row["acc"])
            row["balanced_acc"] = first_present(test_selected, ["cls_balanced_acc_from_reg"], row["balanced_acc"])
            row["macro_f1"] = first_present(test_selected, ["cls_f1_macro_from_reg"], row["macro_f1"])
            if isinstance(final_results.get("selection_strategy"), dict):
                row["selection_regression"] = first_present(final_results["selection_strategy"], ["meta_regression_candidate"], row["selection_regression"])
                row["selection_classification"] = "meta_blend"
    return row


def extract_paper_metrics(base_dir: Path, final_results: dict | None) -> dict:
    paper_candidates = [
        base_dir / "paper_metrics_selected.json",
        base_dir / "paper_metrics.json",
    ]
    paper = None
    for candidate in paper_candidates:
        paper = load_json(candidate)
        if isinstance(paper, dict):
            break
    if paper is None and isinstance(final_results, dict):
        paper = first_present(final_results, ["paper_metrics", "paper_metrics_selected"], {})
    if not isinstance(paper, dict):
        paper = {}

    aami = first_present(paper, ["aami_like"], {})
    bhs = first_present(paper, ["bhs_like"], {})
    return {
        "sbp_mean_error": first_present(aami, ["sbp_mean_error"]),
        "sbp_sd_error": first_present(aami, ["sbp_sd_error"]),
        "dbp_mean_error": first_present(aami, ["dbp_mean_error"]),
        "dbp_sd_error": first_present(aami, ["dbp_sd_error"]),
        "sbp_within_5": first_present(bhs, ["sbp_within_5"]),
        "sbp_within_10": first_present(bhs, ["sbp_within_10"]),
        "sbp_within_15": first_present(bhs, ["sbp_within_15"]),
        "dbp_within_5": first_present(bhs, ["dbp_within_5"]),
        "dbp_within_10": first_present(bhs, ["dbp_within_10"]),
        "dbp_within_15": first_present(bhs, ["dbp_within_15"]),
        "sbp_grade": first_present(bhs, ["sbp_grade"]),
        "dbp_grade": first_present(bhs, ["dbp_grade"]),
    }


def numeric(value):
    return None if value is None else float(value)


def build_audit_rows() -> list[dict]:
    rows: list[dict] = []
    for spec in PROTOCOL_SPECS:
        base_dir = OUTPUT_ROOT / spec["dir"]
        final_results = load_json(base_dir / "final_results.json")
        protocol_summary = load_json(base_dir / "protocol_summary.json")
        metric_row = extract_kind_metrics(spec["kind"], final_results, protocol_summary)
        paper_row = extract_paper_metrics(base_dir, final_results)

        rows.append(
            {
                "label": spec["label"],
                "output_dir": str(base_dir),
                "kind": spec["kind"],
                "has_final_results": int(isinstance(final_results, dict)),
                "has_protocol_summary": int(isinstance(protocol_summary, dict)),
                "figure_png_count": count_files(base_dir / "figures", "*.png"),
                "table_csv_count": count_files(base_dir / "tables", "*.csv"),
                "root_csv_count": count_files(base_dir, "*.csv"),
                "artifact_npz_count": count_files(base_dir / "artifacts", "*.npz"),
                "mae_mean": numeric(metric_row["mae_mean"]),
                "acc": numeric(metric_row["acc"]),
                "balanced_acc": numeric(metric_row["balanced_acc"]),
                "macro_f1": numeric(metric_row["macro_f1"]),
                "selection_regression": metric_row["selection_regression"],
                "selection_classification": metric_row["selection_classification"],
                "selection_tail": metric_row["selection_tail"],
                "sbp_mean_error": numeric(paper_row["sbp_mean_error"]),
                "sbp_sd_error": numeric(paper_row["sbp_sd_error"]),
                "dbp_mean_error": numeric(paper_row["dbp_mean_error"]),
                "dbp_sd_error": numeric(paper_row["dbp_sd_error"]),
                "sbp_within_5": numeric(paper_row["sbp_within_5"]),
                "sbp_within_10": numeric(paper_row["sbp_within_10"]),
                "sbp_within_15": numeric(paper_row["sbp_within_15"]),
                "dbp_within_5": numeric(paper_row["dbp_within_5"]),
                "dbp_within_10": numeric(paper_row["dbp_within_10"]),
                "dbp_within_15": numeric(paper_row["dbp_within_15"]),
                "sbp_grade": paper_row["sbp_grade"],
                "dbp_grade": paper_row["dbp_grade"],
            }
        )
    return rows


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_key_desc(row: dict, key: str):
    value = row.get(key)
    return float("-inf") if value is None else float(value)


def sort_key_asc(row: dict, key: str):
    value = row.get(key)
    return float("inf") if value is None else float(value)


def plot_tradeoff(rows: list[dict], out_dir: Path):
    plot_rows = [row for row in rows if row.get("mae_mean") is not None and row.get("macro_f1") is not None]
    if not plot_rows:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [float(row["mae_mean"]) for row in plot_rows]
    ys = [float(row["macro_f1"]) for row in plot_rows]
    ax.scatter(xs, ys, s=80, color="#145a32", alpha=0.9)
    for row in plot_rows:
        ax.annotate(row["label"], (float(row["mae_mean"]), float(row["macro_f1"])), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("MAE Mean (lower is better)")
    ax.set_ylabel("Macro-F1 (higher is better)")
    ax.set_title("Core Protocol Trade-off: Regression vs Classification")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "core_protocol_tradeoff.png", dpi=220)
    plt.close(fig)


def plot_metric_bars(rows: list[dict], out_dir: Path):
    plot_rows = [row for row in rows if row.get("acc") is not None and row.get("macro_f1") is not None]
    if not plot_rows:
        return
    labels = [row["label"] for row in plot_rows]
    acc = [float(row["acc"]) for row in plot_rows]
    macro_f1 = [float(row["macro_f1"]) for row in plot_rows]
    mae = [float(row["mae_mean"]) if row.get("mae_mean") is not None else 0.0 for row in plot_rows]
    x = list(range(len(labels)))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(x, acc, width=0.35, color="#1f618d", label="Accuracy")
    axes[0].bar([v + 0.38 for v in x], macro_f1, width=0.35, color="#b9770e", label="Macro-F1")
    axes[0].set_ylabel("Classification")
    axes[0].set_title("Core Protocol Classification Comparison")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    axes[1].bar(x, mae, width=0.6, color="#7d3c98")
    axes[1].set_ylabel("MAE Mean")
    axes[1].set_title("Core Protocol Regression Comparison")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    axes[1].set_xticks([v + 0.19 for v in x], labels, rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(out_dir / "core_protocol_metric_bars.png", dpi=220)
    plt.close(fig)


def main():
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_audit_rows()

    save_csv(AUDIT_ROOT / "core_protocol_audit_summary.csv", rows)
    (AUDIT_ROOT / "core_protocol_audit_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    f1_rank = sorted(rows, key=lambda row: sort_key_desc(row, "macro_f1"), reverse=True)
    mae_rank = sorted(rows, key=lambda row: sort_key_asc(row, "mae_mean"))
    save_csv(AUDIT_ROOT / "core_protocol_rank_by_macro_f1.csv", f1_rank)
    save_csv(AUDIT_ROOT / "core_protocol_rank_by_mae_mean.csv", mae_rank)

    plot_tradeoff(rows, AUDIT_ROOT)
    plot_metric_bars(rows, AUDIT_ROOT)

    print(f"Audit summary written to: {AUDIT_ROOT}")


if __name__ == "__main__":
    main()
