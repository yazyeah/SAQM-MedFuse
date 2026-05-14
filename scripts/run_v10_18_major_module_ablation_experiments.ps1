param(
    [string]$DataRoot = "<YOUR_LOCAL_MIMIC_BP_ROOT>",
    [ValidateSet(
        "major_full",
        "no_quality_sparse_fusion",
        "no_uncertainty_credibility",
        "no_anchor_decision_refinement",
        "no_tail_safety_calibration",
        "no_personalized_conformal"
    )]
    [string]$Variant = "",
    [string]$Python = "python",
    [int]$HeadEpochs = 24,
    [int]$HeadPatience = 6,
    [int]$HeadMinEpochs = 8,
    [string]$RunTag = "paperfix",
    [switch]$ForceTrainEachVariant,
    [switch]$RerunExisting,
    [switch]$SummarizeOnly,
    [switch]$DryRun
)

<#
Major-module ablation launcher.

This script reproduces the paper-facing ablation table. The default protocol is
designed for practical runtimes: the full reference and the quality/sparse
fusion proxy retrain the lightweight guided feature head, while downstream
decision, safety, uncertainty, and personalization ablations share the
major_full head and rerun the affected search/evaluation stages. Use
`-ForceTrainEachVariant` only if you explicitly want fully independent heads.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($DataRoot -eq "<YOUR_LOCAL_MIMIC_BP_ROOT>") {
    throw "Replace -DataRoot with your local MIMIC-BP path."
}

$env:AQM_MIMIC_BP_ROOT = $DataRoot
$env:PYTHONDONTWRITEBYTECODE = "1"

$PyArgs = @(
    "-B",
    ".\run_aqm_medfuse_v10_18_lite_train_ablation.py",
    "--variant-set",
    "major",
    "--head-epochs",
    $HeadEpochs,
    "--head-patience",
    $HeadPatience,
    "--head-min-epochs",
    $HeadMinEpochs
)

if ($Variant -ne "") {
    $PyArgs += @("--variant", $Variant)
}
if ($RunTag -ne "") {
    $PyArgs += @("--run-tag", $RunTag)
}
if ($ForceTrainEachVariant) {
    $PyArgs += "--force-train-each-variant"
}
if ($RerunExisting) {
    $PyArgs += "--rerun-existing"
}
if ($SummarizeOnly) {
    $PyArgs += "--summarize-only"
}
if ($DryRun) {
    $PyArgs += "--dry-run"
}

Write-Host "[SAQM-MedFuse ablation] Running major-module suite $Variant"
& $Python @PyArgs

if (-not $DryRun) {
    & $Python -B .\summarize_v10_18_paper_runs.py
    Write-Host "[SAQM-MedFuse ablation] Summary written to outputs\v10_18_paper_summary\major_module_ablation_summary.csv"
}
