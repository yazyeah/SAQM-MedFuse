param(
    [string]$DataRoot = "<YOUR_LOCAL_MIMIC_BP_ROOT>",
    [string[]]$Methods = @(
        "ann_lstm_ecg_ppg",
        "ppg_bilstm_attention",
        "bpnet_cnn",
        "mlp_bp_mixer"
    ),
    [switch]$IncludeHeavy,
    [int]$Epochs = 64,
    [int]$Patience = 12,
    [int]$BatchSize = 64,
    [int]$NumWorkers = 0,
    [double]$Lr = 0.0003,
    [double]$ClassWeightPower = 0.45,
    [double]$ClsLossWeight = 0.25,
    [double]$TailLossWeight = 0.20,
    [ValidateSet("regression", "head")]
    [string]$ClassificationSource = "regression",
    [ValidateSet("none", "subject_offset", "subject_affine")]
    [string]$SupportCalibration = "subject_affine",
    [double]$SupportShrinkage = 0.85,
    [string]$Device = "cuda",
    [int]$Seed = 42,
    [string]$Python = "python"
)

<#
Representative baseline comparison launcher.

The comparison models use the same subject-disjoint few-shot protocol produced
by the SAQM-MedFuse main run. Run `run_v10_18_main.ps1` first, then run this
script to train paper-facing baselines under the same split and calibration
setting.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($DataRoot -eq "<YOUR_LOCAL_MIMIC_BP_ROOT>") {
    throw "Replace -DataRoot with your local MIMIC-BP path."
}

$env:AQM_MIMIC_BP_ROOT = $DataRoot

if ($IncludeHeavy) {
    $Methods = $Methods + @("piso_transformer", "mufubp_dual_feature_pfe")
}

$ProtocolSource = "mimic_bp_reg_v10_18_subjectdisjoint_baselineguard_crisisrepair_proto"

foreach ($Method in $Methods) {
    $OutputName = "mimic_bp_compare_v10_18_${Method}_proto"
    Write-Host "[SAQM-MedFuse comparison] Running $Method -> $OutputName"
    & $Python -B .\run_mimic_bp_v10_13_comparison_experiment.py `
        --method $Method `
        --protocol-source $ProtocolSource `
        --output-name $OutputName `
        --epochs $Epochs `
        --patience $Patience `
        --batch-size $BatchSize `
        --num-workers $NumWorkers `
        --lr $Lr `
        --class-weight-power $ClassWeightPower `
        --cls-loss-weight $ClsLossWeight `
        --tail-loss-weight $TailLossWeight `
        --classification-source $ClassificationSource `
        --support-calibration $SupportCalibration `
        --support-shrinkage $SupportShrinkage `
        --device $Device `
        --seed $Seed
}

& $Python -B .\summarize_v10_18_paper_runs.py
Write-Host "[SAQM-MedFuse comparison] Summary written to outputs\v10_18_paper_summary\comparison_summary.csv"
