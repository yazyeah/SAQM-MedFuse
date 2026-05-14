param(
    [string]$DataRoot = "<YOUR_LOCAL_MIMIC_BP_ROOT>",
    [string]$Python = "python",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

<#
SAQM-MedFuse main experiment launcher.

This wrapper keeps private machine paths out of the Python source code. The
published code expects users to place the MIMIC-BP files locally and pass the
root through `-DataRoot` or the AQM_MIMIC_BP_ROOT environment variable.

The experiment writes generated artifacts to `outputs/` under the repository
root. Those generated files are intentionally ignored by Git.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($DataRoot -eq "<YOUR_LOCAL_MIMIC_BP_ROOT>") {
    throw "Replace -DataRoot with your local MIMIC-BP path, for example: -DataRoot <YOUR_LOCAL_MIMIC_BP_ROOT>"
}

$env:AQM_MIMIC_BP_ROOT = $DataRoot
$env:CUDA_VISIBLE_DEVICES = if ($Device -eq "cuda") { $env:CUDA_VISIBLE_DEVICES } else { "" }

Write-Host "[SAQM-MedFuse] Data root: $env:AQM_MIMIC_BP_ROOT"
Write-Host "[SAQM-MedFuse] Running main v10.18 protocol"

& $Python -B .\train_aqm_medfuse_mimic_bp_reg_v10_18_subjectdisjoint_baselineguard_crisisrepair_protocol.py

& $Python -B .\summarize_v10_18_paper_runs.py
Write-Host "[SAQM-MedFuse] Summary written to outputs\v10_18_paper_summary"
