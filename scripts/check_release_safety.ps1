param(
    [int]$MaxFileSizeMB = 50
)

<#
Pre-upload safety check.

This script scans the release directory for common mistakes before the first
GitHub push: private paths, raw data arrays, model checkpoints, compressed
clinical data, generated output directories, and unexpectedly large files.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$BlockedExtensions = @(".pt", ".pth", ".ckpt", ".npy", ".npz", ".zip", ".tar", ".gz", ".7z")
$BlockedDirs = @("data", "Datasets", "outputs", ".venv", "__pycache__", ".idea", ".vscode", "Papers")
$MaxBytes = $MaxFileSizeMB * 1MB

$Problems = New-Object System.Collections.Generic.List[string]

Get-ChildItem -Recurse -Force -File | ForEach-Object {
    $rel = Resolve-Path -Relative $_.FullName
    if ($_.Length -gt $MaxBytes) {
        $Problems.Add("Large file > ${MaxFileSizeMB}MB: $rel")
    }
    if ($BlockedExtensions -contains $_.Extension) {
        $Problems.Add("Blocked binary/checkpoint extension: $rel")
    }
}

Get-ChildItem -Recurse -Force -Directory | ForEach-Object {
    if ($BlockedDirs -contains $_.Name) {
        $rel = Resolve-Path -Relative $_.FullName
        $Problems.Add("Blocked generated/private directory: $rel")
    }
}

$PrivatePathPatterns = @(
    ("D:" + "\"),
    ("C:" + "\"),
    "/home/",
    "/Users/",
    ("AQM-" + "MedFuse" + "_Project"),
    ("Desk" + "top")
)
$PathPattern = ($PrivatePathPatterns | ForEach-Object { [regex]::Escape($_) }) -join "|"
$SelfPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "check_release_safety.ps1")).Path
$Rg = Get-Command rg -ErrorAction SilentlyContinue
if ($Rg) {
    $PathHits = & rg -n $PathPattern . --glob "!scripts/check_release_safety.ps1" 2>$null
} else {
    $TextExtensions = @(".ps1", ".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".cff")
    $TextFiles = Get-ChildItem -Recurse -Force -File | Where-Object {
        $_.FullName -ne $SelfPath -and (
            $TextExtensions -contains $_.Extension.ToLowerInvariant() -or
            $_.Name -eq ".gitignore" -or
            $_.Name -eq "LICENSE"
        )
    }
    $PathHits = if ($TextFiles) {
        Select-String -Path ($TextFiles | Select-Object -ExpandProperty FullName) -Pattern $PathPattern -ErrorAction SilentlyContinue
    } else {
        @()
    }
}
if ($LASTEXITCODE -eq 0 -and $PathHits) {
    $Problems.Add("Potential private/local path found:`n$PathHits")
}

if ($Problems.Count -gt 0) {
    Write-Host "[release safety] Problems found:" -ForegroundColor Red
    $Problems | ForEach-Object { Write-Host "- $_" -ForegroundColor Red }
    exit 1
}

Write-Host "[release safety] OK: no obvious raw data, checkpoints, private paths, or oversized files found." -ForegroundColor Green
