param(
    [string]$SummaryDir = ".\results\paper_summary"
)

<#
Aggregate-result completion check.

This script checks whether included paper summary CSV files contain incomplete
rows. It is intentionally conservative: any row with a `completed` column that
is not `True` is reported before upload.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$CsvFiles = Get-ChildItem -Path $SummaryDir -Filter "*.csv" -File -ErrorAction SilentlyContinue
$Problems = New-Object System.Collections.Generic.List[string]

foreach ($Csv in $CsvFiles) {
    $Rows = Import-Csv $Csv.FullName
    foreach ($Row in $Rows) {
        $Props = $Row.PSObject.Properties.Name
        if ($Props -contains "completed") {
            $Value = [string]$Row.completed
            if ($Value -and $Value -ne "True" -and $Value -ne "true" -and $Value -ne "1") {
                $Name = if ($Props -contains "name") { $Row.name } else { "<unknown>" }
                $Problems.Add("$($Csv.Name): row '$Name' has completed=$Value")
            }
        }
    }
}

if ($Problems.Count -gt 0) {
    Write-Host "[experiment completion] Incomplete rows found:" -ForegroundColor Yellow
    $Problems | ForEach-Object { Write-Host "- $_" -ForegroundColor Yellow }
    exit 1
}

Write-Host "[experiment completion] OK: included completed flags are all true." -ForegroundColor Green
