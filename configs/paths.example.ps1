# Example local path configuration for SAQM-MedFuse.
# Copy this file to `configs/paths.local.ps1`, edit the placeholders, and
# dot-source it before running experiments:
#
#   . .\configs\paths.local.ps1
#
# Do not commit `paths.local.ps1` if it contains private machine paths.

$env:AQM_MIMIC_BP_ROOT = "<YOUR_LOCAL_MIMIC_BP_ROOT>"
$env:AQM_OUTPUT_ROOT = "<YOUR_LOCAL_OUTPUT_ROOT_OR_LEAVE_EMPTY>"

# Optional: force a device for scripts that expose a `-Device` argument.
# Examples: "cuda", "cuda:0", or "cpu".
$env:AQM_DEVICE = "cuda"
