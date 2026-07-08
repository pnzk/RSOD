$ErrorActionPreference = "Stop"

param(
  [string]$ProjectRoot = "${env:PROJECT_ROOT}",
  [string]$ConfigRoot = "${env:PROJECT_ROOT}\schemes\rsod_main_matrix",
  [string]$RunnerScript = "${env:PROJECT_ROOT}\schemes\rsod_main_matrix\run_matrix_16.ps1"
)

if (-not $ProjectRoot) {
  throw "Set PROJECT_ROOT or pass -ProjectRoot."
}

Write-Host "Project root: $ProjectRoot"
Write-Host "Runner script: $RunnerScript"
Write-Host "Before running, set PROJECT_ROOT, DATA_ROOT, WORK_DIR, and CLIP_ROOT according to docs/REPRODUCIBILITY.md."

powershell -ExecutionPolicy Bypass -File $RunnerScript
