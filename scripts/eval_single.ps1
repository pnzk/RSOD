$ErrorActionPreference = "Stop"

param(
  [Parameter(Mandatory=$true)][string]$ProjectRoot,
  [Parameter(Mandatory=$true)][string]$Config,
  [Parameter(Mandatory=$true)][string]$Checkpoint,
  [string]$Python = "python"
)

Set-Location $ProjectRoot

& $Python -m tools.detection.test $Config $Checkpoint --eval mAP --cfg-options `
  data.test_dataloader.workers_per_gpu=0 `
  data.val_dataloader.workers_per_gpu=0
