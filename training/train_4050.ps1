param(
  [ValidateSet("lp", "partial", "finetune")] [string]$Adaptation = "lp",
  [int]$Epochs = 20,
  [int]$BatchSize = 64,
  [int]$UnfreezeBlocks = 4,
  [string]$Task = "aptos_binary_lp",
  [string]$Weights = "C:\Users\Manan\Downloads\RETFound_mae_natureCFP.pth"
)

$ErrorActionPreference = "Stop"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$root = "D:\HTH\training\RETFound-main"
$python = "D:\HTH\training\.venv\Scripts\python.exe"

$extra = @()
if ($Adaptation -eq "partial") {
  $extra += @("--unfreeze_blocks", "$UnfreezeBlocks", "--set_grad_checkpointing")
}

Push-Location $root
try {
  & $python main_finetune.py `
    --model RETFound_mae --finetune $Weights `
    --adaptation $Adaptation --nb_classes 2 --input_size 224 `
    --data_path ./data --output_dir ./output_dir --task $Task `
    --batch_size $BatchSize --epochs $Epochs --num_workers 4 @extra
}
finally {
  Pop-Location
}
