# Evaluate every trained model on the test split (per-dataset + combined metrics).
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
Set-Location "D:\UDC_AMFM_Project"
$logs = "outputs\runlogs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$jobs = @(
    @("abl_none",   "configs\ablation_none.yaml"),
    @("abl_concat", "configs\ablation_concat.yaml"),
    @("abl_dgf",    "configs\dgf.yaml"),
    @("abl_amfm",   "configs\amfm.yaml"),
    @("dgf_final",  "configs\dgf.yaml"),
    @("dgf_ft",     "configs\finetune_dgf.yaml"),
    @("amfm_ft",    "configs\finetune_amfm.yaml"),
    @("dgf_cont",   "configs\cont_dgf.yaml"),
    @("amfm_warm",  "configs\cont_amfm.yaml")
)
foreach ($j in $jobs) {
    $name = $j[0]; $cfg = $j[1]
    $ckpt = "outputs\checkpoints\$name\best_psnr.pth"
    if (Test-Path $ckpt) {
        Write-Output "=== evaluating $name ==="
        python -u -m src.evaluate --config $cfg --checkpoint $ckpt --name $name `
            --split test *> "$logs\eval_$name.log"
    } else {
        Write-Output "=== skip $name (no checkpoint) ==="
    }
}

# Test-time augmentation (flip ensembling) for the two best models.
foreach ($t in @(@("amfm_ft", "configs\finetune_amfm.yaml"), @("dgf_ft", "configs\finetune_dgf.yaml"))) {
    $name = $t[0]; $cfg = $t[1]
    $ckpt = "outputs\checkpoints\$name\best_psnr.pth"
    if (Test-Path $ckpt) {
        Write-Output "=== evaluating $name + TTA ==="
        python -u -m src.evaluate --config $cfg --checkpoint $ckpt `
            --name "${name}_tta" --split test --tta *> "$logs\eval_${name}_tta.log"
    }
}
Write-Output "ALL EVAL DONE"
