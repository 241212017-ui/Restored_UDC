# Run everything that depends on trained checkpoints.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
Set-Location "D:\UDC_AMFM_Project"

Write-Output "### Stage C: full-resolution fine-tuning ###"
powershell -ExecutionPolicy Bypass -File scripts\run_finetune.ps1

Write-Output "### evaluation ###"
powershell -ExecutionPolicy Bypass -File scripts\run_eval.ps1

Write-Output "### figures ###"
python scripts\make_figures.py

Write-Output "### report ###"
python report\make_report.py

Write-Output "### slides ###"
python slides\make_slides.py

Write-Output "### tests ###"
python -m pytest -q tests

Write-Output "ALL DONE"
