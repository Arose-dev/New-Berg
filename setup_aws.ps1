# =============================================================
# AWS Windows GPU Instance Setup Script for LEGO Benchmark
# =============================================================
# Run this script on your Windows AWS GPU instance (e.g. g5.2xlarge with Windows AMI).
#
# Prerequisites:
#   - AWS instance with NVIDIA GPU and Windows Server 2022
#   - Python 3.10 or 3.11 installed (https://www.python.org/downloads/)
#   - Git installed (https://git-scm.com/download/win)
#   - NVIDIA drivers + CUDA toolkit installed
#
# Usage:
#   .\setup_aws.ps1
# =============================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }

# Find Python
$pyExe = $null
foreach ($v in @("-3.11", "-3.10", "")) {
    $arg = if ($v) { @($v, "-c", "import sys") } else { @("-c", "import sys") }
    try { $null = & py @arg 2>$null; if ($LASTEXITCODE -eq 0) { $pyExe = if ($v) { "py", $v } else { "py" }; break } } catch {}
}
if (-not $pyExe) { $pyExe = @("python") }
function Run-Py { & $pyExe[0] @($pyExe[1..($pyExe.Length-1)] + $args) }
Write-Host "Using Python: $($pyExe -join ' ')"

# Verify CUDA is available
Write-Host "Checking CUDA availability..."
Run-Py -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: PyTorch not installed or CUDA not detected. Installing PyTorch with CUDA..."
    Run-Py -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
}

Push-Location $ProjectRoot

# 1. Install base requirements
Write-Host "`nInstalling base requirements..."
Run-Py -m pip install -r requirements.txt --quiet
Run-Py -m pip install pandas --quiet

# 2. Set up vision models (SAM2, UniDepth, GroundingDINO)
Write-Host "`nSetting up vision models..."
if (-not (Test-Path "models")) { New-Item -ItemType Directory -Path "models" | Out-Null }
Set-Location "models"

# SAM2
if (-not (Test-Path "sam2")) {
    Write-Host "Cloning SAM2..."
    git clone https://github.com/facebookresearch/sam2.git
}
Set-Location "sam2"
Write-Host "Installing SAM2..."
Run-Py -m pip install -e . --quiet
if (-not (Test-Path "checkpoints")) { New-Item -ItemType Directory -Path "checkpoints" | Out-Null }
Set-Location "checkpoints"
$ckpt = "sam2.1_hiera_base_plus.pt"
if (-not (Test-Path $ckpt)) {
    Write-Host "Downloading SAM2 checkpoint..."
    Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/$ckpt" -OutFile $ckpt -UseBasicParsing
}
Set-Location (Join-Path $ProjectRoot "models")

# UniDepth
if (-not (Test-Path "UniDepth")) {
    Write-Host "Cloning UniDepth..."
    git clone https://github.com/lpiccinelli-eth/UniDepth.git
}
Set-Location "UniDepth"
Write-Host "Installing UniDepth..."
Run-Py -m pip install -e . --quiet
Set-Location (Join-Path $ProjectRoot "models")

# GroundingDINO
if (-not (Test-Path "GroundingDINO")) {
    Write-Host "Cloning GroundingDINO..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git
}
Set-Location "GroundingDINO"
Write-Host "Installing GroundingDINO..."
Run-Py -m pip install -e . --quiet
if (-not (Test-Path "weights")) { New-Item -ItemType Directory -Path "weights" | Out-Null }
Set-Location "weights"
if (-not (Test-Path "groundingdino_swint_ogc.pth")) {
    Write-Host "Downloading GroundingDINO weights..."
    Invoke-WebRequest -Uri "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" -OutFile "groundingdino_swint_ogc.pth" -UseBasicParsing
}
Set-Location $ProjectRoot

# 3. Re-install PyTorch with CUDA (in case model installs overwrote it)
Write-Host "`nEnsuring PyTorch has CUDA support..."
Run-Py -m pip install torch==2.2.0 torchvision==0.17.0 xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --quiet

# 4. Download LEGO benchmark data
Write-Host "`nDownloading LEGO benchmark data..."
Run-Py -m datasets.lego_dataset --output-dir data/lego

# 5. Final CUDA check
Write-Host "`n============================================"
Write-Host "Verifying CUDA setup..."
Run-Py -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE!'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

Write-Host "`n============================================"
Write-Host "Setup complete!"
Write-Host "============================================"
Write-Host ""
Write-Host "To run the LEGO Benchmark:"
Write-Host ""
Write-Host "  # Make sure api.key has your OpenAI key"
Write-Host "  # Test with 5 questions:"
Write-Host "  py -3.11 run_lego.py --max-questions 5"
Write-Host ""
Write-Host "  # LEGO-Lite (220 questions):"
Write-Host "  py -3.11 run_lego.py --lite"
Write-Host ""
Write-Host "  # Full benchmark (1100 questions):"
Write-Host "  py -3.11 run_lego.py"
Write-Host "============================================"

Pop-Location
