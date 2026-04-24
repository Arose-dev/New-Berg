# Windows setup for LEGO Benchmark - installs SAM2, UniDepth, GroundingDINO and checkpoints
# Run from project root: .\setup.ps1
# Requires: Python 3.10 or 3.11 (SAM2/UniDepth need >=3.10). Install from https://www.python.org/downloads/

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }

# Prefer Python 3.10 or 3.11 (required by SAM2/UniDepth); fallback to default
$pyExe = $null
foreach ($v in @("-3.11", "-3.10", "")) {
    $arg = if ($v) { @($v, "-c", "import sys") } else { @("-c", "import sys") }
    try { $null = & py @arg 2>$null; if ($LASTEXITCODE -eq 0) { $pyExe = if ($v) { "py", $v } else { "py" }; break } } catch {}
}
if (-not $pyExe) { $pyExe = @("py") }
function Run-Py { & $pyExe[0] @($pyExe[1..($pyExe.Length-1)] + $args) }
Write-Host "Using: $($pyExe -join ' ')"

Push-Location $ProjectRoot

# 1. Create models directory
if (-not (Test-Path "models")) { New-Item -ItemType Directory -Path "models" | Out-Null }
Set-Location "models"

# 2. SAM2
if (-not (Test-Path "sam2")) {
    Write-Host "Cloning SAM2..."
    git clone https://github.com/facebookresearch/sam2.git
}
Set-Location "sam2"
Write-Host "Installing SAM2..."
Run-Py -m pip install -e . --quiet
Set-Location "checkpoints"
$base = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
$ckpt = "sam2.1_hiera_base_plus.pt"
if (-not (Test-Path $ckpt)) {
    Write-Host "Downloading SAM2 checkpoint ($ckpt)..."
    Invoke-WebRequest -Uri "$base/$ckpt" -OutFile $ckpt -UseBasicParsing
}
Set-Location (Join-Path $ProjectRoot "models")

# 3. UniDepth
if (-not (Test-Path "UniDepth")) {
    Write-Host "Cloning UniDepth..."
    git clone https://github.com/lpiccinelli-eth/UniDepth.git
}
Set-Location "UniDepth"
Write-Host "Installing UniDepth..."
Run-Py -m pip install -e . --quiet
Set-Location (Join-Path $ProjectRoot "models")

# 4. GroundingDINO
if (-not (Test-Path "GroundingDINO")) {
    Write-Host "Cloning GroundingDINO..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git
}
Set-Location "GroundingDINO"
Write-Host "Installing GroundingDINO..."
Run-Py -m pip install -e . --quiet
if (-not (Test-Path "weights")) { New-Item -ItemType Directory -Path "weights" | Out-Null }
Set-Location "weights"
$dinoUrl = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
if (-not (Test-Path "groundingdino_swint_ogc.pth")) {
    Write-Host "Downloading GroundingDINO weights..."
    Invoke-WebRequest -Uri $dinoUrl -OutFile "groundingdino_swint_ogc.pth" -UseBasicParsing
}
Set-Location $ProjectRoot

Write-Host "Installing requirements..."
Run-Py -m pip install -r requirements.txt --quiet

Pop-Location
Write-Host "Setup complete. Ensure api.key contains your OpenAI API key and data/ has annotations and images."
