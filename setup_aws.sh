#!/bin/bash
# =============================================================
# AWS GPU Instance Setup Script for the LEGO spatial pipeline (LINUX)
# =============================================================
# For WINDOWS AWS instances, use setup_aws.ps1 instead!
#
# Run this script on your AWS Linux GPU instance after SSH-ing in.
#
# Prerequisites:
#   - AWS instance with NVIDIA GPU (g5.2xlarge or g4dn.xlarge recommended)
#   - Ubuntu 22.04 AMI with NVIDIA drivers and CUDA toolkit
#
# Usage:
#   chmod +x setup_aws.sh
#   ./setup_aws.sh
# =============================================================

set -e

echo "============================================"
echo "Setting up the LEGO spatial pipeline on AWS"
echo "============================================"

# 1. System packages
echo "Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y git python3-pip python3-venv wget unzip

# 2. Create and activate virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv ~/lego-spatial-env
source ~/lego-spatial-env/bin/activate

# 3. Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Install pipeline dependencies
echo "Installing pipeline dependencies..."
pip install -r requirements.txt

# 5. Install additional dependencies for LEGO adapter
pip install pandas

# 6. Set up vision models
echo "Setting up vision models..."
mkdir -p models
cd models

# SAM2
if [ ! -d "sam2" ]; then
    echo "Cloning SAM2..."
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    pip install -e .
    mkdir -p checkpoints
    cd checkpoints
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
    cd ../..
fi

# UniDepth
if [ ! -d "UniDepth" ]; then
    echo "Cloning UniDepth..."
    git clone https://github.com/lpiccinelli-eth/UniDepth.git
    cd UniDepth
    pip install -e .
    cd ..
fi

# GroundingDINO
if [ ! -d "GroundingDINO" ]; then
    echo "Cloning GroundingDINO..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO
    pip install -e .
    mkdir -p weights
    cd weights
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    cd ../..
fi

cd ..

# 7. Download LEGO benchmark data
echo "Downloading LEGO benchmark data..."
python -m datasets.lego_dataset --output-dir data/lego

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To run the LEGO spatial pipeline:"
echo "  source ~/lego-spatial-env/bin/activate"
echo ""
echo "  # Test with 5 questions first:"
echo "  python run_lego.py --max-questions 5"
echo ""
echo "  # Run LEGO-Lite (220 questions):"
echo "  python run_lego.py"
echo ""
echo "  # Run full benchmark (1100 questions):"
echo "  python run_lego.py --full"
echo ""
echo "Set PARASAIL_API_KEY and optional PARASAIL_MODEL_NAME before you run it."
echo "============================================"
