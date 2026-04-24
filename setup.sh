mkdir -p models
cd models

git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd checkpoints && \
./download_ckpts.sh && \
cd ../..

git clone https://github.com/lpiccinelli-eth/UniDepth.git && cd UniDepth
pip install -e .
cd ..

pip install torch==2.2.0 torchvision==0.17.0 xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu122 --force-reinstall

git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO/
pip install -e .
mkdir weights && cd weights && \
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../../..

pip install -r requirements.txt