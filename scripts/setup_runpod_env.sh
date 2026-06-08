git clone --recurse-submodules https://github.com/kwangyel/FlashAvatar-torch2.4.git
python -m pip install -U pip setuptools wheel
pip install ninja

# Install torch 2.4 CUDA 12.4 build.
# python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# User-requested PyTorch3D installation path.
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Project Python dependencies.
python -m pip install --no-build-isolation -r requirements-runpod.txt
pip install gdown
gdown https://drive.google.com/file/d/1LRf0fdlGDq_gTiJysaCmtub6EUodC6Y0/view?usp=sharing -O kinley.zip
gdown https://drive.google.com/file/d/1455XCYSZWmfILCYKS8IPBwKJaIF-zsri/view?usp=sharing -O FLAME2020.zip
gdown https://drive.google.com/file/d/1c9GPL7K7vgVDEHhOxXZrr08P4KP4Ef4R/view?usp=sharing -O FLAME_masks.zip

apt-get update
apt-get install unzip

mkdir -p kinley
mkdir -p FLAME2020
mkdir -p FLAME_masks

unzip -d kinley.zip -q ./kinley
unzip -q FLAME2020.zip -d ./FLAME2020
unzip -q FLAME_masks.zip -d ./FLAME_masks

mv kinley/kinley/dataset dataset
mv kinley/kinley/metrical-tracker metrical-tracker

mv FLAME2020/FLAME2020/generic_model.pkl flame/generic_model.pkl
mv FLAME_masks/FLAME_masks.pkl flame/FLAME_masks/FLAME_masks.pkl


# Build CUDA extensions in the active torch environment.
python -m pip install --no-build-isolation -e submodules/simple-knn
python -m pip install --no-build-isolation -e submodules/diff-gaussian-rasterization

pip install yacs
python scripts/verify_cuda_extensions.py

# for renaminig the dataset since i messed up on the name of the dataset
cd ~/FlashAvatar-torch2.4/dataset/kinley/parsing
# actually rename
for f in *_headneck.png; do mv "$f" "${f/_headneck/_neckhead}"; done

cd ~/FlashAvatar-torch2.4