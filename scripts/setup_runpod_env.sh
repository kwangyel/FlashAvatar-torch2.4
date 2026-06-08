git clone --recurse-submodules https://github.com/kwangyel/FlashAvatar-torch2.4.git
python -m pip install -U pip setuptools wheel
pip install ninja

# Install torch 2.4 CUDA 12.4 build.
# python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# User-requested PyTorch3D installation path.
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

cd FlashAvatar-torch2.4
# Project Python dependencies.
python -m pip install --no-build-isolation -r requirements-runpod.txt

pip install gdown
gdown https://drive.google.com/file/d/1455XCYSZWmfILCYKS8IPBwKJaIF-zsri/view?usp=sharing -O FLAME2020.zip
gdown https://drive.google.com/file/d/1c9GPL7K7vgVDEHhOxXZrr08P4KP4Ef4R/view?usp=sharing -O FLAME_masks.zip
gdown https://drive.google.com/file/d/1QtBlUS0OLqyenEh-8uhngM2eaoppqsGK/view?usp=sharing -O masks.tar.gz
gdown https://drive.google.com/file/d/1T8OK_zuhv0BbJLC6sHmmQPOSQ7lTgGC8/view?usp=sharing -O dataset.tar.gz
gdown https://drive.google.com/file/d/1rDHSHglFoBBTh9vqjEq5SGH3HkN3rVwq/view?usp=sharing -O metrical-tracker.zip

apt-get update
apt-get install unzip

mkdir -p FLAME2020
mkdir -p FLAME_masks
mkdir -p masks
mkdir -p dataset
mkdir -p metrical-tracker

unzip -q FLAME2020.zip -d ./FLAME2020
unzip -q FLAME_masks.zip -d ./FLAME_masks
tar -xzf masks.tar.gz -C ./masks
tar -xzf dataset.tar.gz -C ./dataset
unzip -q metrical-tracker.zip -d ./metrical-tracker
mkdir -p flame/FLAME_masks

mv FLAME2020/FLAME2020/generic_model.pkl flame/generic_model.pkl
mv FLAME_masks/FLAME_masks.pkl flame/FLAME_masks/FLAME_masks.pkl


# Build CUDA extensions in the active torch environment.
python -m pip install --no-build-isolation -e submodules/simple-knn
python -m pip install --no-build-isolation -e submodules/diff-gaussian-rasterization

pip install yacs
python scripts/verify_cuda_extensions.py

# for renaminig the dataset since i messed up on the name of the dataset
cd ~/FlashAvatar-torch2.4/dataset/Obama/parsing
# actually rename
for f in *_headneck.png; do mv "$f" "${f/_headneck/_neckhead}"; done

cd ~/FlashAvatar-torch2.4