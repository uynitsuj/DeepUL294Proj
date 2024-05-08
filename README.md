# Foveated CV

## Prerequisites 
Have Conda installed on Ubuntu.

## Installation (Bash Script)

Clone the repo:
```
cd ~/
git clone https://github.com/uynitsuj/DeepUL294Proj.git
```

Run the setup bash script
```
bash env_setup.bash
```

## Installation (Step by Step)
### Step 1: Setup Conda environment

```
conda create -n fov python=3.11.8 -y
conda activate fov
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install tqdm==4.66.2
pip install datasets==2.18.0
pip install matplotlib==3.8.0
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0%2Bcu118.html
pip install torch-geometric==2.5.2
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0%2Bcu118.html
pip install timm==0.9.16
cd DUL294P/lib/pointops2
python setup.py install
pip install open-clip-torch
```