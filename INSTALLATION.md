# Installation

We use the `pointact` environment for model training, checkpoint loading, data preprocessing, and the policy server; use simulator-specific environments for environment rollouts and evaluation clients.


## PointAct Environment

```bash
git clone https://github.com/cshizhe/PointAct.git
cd PointAct

conda create -n pointact python=3.10 -y
conda activate pointact

conda install conda-forge::evdev

# Install a CUDA toolkit in the env if CUDA is not provided by the system.
conda install nvidia/label/cuda-12.8.0::cuda-toolkit -y
# export CUDA_HOME=$CONDA_PREFIX

# Install PointAct and the Python dependencies declared in pyproject.toml.
pip install -e .

conda install -c conda-forge ffmpeg=6 -y

pip install ipython

# Point Transformer V3 / PointAct 3D backbone dependencies.
FORCE_CUDA=1 pip install spconv-cu120
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu126.html

pip install --no-build-isolation \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Run this after installing the core environment:

```bash
python - <<'PY'
import torch
import flash_attn
import spconv.pytorch as spconv
import torch_scatter
import open3d
import pointact

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("pointact import ok")
PY
```

## Pretrained Backbones

Download the required checkpoint, place it under your preferred storage path, and update the script argument accordingly. 

On offline clusters, download models before setting `TRANSFORMERS_OFFLINE=1`.

```bash
# Point transformer v3: concerto
hf download --repo-type model Pointcept/Concerto \
  --local-dir $SCRATCH/datasets/pretrained/Pointcept-Concerto

# Point transformer v3: utonia
hf download --repo-type model Pointcept/Utonia \
  --local-dir $SCRATCH/datasets/pretrained/Pointcept-Utonia

# Qwen 2.5
hf download --repo-type model Qwen/Qwen2.5-VL-3B-Instruct
```


## Simulator Environments

Do not install simulator packages into the `pointact` environment. Keep one conda environment per simulator, then use server-client evaluation when needed: the PointAct policy server runs in `pointact`, and the simulator client runs in the simulator environment.

| Simulator / platform | Environment | Installation and experiment notes |
| --- | --- | --- |
| LIBERO | `libero` | [experiments/2_libero/README.md](experiments/2_libero/README.md) |
| RLBench | `rlbench` | [experiments/10_rlbench/README.md](experiments/10_rlbench/README.md) |
| RoboCASA365 | `robocasa365` | [experiments/13_robocasa365/README.md](experiments/13_robocasa365/README.md) |
