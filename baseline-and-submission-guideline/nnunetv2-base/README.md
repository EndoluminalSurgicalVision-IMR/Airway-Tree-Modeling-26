# ATM26 nnU-Net v2.6.4 base image — CUDA 11.8 (slim)

Source for the `atm26-nnunetv2:2.6.4-cuda11.8` Docker Hub tag used by the
batch baselines (Validation & Final Test phases).

**Why this exists:** the evaluation platform's GPU driver supports CUDA up to
12.0, so submission images must be built on a CUDA ≤ 12.0 base (cuda12.4
images will not run there). Naively building from a PyTorch image + installing
requirements through a generic PyPI mirror lets the resolver stack a second,
cu124 torch on top of the base (~17 GB image). This recipe installs
`torch==2.5.1+cu118` from the official cu118 wheel index **first** and starts
from the minimal `nvidia/cuda:11.8.0-runtime-ubuntu22.04` — the result is
~7 GB.

```bash
cd nnunetv2-base
docker build -f Dockerfile.cu118 -t atm26-nnunetv2:2.6.4-cuda11.8 .
```

Build context: the `requirements.txt` in this folder (pinned extras —
`monai==1.3.2` because monai 1.4+ demands torch>=2.6, which does not exist
for cu118) plus the nnU-Net v2.6.4 source tree (`nnunetv2/`, `setup.py`,
`pyproject.toml`, its `LICENSE`) from https://github.com/MIC-DKFZ/nnUNet at
v2.6.4.
