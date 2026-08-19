# T2_baseline_finaltest_batch

The official **Track-2 (branch-wise anatomical labeling)** leaderboard baseline repackaged for the **Final Test Phase** with the ATM26 local evaluation platform's batch contract.

## What is different from `T2_baseline`

| | `T2_baseline` | `T2_baseline_finaltest_batch` |
|---|---|---|
| Execution | One container per case | `LABEL org.atm26.batch="1"`: ONE container reads the whole test set from `/input/images/lung-ct` and loops over every case internally |
| Model loading | Once per case | Once per submission (big wall-clock win) |
| Outputs | `/output/images/multi-class-airway-segmentation/output.mha` | `/output/images/multi-class-airway-segmentation/<case-id>.mha` per case |
| Grand-Challenge compatibility | Yes | Yes — with a single case at `/input` the dual-mode wrapper behaves exactly like the per-case baseline |

The prediction core is unchanged: same nnU-Net preprocessing, same GPU patch inference, same low-memory argmax (Track 2). GPU-VRAM/memory hygiene in the loop: `torch.cuda.empty_cache()`, `gc.collect()`, per-case `del` of arrays and images.

## Checkpoints

The `resources/` folder here is a placeholder. Download the leaderboard baseline checkpoints from the [challenge Huggingface](https://huggingface.co/Endoluminal-Surg-Vision-IMR/Airway-Tree-Modeling-26-baselines) and place them under `resources/nnUNet_ckpts/` (same layout as `T2_baseline/resources`).

## Build, test, save

```bash
cd baseline-and-submission-guideline/T2_baseline_finaltest_batch
bash do_test_run.sh   # single-case and/or batch smoke test (needs test inputs)
bash do_save.sh       # timestamped .tar.gz for upload
```

The `do_test_run.sh` exercises both modes when inputs exist:
`test/input/case1` (one `.mha`) for the per-case path and `test/input/batch`
(multiple `.mha` files) for the batch path.

## CUDA requirement

The evaluation platform's GPU driver supports **CUDA up to 12.0**. Build from the
`atm26-nnunetv2:2.6.4-cuda11.8` base image (images built on a cuda12.4 base will
NOT run there). The base's own Dockerfile lives in `../nnunetv2-base/` — it is
the SLIM recipe (~7 GB): minimal `nvidia/cuda:11.8.0-runtime` + torch installed
from the official cu118 wheel index FIRST, so later installs never stack a
second (cu124) torch on top.
