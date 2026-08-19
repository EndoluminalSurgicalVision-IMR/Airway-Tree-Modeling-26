## Baseline and Submission Guideline

![Challenge Banner](../assets/logo_v6_01.png)

We provide the official example algorithm and our baseline algorithm in this directory. Participants are expected to pack their own solutions, configure their docker image correctly and then build the docker image.

Baseline checkpoints can be found on [Huggingface](https://huggingface.co/Endoluminal-Surg-Vision-IMR/Airway-Tree-Modeling-26-baselines). You can download and replace the corresponding files and folders.

If you would like to use the nnUNet docker, you can pull from Docker Hub:
```bash
docker pull kkdls19/atm26-nnunetv2:2.6.4
```

### Track-1: Binary Airway Segmentation
An example algorithm for Track-1 can be found under `T1_example_algorithm`. It defines the I/O interface for accessing lung CT images on Grand Challenge platform.

The corresponding baseline algorithm for Track-1 can be found under `T1_baseline`. It follows the I/O interface provided by Grand Challenge platform and pack an nnUNet model as solution.

Please carefully configure the `Dockerfile` before docker image saving and testing.

To build and test the docker, you can run:
```bash
cd baseline-and-submission-guideline/T1_baseline
bash do_test_run.sh
```

To build and save the docker, you can run:
```bash
cd baseline-and-submission-guideline/T1_baseline
bash do_save.sh
```

### Track-2: Branch-wise Anatomical Labeling
An example algorithm for Track-2 can be found under `T2_example_algorithm`. It defines the I/O interface for accessing lung CT images on Grand Challenge platform.

The corresponding baseline algorithm for Track-2 can be found under `T2_baseline`. It follows the I/O interface provided by Grand Challenge platform and pack an nnUNet model as solution.

Please carefully configure the `Dockerfile` before docker image saving and testing.

To build and test the docker, you can run:
```bash
cd baseline-and-submission-guideline/T2_baseline
bash do_test_run.sh
```

To build and save the docker, you can run:
```bash
cd baseline-and-submission-guideline/T2_baseline
bash do_save.sh
```
### Final Test Phase (batch execution)

For the Final Test Phase, three additional folders are provided:

- `finaltest_batch_template` — a generic Docker packing template for the ATM26 local evaluation platform's batch contract. On the real Grand-Challenge platform the container receives ONE case and behaves like a standard per-case algorithm; on our ATM26 local inference + evaluation platform it receives the whole test set under `/input` and performs internal sequential inference (model loaded ONCE, per-case loop with explicit GPU-VRAM/memory hygiene).
- `T1_baseline_finaltest_batch` / `T2_baseline_finaltest_batch` — the official leaderboard baselines repackaged with the batch contract. Prediction code is unchanged from `T1_baseline` / `T2_baseline`; checkpoints come from the same [Huggingface](https://huggingface.co/Endoluminal-Surg-Vision-IMR/Airway-Tree-Modeling-26-baselines) locations.

Key batch-contract differences from the per-case baselines: the container reads all `*.mha` files under `/input/images/lung-ct` itself and writes one output per case to `/output/images/<output-slug>/<case-id>.mha`.

**The evaluation platform's GPU driver supports CUDA up to 12.0**; the three finaltest folders build on the slim `atm26-nnunetv2:2.6.4-cuda11.8` base whose Dockerfile is provided in `nnunetv2-base/` (cuda12.4-based images will not run on the platform).
