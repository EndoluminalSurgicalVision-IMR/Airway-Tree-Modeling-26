## Baseline and Submission Guideline

![Challenge Banner](../assets/logo_v6_01.png)

We provide the official example algorithm and our baseline algorithm in this directory. Participants are expected to pack their own solutions, configure their docker image correctly and then build the docker image.

Baseline checkpoints can be found on [Huggingface](https://huggingface.co/Endoluminal-Surg-Vision-IMR/Airway-Tree-Modeling-26-baselines). You can download and replace the corresponding files and folders.

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