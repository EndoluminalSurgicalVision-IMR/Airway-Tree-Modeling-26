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
### Track-3: Landmark Recognition for Endobronchial Intervention
Track 3 moves from CT volumes to the endobronchial view. A case is a rendered
bronchoscopy sequence along the airway tree, and every frame carries the
anatomical landmark visible at that moment. Participants submit **one
self-contained inference container** that reads PNG sequences and writes a
per-frame label CSV; the organizers mount the data, run the container and score
the predictions on the host.

An example algorithm for Track-3 can be found under `T3_example_algorithm`. It
is standard-library only, writes the constant label 0, and defines the I/O
interface for reading PNG sequences and writing frame-label CSVs.

The corresponding baseline algorithm for Track-3 can be found under
`T3_baseline`: a CLIP+MLP model that processes exactly one sequence per run. The
same model repackaged with the batch contract is in
`T3_baseline_finaltest_batch`: it loads the checkpoint once and predicts the
whole split sequentially, releasing the per-sequence tensors between runs.

The participant batch template is `finaltest_batch_template_track3`. It is a
skeleton with a small JSON placeholder: replace `build_predictor` and
`predict_case` with your own method, keep the I/O paths and the label schema. It
is not another official baseline.

The checkpoints are not committed to Git. Stage them in either baseline
directory with `download_weights.py` (`--source` for a local copy, `--url` for
the published file); the script verifies the fixed SHA-256. See
[RESOURCES_Track3.md](RESOURCES_Track3.md) for the checksum, the image
distribution and what is deliberately excluded from the repository.

Please carefully configure the `Dockerfile` before docker image saving and
testing. Track-3 images must declare `LABEL org.atm26.batch="1"` and
`LABEL org.grand-challenge.api-method="exec"`, and must run as a non-root user
(`USER app`); the platform rejects an image that runs as root. The example and
the template already end with `USER app`.

To build and test the docker, you can run:
```bash
cd baseline-and-submission-guideline/T3_baseline
bash do_build.sh
DEVICE=cuda bash do_test_run.sh /absolute/input_root /absolute/empty_output_root
```

Use `DEVICE=cpu` for a CPU-only smoke test, and use `T3_baseline_finaltest_batch`
instead of `T3_baseline` to test the batch packaging. No container arguments are
required.

To build and save the docker, you can run:
```bash
cd baseline-and-submission-guideline/T3_baseline
bash do_save.sh /absolute/submission.tar.gz
```

The four images are tagged `atm26-track3-example:v3`, `atm26-track3-baseline:v3`,
`atm26-track3-baseline-batch:v3` and `atm26-track3-template:v3`.

**Labels.** Predictions use the fixed 32-class schema (13 proximal and 19
segmental classes) shipped as `label_schema.json` in every algorithm directory:
ID `0` is a real class (LB1+2), not background, and ID `18` is unused. The
illustrated label guide is published on the challenge website:
<https://journey1234-liu.github.io/atm26/#/tracks>.

**Container contract** (Track 3 has its own I/O paths — not the CT ones):

- Input: `/input/sequences/<sequence-id>/000000.png … 000200.png` (201 frames,
  read-only). The container receives all sequences of the split in one run; load
  the model once and loop over the sequences.
- Output: `/output/predictions/<sequence-id>.csv`, header `filename,label`, one
  row per input frame in numeric order, values from the 32-class schema, exactly
  one CSV per sequence.
- No network access, no command-line arguments.

The same image serves the validation and test splits; only the mounted paths
differ. The official scorer is in
[`evaluation/T3_evaluation/`](../evaluation/T3_evaluation/) and the metrics are
documented in [../evaluation/README.md](../evaluation/README.md).

### Batch execution (Validation & Final Test phases)

The batch contract below applies to **both the Validation Phase and the Final Test
Phase** — the submission images are identical; only the phase/leaderboard that receives
the upload differs. Three folders are provided:

- `finaltest_batch_template` — the generic Docker packing template for the ATM26 local
  evaluation platform's batch contract. On the real Grand-Challenge platform the
  container receives ONE case and behaves like a standard per-case algorithm; on our
  ATM26 local inference + evaluation platform (Validation or Final Test phase) it
  receives the whole test set under `/input` and performs internal sequential inference
  (model loaded ONCE, per-case loop with explicit GPU-VRAM/memory hygiene).
- `T1_baseline_finaltest_batch` / `T2_baseline_finaltest_batch` — the official
  leaderboard baselines repackaged with the batch contract. Prediction code is unchanged
  from `T1_baseline` / `T2_baseline`; checkpoints come from the same
  [Huggingface](https://huggingface.co/Endoluminal-Surg-Vision-IMR/Airway-Tree-Modeling-26-baselines)
  locations.
- `nnunetv2-base/` — the slim CUDA 11.8 base-image recipe these folders build on.

Key batch-contract rules (the platform enforces these):

- The Dockerfile **must** declare `LABEL org.atm26.batch="1"` (selects batch execution;
  without it the platform runs your image per case) and
  `LABEL org.grand-challenge.api-method="exec"`.
- The container reads all `*.mha` files under `/input/images/lung-ct` itself and writes
  **one output per case** to `/output/images/<output-slug>/<case-id>.mha`. In batch mode it
  must NOT write `output.mha`; the per-case stem is `<case>`, `<case>_0000`, or the
  platform-assigned job id.
- The submitter-facing output format is **per-case `.mha`** (NOT `.nii.gz`): uint8 pixel
  data embedded in the file (`ElementDataFile=LOCAL`), compressed. Exactly one file per
  case, no extras, no duplicates.
- Rule of thumb: with exactly ONE case at `/input`, the same image writes
  `/output/images/<output-slug>/output.mha` and behaves like the standard per-case
  baseline, so one image serves both the Grand Challenge platform and our platform's
  batch runs.

**The evaluation platform's GPU driver supports CUDA up to 12.0**; the batch folders build
on the slim `atm26-nnunetv2:2.6.4-cuda11.8` base whose Dockerfile is provided in
`nnunetv2-base/` (cuda12.4-based images will not run on the platform).

For the **Validation Phase** you use the exact same folders and commands: build your image
with the template/baseline scripts and upload it to the phase you are submitting to
(Validation Phase on the challenge site; Final Test Phase via the Google Form / off-GC
submission) — our local evaluation platform runs both with the same batch contract.
