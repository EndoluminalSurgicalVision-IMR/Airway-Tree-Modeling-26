## Track-3 weights and images

Source code lives in this repository; model weights and runnable images are
distributed separately, the same way as for Track 1 and Track 2.

### Weights

Both Track-3 baselines (`T3_baseline/`, `T3_baseline_finaltest_batch/`) use the
same CLIP+MLP checkpoint:

```
resources/clip_mlp/checkpoint.pt
SHA-256 2cc20ae3ca7a0aa86bba708fc9e0d7da68fc78aff2f5d780ffc6790d514d04fe
```

The checkpoint is **not** committed. Stage it with the download script in
either baseline directory, which verifies the SHA-256 and refuses to overwrite
a different file:

```bash
cd baseline-and-submission-guideline/T3_baseline
python download_weights.py --url 'https://<published-host>/path/checkpoint.pt'
```

`--source /path/to/checkpoint.pt` works for an offline copy. The example
algorithm and the batch template ship a small JSON placeholder and need no
network weights at all.

Do not substitute the older 27-class checkpoint: the label schema of this
track has 32 classes.

### Images

The baselines can be built locally with the scripts in their directories
(`bash do_build.sh`, `bash do_test_run.sh`, `bash do_save.sh`). A submission is
a single self-contained `docker save` archive, for example:

```
docker save atm26-track3-baseline-batch:v3 | gzip > atm26-track3-baseline-batch-v3.tar.gz
```

The build needs network access for `torch==2.6.0` (CUDA 11.8 wheel index) and
the packages in `requirements.txt`; the resulting image runs fully offline.

An archive containing all four Track-3 images is only for local distribution
and testing — it is **not** a valid single submission. Submit one image, built
from one of the directories above.

### What is not in Git

- `**/resources/**/*.pt` — checkpoints (see above),
- `**/Dockerfile.local` — machine-specific build files,
- `distribution/`, `verification/local/`, `*.log` — local artifacts.

The participant-facing submission is the Docker image; the host mounts the
input and output directories and runs the official scorer, so no checkpoint or
dataset needs to be shipped with it.
