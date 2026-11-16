# T3_baseline_finaltest_batch

Sequential batch trained baseline. One or many sequences. Loads the same model once, predicts sequentially and releases per-sequence tensors/cache.

Fixed input: /input/sequences/<id>/*.png. Fixed output: /output/predictions/<id>.csv, columns filename,label. Retain every filename and numeric order. Valid/test share the same image; change host mounts only.

Stage resources/clip_mlp/checkpoint.pt using download_weights.py --source /path/to/checkpoint.pt or a real published --url. See [RESOURCES_Track3.md](../RESOURCES_Track3.md).

```bash
bash do_build.sh
DEVICE=cuda bash do_test_run.sh /absolute/input_root /absolute/empty_output_root
bash do_save.sh /absolute/submission.tar.gz
```

Image: `atm26-track3-baseline-batch:v3`. No container arguments are required. Use DEVICE=cpu for a CPU-only smoke test.

See [root README](../../README.md) and [resources](../RESOURCES_Track3.md).
