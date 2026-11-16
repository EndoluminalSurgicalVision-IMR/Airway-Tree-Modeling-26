# T3_example_algorithm

Single-sequence IO example. Constant legal label 0 only. Replace build_predictor and predict_case; this is not a trained model.

Fixed input: /input/sequences/<id>/*.png. Fixed output: /output/predictions/<id>.csv, columns filename,label. Retain every filename and numeric order. Valid/test share the same image; change host mounts only.

Includes resources/placeholder/model.json. Label 0 is LB1+2, not background. Placeholder predictions are for IO testing only.

```bash
bash do_build.sh
DEVICE=cuda bash do_test_run.sh /absolute/input_root /absolute/empty_output_root
bash do_save.sh /absolute/submission.tar.gz
```

Image: `atm26-track3-example:v3`. No container arguments are required. Use DEVICE=cpu for a CPU-only smoke test.

See [root README](../../README.md) and [resources](../RESOURCES_Track3.md).
