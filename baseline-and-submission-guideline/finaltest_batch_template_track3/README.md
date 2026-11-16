# finaltest_batch_template_track3

Participant batch packing template. One or many sequences. Constant label 0 by default; replace build_predictor and predict_case with your method.

Fixed input: /input/sequences/<id>/*.png. Fixed output: /output/predictions/<id>.csv, columns filename,label. Retain every filename and numeric order. Valid/test share the same image; change host mounts only.

Includes resources/placeholder/model.json. Label 0 is LB1+2, not background. Placeholder predictions are for IO testing only.

```bash
bash do_build.sh
DEVICE=cuda bash do_test_run.sh /absolute/input_root /absolute/empty_output_root
bash do_save.sh /absolute/submission.tar.gz
```

The Dockerfile ends with `USER app`: the evaluation platform rejects an image
that runs as root. If you replace the base image or rewrite the Dockerfile, keep
a non-root user.

Image: `atm26-track3-template:v3`. No container arguments are required. Use DEVICE=cpu for a CPU-only smoke test.

See [root README](../../README.md) and [resources](../RESOURCES_Track3.md).
