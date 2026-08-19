# ATM26 Final Test Phase — Batch Docker Packing Template

Dual-mode submission template for the ATM26 Final Test Phase. The same image
works in both environments:

| Environment | What the container sees | Behaviour |
| --- | --- | --- |
| Grand Challenge (official) | ONE case at `/input` | standard per-case algorithm (writes one output) |
| ATM26 local evaluation platform | the WHOLE test set at `/input` | **internal sequential inference**: model loaded once, every case predicted in a loop, one `<case>.mha` per case (selected via `org.atm26.batch=1`) |

## How to adapt

1. Set `OUTPUT_SLUG` (and `MAX_CLASS_LABEL`) in `inference.py`:
   - Track-1: `binary-airway-segmentation` (binary mask)
   - Track-2: `multi-class-airway-segmentation` (labels 0–20)
2. Replace `build_predictor()` and `predict_case()` with your model. The
   shipped versions return an all-zero prediction so the template runs out of
   the box.
3. Add your Python dependencies to `requirements.txt` and your model files to
   the build context (copy them in the Dockerfile if needed).

## GPU-VRAM / memory hygiene (already wired in)

- `torch.inference_mode()` around all prediction code.
- The predictor is built **once** — per-case model loading is the dominant
  cost of the classic per-case pattern and is avoided here.
- Guidance in `predict_case`: run only patch inference on the GPU and keep the
  label accumulation on the CPU (see the low-memory nnU-Net pattern).
- Between cases: `del` of large references, `torch.cuda.empty_cache()`,
  `gc.collect()` — peak VRAM/RAM stays flat over hundreds of cases.
- Outputs are uint8 + compressed MHA; per-case timings are logged.

## Build, test, save

```bash
bash do_build.sh                      # build atm26-batch-template:latest
bash do_test_run.sh                   # runs against ./test/input -> ./test/output
bash do_save.sh                       # writes atm26-batch-template.tar.gz
```

The saved `.tar.gz` is what you upload through the ATM26 Test Phase form.
