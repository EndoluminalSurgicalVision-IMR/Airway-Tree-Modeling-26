"""ATM26 Final Test Phase batch inference template (dual-mode).

Behaviour
---------
- One case mounted at /input (real Grand-Challenge): behaves like the standard
  per-case example algorithm and writes a single output image.
- Whole test set mounted at /input (ATM26 local platform, ``org.atm26.batch=1``):
  performs internal sequential inference - the model is loaded ONCE, every
  ``/input/images/lung-ct/*.mha`` case is predicted in a loop, and one
  ``<case>.mha`` is written per case to ``/output/images/<output-slug>/``.

GPU-VRAM / memory hygiene
-------------------------
- ``torch.inference_mode()`` for the whole loop (no autograd graph).
- The predictor (and its weights) is built a single time.
- Keep large accumulators on the CPU and run only patch inference on the GPU
  (see ``predict_case`` comments).
- ``torch.cuda.empty_cache()`` + ``gc.collect()`` + explicit ``del`` between
  cases so peak VRAM/RAM stays flat across hundreds of cases.
- Outputs are uint8 and MHA-compressed.

To adapt this template: set OUTPUT_SLUG (and MAX_CLASS_LABEL) for your track,
then replace ``build_predictor`` / ``predict_case`` with your model. The
shipped implementations return an all-zero prediction so the template is
testable end-to-end before you plug in your network.
"""
import gc
import glob
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

# Track-1: "binary-airway-segmentation"  (binary mask, values {0, 1})
# Track-2: "multi-class-airway-segmentation" (labels 0..20)
OUTPUT_SLUG = "multi-class-airway-segmentation"
MAX_CLASS_LABEL = 20


def build_predictor():
    """Build and initialize your model ONCE. REPLACE THIS.

    Return any object you need inside ``predict_case`` (e.g. an
    ``nnUNetPredictor`` whose checkpoint is already loaded, a torch module in
    ``eval()`` mode, or a tuple of them).
    """
    # Placeholder: no-op predictor so the template runs out of the box.
    return None


def predict_case(predictor, input_image, input_array):
    """Predict ONE case. REPLACE THIS.

    ``input_array`` is a numpy array shaped (Z, Y, X) of the input CT;
    ``input_image`` carries the metadata for the output header.

    Memory guidance for large volumes:
    - run only the patch/tile forward passes on the GPU,
    - accumulate the label map in a small dtype on the CPU,
    - do NOT keep per-voxel probability maps for the whole volume in VRAM.

    Return a numpy label map of the same (Z, Y, X) shape with values in
    [0, MAX_CLASS_LABEL].
    """
    # Placeholder: an all-zero prediction in the input geometry.
    return np.zeros(input_array.shape, dtype=np.uint8)


def write_output(image_reference, prediction, name):
    output_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    output_dir.mkdir(parents=True, exist_ok=True)
    output = sitk.GetImageFromArray(prediction.astype(np.uint8, copy=False))
    output.CopyInformation(image_reference)
    sitk.WriteImage(output, str(output_dir / name), useCompression=True)


def case_id_of(path):
    stem = path.stem
    return stem[: -len("_0000")] if stem.endswith("_0000") else stem


def main():
    inputs = sorted(glob.glob(str(INPUT_PATH / "images" / "lung-ct" / "*.mha")))
    if not inputs:
        print("no .mha inputs found under /input/images/lung-ct", flush=True)
        return 1

    if len(inputs) == 1:
        # Grand-Challenge single-case behaviour.
        path = Path(inputs[0])
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        predictor = build_predictor()
        with torch.inference_mode():
            prediction = predict_case(predictor, image, array)
        write_output(image, prediction, "output.mha")
        print("single-case inference complete", flush=True)
        return 0

    # Batch mode: model loaded once, one pass per case.
    print(f"batch mode: {len(inputs)} cases", flush=True)
    predictor = build_predictor()
    started = time.perf_counter()
    with torch.inference_mode():
        for index, source in enumerate(inputs, start=1):
            case_started = time.perf_counter()
            path = Path(source)
            image = sitk.ReadImage(str(path))
            array = sitk.GetArrayFromImage(image)
            prediction = predict_case(predictor, image, array)
            del array
            case_id = case_id_of(path)
            write_output(image, prediction, f"{case_id}.mha")
            del image, prediction
            torch.cuda.empty_cache()
            gc.collect()
            print(
                f"[{index}/{len(inputs)}] {case_id}: "
                f"{time.perf_counter() - case_started:.1f}s",
                flush=True,
            )
    print(
        f"batch inference complete: {len(inputs)} cases in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
