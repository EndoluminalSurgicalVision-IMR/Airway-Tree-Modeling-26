#!/usr/bin/env python3
"""ATM26 Track-1 leaderboard baseline: internal sequential (batch) wrapper.

Dual-mode: one case at /input behaves like the original per-case algorithm;
the whole test set at /input is processed with the model loaded ONCE and a
per-case loop with GPU-VRAM/memory hygiene. Prediction code is unchanged from
imr_atm26_track1_leaderboard_baseline/inference.py.
"""
import gc
import glob
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
OUTPUT_SLUG = "binary-airway-segmentation"
MODEL_PATH = Path(__file__).resolve().parent / "resources" / "nnUNet_ckpts"
FOLDS = (0,)


def build_predictor():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ATM26 Track 1 baseline inference")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_PATH),
        use_folds=FOLDS,
        checkpoint_name="checkpoint_best.pth",
    )
    return predictor


def predict_case(predictor, image, array):
    image_properties = {
        "spacing": [abs(value) for value in image.GetSpacing()[::-1]],
    }
    prediction = predictor.predict_single_npy_array(
        input_image=array[None].astype(np.float32, copy=False),
        image_properties=image_properties,
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=False,
    )
    prediction = (np.asarray(prediction) > 0).astype(np.uint8, copy=False)
    return prediction


def write_output(image_reference, prediction, name):
    output_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    output_dir.mkdir(parents=True, exist_ok=True)
    output = sitk.GetImageFromArray(prediction)
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
        path = Path(inputs[0])
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        predictor = build_predictor()
        with torch.inference_mode():
            prediction = predict_case(predictor, image, array)
        write_output(image, prediction, "output.mha")
        return 0

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
