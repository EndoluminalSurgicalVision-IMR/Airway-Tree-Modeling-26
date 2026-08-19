#!/usr/bin/env python3
"""ATM26 Track-2 leaderboard baseline: internal sequential (batch) wrapper.

Dual-mode: one case at /input behaves like the original per-case algorithm;
the whole test set at /input is processed with the model loaded ONCE and a
per-case loop with GPU-VRAM/memory hygiene. The prediction core
(preprocessing -> GPU patch inference -> low-memory argmax conversion) is
unchanged from imr_atm26_track2_leaderboard_baseline/inference.py.
"""
import gc
import glob
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as torch_functional
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
OUTPUT_SLUG = "multi-class-airway-segmentation"
MODEL_PATH = Path(__file__).resolve().parent / "resources" / "nnUNet_ckpts"
FOLDS = (0,)


def build_predictor():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ATM26 Track 2 baseline inference")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        # Keep the full 21-channel accumulator on the CPU. Patch inference
        # still runs on CUDA.
        perform_everything_on_device=False,
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
    nnunet_input = array[None].astype(np.float32, copy=False)
    preprocessing_adapter = PreprocessAdapterFromNpy(
        [nnunet_input],
        [None],
        [image_properties],
        [None],
        predictor.plans_manager,
        predictor.dataset_json,
        predictor.configuration_manager,
        num_threads_in_multithreaded=1,
        verbose=predictor.verbose,
    )
    preprocessed = next(preprocessing_adapter)
    data = preprocessed.pop("data")
    data_properties = preprocessed["data_properties"]
    del preprocessed, preprocessing_adapter, nnunet_input

    predicted_logits = predictor.predict_logits_from_preprocessed_data(data).cpu()
    del data
    torch.cuda.empty_cache()

    if predictor.label_manager.has_regions:
        raise ValueError("region-based labels are not supported")
    plans_manager = predictor.plans_manager
    target_shape = tuple(
        data_properties["shape_after_cropping_and_before_resampling"]
    )
    segmentation = torch.argmax(predicted_logits, dim=0).to(torch.uint8)
    if tuple(segmentation.shape) != target_shape:
        segmentation = torch_functional.interpolate(
            segmentation[None, None],
            size=target_shape,
            mode="nearest-exact",
        )[0, 0]
    segmentation = segmentation.numpy()
    output_dtype = np.uint8
    segmentation_reverted_cropping = np.zeros(
        data_properties["shape_before_cropping"], dtype=output_dtype
    )
    segmentation_reverted_cropping = insert_crop_into_image(
        segmentation_reverted_cropping,
        segmentation,
        data_properties["bbox_used_for_cropping"],
    )
    del segmentation, predicted_logits
    prediction = segmentation_reverted_cropping.transpose(
        plans_manager.transpose_backward
    )
    expected_shape = tuple(reversed(image.GetSize()))
    if prediction.shape != expected_shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match input shape "
            f"{expected_shape}"
        )
    if prediction.size and (prediction.min() < 0 or prediction.max() > 20):
        raise ValueError(
            f"nnU-Net prediction labels must be in [0, 20], got "
            f"[{prediction.min()}, {prediction.max()}]"
        )
    return prediction.astype(np.uint8, copy=False)


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
