#!/usr/bin/env python
"""Grand Challenge inference entry point for the ATM26 Track 2 baseline."""

import glob
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as torch_functional
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path(__file__).resolve().parent / "resources" / "nnUNet_ckpts"
FOLDS = (0,)


def run() -> int:
    interface_key = get_interface_key()
    handler = {
        ("lung-ct",): interf0_handler,
    }[interface_key]
    return handler()


def interf0_handler() -> int:
    total_start = perf_counter()
    input_start = perf_counter()
    input_image, input_array = load_image_file_as_array(
        location=INPUT_PATH / "images" / "lung-ct"
    )
    if input_array.ndim != 3:
        raise ValueError(f"lung-ct must be 3D, got shape {input_array.shape}")
    input_seconds = perf_counter() - input_start

    model_start = perf_counter()
    predictor = build_predictor()
    model_seconds = perf_counter() - model_start
    nnunet_input = input_array[None].astype(np.float32, copy=False)
    del input_array
    image_properties = {
        "spacing": [abs(value) for value in input_image.GetSpacing()[::-1]],
    }
    prediction, prediction_timings = predict_array_low_memory(
        predictor=predictor,
        input_image=nnunet_input,
        image_properties=image_properties,
    )
    del nnunet_input, predictor
    prediction = np.asarray(prediction)
    expected_shape = tuple(reversed(input_image.GetSize()))
    if prediction.shape != expected_shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match input shape "
            f"{expected_shape}"
        )
    if not np.issubdtype(prediction.dtype, np.integer):
        if not np.all(np.isfinite(prediction)) or not np.all(
            prediction == np.rint(prediction)
        ):
            raise ValueError("nnU-Net prediction contains non-integral labels")
    if prediction.size and (prediction.min() < 0 or prediction.max() > 20):
        raise ValueError(
            f"nnU-Net prediction labels must be in [0, 20], got "
            f"[{prediction.min()}, {prediction.max()}]"
        )
    prediction = prediction.astype(np.uint8, copy=False)

    output_start = perf_counter()
    write_array_as_image_file(
        location=OUTPUT_PATH / "images" / "multi-class-airway-segmentation",
        array=prediction,
        reference=input_image,
    )
    output_seconds = perf_counter() - output_start
    total_seconds = perf_counter() - total_start
    print(
        "Execution time: "
        f"input={input_seconds:.2f}s, "
        f"model_initialization={model_seconds:.2f}s, "
        f"preprocessing={prediction_timings['preprocessing']:.2f}s, "
        f"inference={prediction_timings['inference']:.2f}s, "
        f"conversion={prediction_timings['conversion']:.2f}s, "
        f"output={output_seconds:.2f}s, "
        f"total={total_seconds:.2f}s"
    )
    return 0


def predict_array_low_memory(
    *,
    predictor: nnUNetPredictor,
    input_image: np.ndarray,
    image_properties: dict,
) -> tuple[np.ndarray, dict[str, float]]:
    preprocessing_start = perf_counter()
    preprocessing_adapter = PreprocessAdapterFromNpy(
        [input_image],
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
    del preprocessed, preprocessing_adapter
    preprocessing_seconds = perf_counter() - preprocessing_start

    inference_start = perf_counter()
    predicted_logits = predictor.predict_logits_from_preprocessed_data(data).cpu()
    del data
    torch.cuda.empty_cache()
    inference_seconds = perf_counter() - inference_start

    conversion_start = perf_counter()
    prediction = convert_logits_low_memory(
        predicted_logits=predicted_logits,
        predictor=predictor,
        properties=data_properties,
    )
    del predicted_logits
    conversion_seconds = perf_counter() - conversion_start

    return prediction, {
        "preprocessing": preprocessing_seconds,
        "inference": inference_seconds,
        "conversion": conversion_seconds,
    }


def convert_logits_low_memory(
    *,
    predicted_logits: torch.Tensor,
    predictor: nnUNetPredictor,
    properties: dict,
) -> np.ndarray:
    """Convert logits quickly by resampling the argmax label map once."""
    if predicted_logits.device.type != "cpu":
        raise ValueError("predicted logits must be on the CPU")
    if predicted_logits.ndim != 4:
        raise ValueError(
            f"predicted logits must have shape (C, Z, Y, X), got "
            f"{tuple(predicted_logits.shape)}"
        )
    if predictor.label_manager.has_regions:
        raise ValueError("low-memory conversion does not support region-based labels")

    plans_manager = predictor.plans_manager
    target_shape = tuple(properties["shape_after_cropping_and_before_resampling"])

    num_classes = predicted_logits.shape[0]
    if num_classes > np.iinfo(np.uint8).max:
        raise ValueError(f"too many output classes for uint8 labels: {num_classes}")

    print(
        f"Fast conversion: argmax over {num_classes} channels, then one "
        f"nearest-neighbor label resampling to {target_shape}"
    )
    segmentation = torch.argmax(predicted_logits, dim=0).to(torch.uint8)
    if tuple(segmentation.shape) != target_shape:
        segmentation = torch_functional.interpolate(
            segmentation[None, None],
            size=target_shape,
            mode="nearest-exact",
        )[0, 0]
    segmentation = segmentation.numpy()

    output_dtype = (
        np.uint8
        if len(predictor.label_manager.foreground_labels) < 255
        else np.uint16
    )
    segmentation_reverted_cropping = np.zeros(
        properties["shape_before_cropping"], dtype=output_dtype
    )
    segmentation_reverted_cropping = insert_crop_into_image(
        segmentation_reverted_cropping,
        segmentation,
        properties["bbox_used_for_cropping"],
    )
    del segmentation

    return segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)


def build_predictor() -> nnUNetPredictor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ATM26 Track 2 baseline inference")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        # Keep the full 21-channel accumulator on the CPU. Patch inference still
        # runs on CUDA and avoids nnU-Net's failed all-GPU attempt and full rerun.
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


def get_interface_key() -> tuple[str, ...]:
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [value["socket"]["slug"] for value in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location: Path):
    with location.open() as file:
        return json.load(file)


def load_image_file_as_array(*, location: Path) -> tuple[sitk.Image, np.ndarray]:
    input_files = sorted(glob.glob(str(location / "*.mha")))
    if len(input_files) != 1:
        raise ValueError(
            f"expected exactly one .mha file in {location}, found {len(input_files)}"
        )

    image = sitk.ReadImage(input_files[0])
    return image, sitk.GetArrayFromImage(image)


def write_array_as_image_file(
    *, location: Path, array: np.ndarray, reference: sitk.Image
) -> None:
    location.mkdir(parents=True, exist_ok=True)
    output = sitk.GetImageFromArray(array, isVector=False)
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(location / "output.mha"), useCompression=True)


if __name__ == "__main__":
    raise SystemExit(run())
