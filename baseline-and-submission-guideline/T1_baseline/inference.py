#!/usr/bin/env python
"""Grand Challenge inference entry point for the ATM26 Track 1 baseline."""

import glob
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
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
    input_image, input_array = load_image_file_as_array(
        location=INPUT_PATH / "images" / "lung-ct"
    )
    if input_array.ndim != 3:
        raise ValueError(f"lung-ct must be 3D, got shape {input_array.shape}")

    predictor = build_predictor()
    nnunet_input = input_array[None].astype(np.float32, copy=False)
    image_properties = {
        "spacing": [abs(value) for value in input_image.GetSpacing()[::-1]],
    }
    prediction = predictor.predict_single_npy_array(
        input_image=nnunet_input,
        image_properties=image_properties,
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=False,
    )
    prediction = (np.asarray(prediction) > 0).astype(np.uint8, copy=False)
    if prediction.shape != input_array.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match input shape "
            f"{input_array.shape}"
        )

    write_array_as_image_file(
        location=OUTPUT_PATH / "images" / "binary-airway-segmentation",
        array=prediction,
        reference=input_image,
    )
    return 0


def build_predictor() -> nnUNetPredictor:
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
