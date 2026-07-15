#!/usr/bin/env python
"""Validate one local Grand Challenge algorithm test result."""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def same_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing())
        and np.allclose(first.GetOrigin(), second.GetOrigin())
        and np.allclose(first.GetDirection(), second.GetDirection())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_files = sorted((args.input / "images" / "lung-ct").glob("*.mha"))
    output_files = sorted(
        (args.output / "images" / "binary-airway-segmentation").glob("*.mha")
    )
    if len(input_files) != 1:
        raise ValueError(f"expected one input .mha, found {len(input_files)}")
    if len(output_files) != 1:
        raise ValueError(f"expected one output .mha, found {len(output_files)}")

    input_image = sitk.ReadImage(str(input_files[0]))
    output_image = sitk.ReadImage(str(output_files[0]))
    if input_image.GetDimension() != 3 or output_image.GetDimension() != 3:
        raise ValueError("input and output must both be 3D")
    if not same_geometry(input_image, output_image):
        raise ValueError("output geometry does not match the cropped input")
    if output_image.GetPixelID() != sitk.sitkUInt8:
        raise ValueError(
            f"output must be uint8, got {output_image.GetPixelIDTypeAsString()}"
        )

    output_array = sitk.GetArrayFromImage(output_image)
    values = np.unique(output_array)
    if not set(values.tolist()).issubset({0, 1}):
        raise ValueError(f"output is not binary: values={values.tolist()}")
    if not np.any(output_array):
        raise ValueError("output segmentation is empty")

    print(f"Validated {output_files[0]}")
    print(f"  size={output_image.GetSize()}")
    print(f"  foreground_voxels={int(np.count_nonzero(output_array))}")


if __name__ == "__main__":
    main()
