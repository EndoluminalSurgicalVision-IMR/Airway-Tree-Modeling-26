"""Grand Challenge evaluation entry point for ATM26 Track 1."""

import json
import logging
from pathlib import Path
from pprint import pformat
from statistics import mean

import numpy as np
import SimpleITK as sitk

from helpers import run_prediction_processing, setup_logger, tree
from metrics import (
    betti0_error,
    branch_detected,
    cl_dice,
    dice_coefficient,
    keep_largest_component,
    tree_length_detected,
)

logger = logging.getLogger("evaluate")

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")
BD_THRESHOLD = 0.8
LEADERBOARD_METRICS = ("DSC", "clDice", "TLD", "BD", "Betti0Error")


def main():
    setup_logger(level=logging.INFO)
    log_inputs()

    results = run_prediction_processing(fn=process, predictions=read_predictions())
    results.sort(key=lambda result: result["case"])

    metrics = {"results": results}
    if results:
        metrics["aggregates"] = {
            metric: mean(result[metric] for result in results)
            for metric in LEADERBOARD_METRICS
        }

    write_metrics(metrics=metrics)
    return 0


def process(job):
    interface_key = get_interface_key(job)
    handler = {
        ("lung-ct",): process_interf0,
    }[interface_key]
    return handler(job)


def process_interf0(job):
    """Evaluate one binary-airway-segmentation output."""
    prediction_location = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="binary-airway-segmentation",
    )
    prediction_image, prediction = load_single_mha(location=prediction_location)

    image_name = get_image_name(values=job["inputs"], slug="lung-ct")
    case_id, ground_truth_paths = derive_ground_truth_paths(image_name=image_name)

    label_image, label = load_image(path=ground_truth_paths["label"])
    skeleton_image, skeleton = load_image(path=ground_truth_paths["skeleton"])
    parsing_image, parsing = load_image(path=ground_truth_paths["parsing"])

    validate_images(
        prediction_image=prediction_image,
        label_image=label_image,
        skeleton_image=skeleton_image,
        parsing_image=parsing_image,
        case_id=case_id,
    )
    validate_parsing(parsing=parsing, skeleton=skeleton, case_id=case_id)

    prediction_raw = (prediction > 0).astype(np.uint8)
    label = (label > 0).astype(np.uint8)
    skeleton = (skeleton > 0).astype(np.uint8)
    prediction_lcc = keep_largest_component(prediction_raw)

    total_branches, detected_branches, bd = branch_detected(
        prediction_lcc,
        parsing,
        skeleton,
        threshold=BD_THRESHOLD,
    )
    result = {
        "case": case_id,
        "DSC": dice_coefficient(prediction_raw, label),
        "clDice": cl_dice(prediction_raw, label, skeleton),
        "TLD": tree_length_detected(prediction_lcc, skeleton),
        "BD": bd,
        "Betti0Error": betti0_error(prediction_raw),
        "branch_detected": detected_branches,
        "branch_total": total_branches,
    }

    logger.info("Processing job:\n%s\nMetrics:\n%s", pformat(job), pformat(result))
    return result


def derive_ground_truth_paths(*, image_name):
    """Derive flat GT paths from a `<case>_0000.mha` input image name."""
    filename = Path(image_name)
    if filename.name != image_name or filename.suffix.lower() != ".mha":
        raise ValueError(
            f"lung-ct image name must be a flat .mha filename, got {image_name!r}"
        )

    stem = filename.stem
    if not stem.endswith("_0000") or len(stem) == len("_0000"):
        raise ValueError(
            "lung-ct image name must follow '<case>_0000.mha', "
            f"got {image_name!r}"
        )

    case_id = stem[: -len("_0000")]
    return case_id, {
        "label": resolve_ground_truth_path(stem=case_id),
        "skeleton": resolve_ground_truth_path(stem=f"{case_id}_skeleton"),
        "parsing": resolve_ground_truth_path(stem=f"{case_id}_parsing"),
    }


def resolve_ground_truth_path(*, stem):
    candidates = [
        GROUND_TRUTH_DIRECTORY / f"{stem}.mha",
        GROUND_TRUTH_DIRECTORY / f"{stem}.nii.gz",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one ground-truth image for {stem!r}; "
            f"checked: {', '.join(str(path) for path in candidates)}"
        )
    return matches[0]


def validate_images(
    *, prediction_image, label_image, skeleton_image, parsing_image, case_id
):
    images = {
        "prediction": prediction_image,
        "label": label_image,
        "skeleton": skeleton_image,
        "parsing": parsing_image,
    }
    for name, image in images.items():
        if image.GetDimension() != 3:
            raise ValueError(
                f"{case_id}: {name} must be 3D, got {image.GetDimension()}D"
            )

    reference = label_image
    for name, image in images.items():
        if image.GetSize() != reference.GetSize():
            raise ValueError(
                f"{case_id}: {name} size {image.GetSize()} does not match "
                f"label size {reference.GetSize()}"
            )
        for attribute in ("Spacing", "Origin", "Direction"):
            actual = getattr(image, f"Get{attribute}")()
            expected = getattr(reference, f"Get{attribute}")()
            if not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
                raise ValueError(
                    f"{case_id}: {name} {attribute.lower()} does not match label"
                )


def validate_parsing(*, parsing, skeleton, case_id):
    if not np.issubdtype(parsing.dtype, np.integer):
        if not np.all(np.isfinite(parsing)) or not np.all(parsing == np.rint(parsing)):
            raise ValueError(f"{case_id}: parsing volume must contain integer labels")
    if np.any(parsing < 0):
        raise ValueError(f"{case_id}: parsing volume cannot contain negative labels")
    if np.any((skeleton > 0) & (parsing <= 0)):
        raise ValueError(
            f"{case_id}: every skeleton voxel must have a positive branch ID"
        )

    branch_ids = np.unique(parsing[parsing > 0]).astype(np.int64)
    expected = np.arange(1, len(branch_ids) + 1, dtype=np.int64)
    if not np.array_equal(branch_ids, expected):
        raise ValueError(
            f"{case_id}: parsing branch IDs must be contiguous and start at 1"
        )


def log_inputs():
    logger.info("Input Files:")
    for line in tree(INPUT_DIRECTORY):
        logger.info(line)


def read_predictions():
    return load_json_file(location=INPUT_DIRECTORY / "predictions.json")


def get_interface_key(job):
    socket_slugs = [value["socket"]["slug"] for value in job["inputs"]]
    return tuple(sorted(socket_slugs))


def get_image_name(*, values, slug):
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["image"]["name"]
    raise RuntimeError(f"Image with interface {slug} not found")


def get_interface_relative_path(*, values, slug):
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["socket"]["relative_path"]
    raise RuntimeError(f"Value with interface {slug} not found")


def get_file_location(*, job_pk, values, slug):
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    with location.open() as file:
        return json.load(file)


def load_single_mha(*, location):
    input_files = sorted(
        path for path in location.iterdir() if path.is_file() and path.suffix.lower() == ".mha"
    )
    if len(input_files) != 1:
        raise ValueError(
            f"expected exactly one .mha prediction in {location}, found {len(input_files)}"
        )
    return load_image(path=input_files[0])


def load_image(*, path):
    if not path.is_file():
        raise FileNotFoundError(f"required image not found: {path}")
    image = sitk.ReadImage(str(path))
    return image, sitk.GetArrayFromImage(image)


def write_metrics(*, metrics):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_json_file(location=OUTPUT_DIRECTORY / "metrics.json", content=metrics)


def write_json_file(*, location, content):
    with location.open("w") as file:
        json.dump(content, file, indent=4)


if __name__ == "__main__":
    raise SystemExit(main())
