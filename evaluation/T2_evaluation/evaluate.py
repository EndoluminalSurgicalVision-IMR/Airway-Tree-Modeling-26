"""Grand Challenge evaluation entry point for ATM26 Track 2."""

import json
import logging
from pathlib import Path
from pprint import pformat
from statistics import mean

import numpy as np
import SimpleITK as sitk

from graph_utils import build_mask_top, build_spd
from helpers import run_prediction_processing, setup_logger, tree
from metrics import (
    BACKGROUND_LABEL,
    build_class_distance_matrix,
    compute_acc_f1,
    compute_sc,
    compute_tacc,
    compute_td,
)
from projection import keep_largest_component, project_image_to_nodes
from voxel_metrics import multi_class_cldice, multi_class_dice

logger = logging.getLogger("evaluate")

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")
OUTPUT_SLUG = "multi-class-airway-segmentation"
TD_FILL = 50.0
MAX_CLASS_LABEL = 20
LEADERBOARD_METRICS = ("ACC", "F1", "SC", "TD", "TAcc", "mDice", "mclDice")


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
        ("lung-ct",): process_lung_ct,
    }[interface_key]
    return handler(job)


def process_lung_ct(job):
    prediction_location = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug=OUTPUT_SLUG,
    )
    prediction_image, prediction_raw = load_single_mha(
        location=prediction_location,
        name="prediction",
        dtype=np.uint8,
        maximum=MAX_CLASS_LABEL,
    )

    image_name = get_image_name(values=job["inputs"], slug="lung-ct")
    case_id, ground_truth_paths = derive_ground_truth_paths(image_name=image_name)

    label_image, label = load_integer_image(
        path=ground_truth_paths["label"],
        name=f"{case_id} label",
        dtype=np.uint8,
        maximum=MAX_CLASS_LABEL,
    )
    parsing_image, parsing = load_integer_image(
        path=ground_truth_paths["parsing"],
        name=f"{case_id} parsing",
        dtype=np.int32,
    )
    skeleton_image, skeleton_values = load_integer_image(
        path=ground_truth_paths["skeleton"],
        name=f"{case_id} skeleton",
        dtype=np.uint8,
    )
    skeleton = (skeleton_values > 0).astype(np.uint8, copy=False)
    edges = load_graph(path=ground_truth_paths["graph"], case_id=case_id)

    validate_images(
        prediction_image=prediction_image,
        label_image=label_image,
        parsing_image=parsing_image,
        skeleton_image=skeleton_image,
        case_id=case_id,
    )
    node_num = validate_metadata(
        parsing=parsing,
        label=label,
        skeleton=skeleton,
        edges=edges,
        case_id=case_id,
    )

    prediction_lcc = keep_largest_component(prediction_raw)
    num_classes = max(
        int(label.max()), int(prediction_raw.max()), MAX_CLASS_LABEL
    ) + 1

    gt_label = project_image_to_nodes(label, parsing, node_num, num_classes)
    pred_label = project_image_to_nodes(
        prediction_lcc,
        parsing,
        node_num,
        num_classes,
    )
    valid = gt_label != BACKGROUND_LABEL

    spd = build_spd(edges)
    mask_top = build_mask_top(edges)
    accuracy, f1, _ = compute_acc_f1(gt_label, pred_label, valid)
    distance_matrix, distance_max = build_class_distance_matrix(MAX_CLASS_LABEL)
    mean_dice, dice_per_class = multi_class_dice(prediction_raw, label)
    mean_cldice, cldice_per_class = multi_class_cldice(
        prediction_raw,
        label,
        skeleton,
    )

    result = {
        "case": case_id,
        "ACC": accuracy,
        "F1": f1,
        "SC": compute_sc(pred_label, mask_top),
        "TD": compute_td(spd, gt_label, pred_label, valid, td_fill=TD_FILL),
        "TAcc": compute_tacc(
            gt_label,
            pred_label,
            valid,
            distance_matrix,
            distance_max,
        ),
        "mDice": mean_dice,
        "mclDice": mean_cldice,
        "dice_per_class": {
            str(class_label): score
            for class_label, score in dice_per_class.items()
        },
        "cldice_per_class": {
            str(class_label): score
            for class_label, score in cldice_per_class.items()
        },
        "num_nodes": int(node_num),
        "num_valid": int(valid.sum()),
    }

    logger.info("Processing job:\n%s\nMetrics:\n%s", pformat(job), pformat(result))
    return result


def derive_ground_truth_paths(*, image_name):
    """Derive flat ground-truth paths from a ``<case>_0000.mha`` image name."""
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
        "label": resolve_ground_truth_image(stem=case_id),
        "parsing": resolve_ground_truth_image(stem=f"{case_id}_parsing"),
        "skeleton": resolve_ground_truth_image(stem=f"{case_id}_skeleton"),
        "graph": GROUND_TRUTH_DIRECTORY / f"{case_id}_graph.npy",
    }


def resolve_ground_truth_image(*, stem):
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


def load_single_mha(*, location, name, dtype, maximum):
    input_files = sorted(
        path
        for path in location.iterdir()
        if path.is_file() and path.suffix.lower() == ".mha"
    )
    if len(input_files) != 1:
        raise ValueError(
            f"expected exactly one .mha prediction in {location}, "
            f"found {len(input_files)}"
        )
    return load_integer_image(
        path=input_files[0],
        name=name,
        dtype=dtype,
        maximum=maximum,
    )


def load_integer_image(*, path, name, dtype, maximum=None):
    if not path.is_file():
        raise FileNotFoundError(f"required image not found: {path}")

    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError(f"{name} must be 3D, got {image.GetDimension()}D")

    values = sitk.GetArrayFromImage(image)
    converted = np.empty(values.shape, dtype=dtype)
    dtype_limits = np.iinfo(dtype)
    upper_bound = dtype_limits.max if maximum is None else maximum

    # Validate in slabs to avoid allocating another full-volume floating array.
    for start in range(0, values.shape[0], 16):
        block = values[start: start + 16]
        if not np.isfinite(block).all():
            raise ValueError(f"{name} contains non-finite values")
        if np.issubdtype(block.dtype, np.floating):
            rounded = np.rint(block)
            if not np.array_equal(block, rounded):
                raise ValueError(f"{name} must contain integer labels")
            block = rounded
        if block.size and (block.min() < 0 or block.max() > upper_bound):
            raise ValueError(
                f"{name} labels must be in [0, {upper_bound}], "
                f"got [{block.min()}, {block.max()}]"
            )
        converted[start: start + 16] = block

    return image, converted


def load_graph(*, path, case_id):
    if not path.is_file():
        raise FileNotFoundError(f"required graph not found: {path}")
    edges = np.load(path, allow_pickle=False)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] == 0:
        raise ValueError(f"{case_id}: graph must have shape (2, E) with E > 0")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError(f"{case_id}: graph must contain integer node indices")
    edges = edges.astype(np.int64, copy=False)
    if np.any(edges < 0):
        raise ValueError(f"{case_id}: graph cannot contain negative node indices")
    return edges


def validate_images(
    *, prediction_image, label_image, parsing_image, skeleton_image, case_id
):
    images = {
        "prediction": prediction_image,
        "parsing": parsing_image,
        "skeleton": skeleton_image,
    }
    for name, image in images.items():
        if image.GetSize() != label_image.GetSize():
            raise ValueError(
                f"{case_id}: {name} size {image.GetSize()} does not match "
                f"label size {label_image.GetSize()}"
            )
        for attribute in ("Spacing", "Origin", "Direction"):
            actual = getattr(image, f"Get{attribute}")()
            expected = getattr(label_image, f"Get{attribute}")()
            if not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
                raise ValueError(
                    f"{case_id}: {name} {attribute.lower()} does not match label"
                )


def validate_metadata(*, parsing, label, skeleton, edges, case_id):
    node_num = int(edges.max()) + 1
    branch_ids = np.unique(parsing[parsing > 0]).astype(np.int64)
    expected_ids = np.arange(1, node_num + 1, dtype=np.int64)
    if not np.array_equal(branch_ids, expected_ids):
        raise ValueError(
            f"{case_id}: parsing IDs must be contiguous 1..{node_num} and "
            "match the graph"
        )
    if edges.shape[1] != node_num - 1:
        raise ValueError(
            f"{case_id}: graph has {edges.shape[1]} edges for {node_num} nodes; "
            "expected a tree with node_num - 1 edges"
        )
    if np.any((skeleton > 0) & (parsing <= 0)):
        raise ValueError(
            f"{case_id}: every skeleton voxel must have a positive branch ID"
        )
    if np.any((skeleton > 0) & (label <= 0)):
        raise ValueError(
            f"{case_id}: every skeleton voxel must lie inside the labeled airway"
        )
    return node_num


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


def write_metrics(*, metrics):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_json_file(location=OUTPUT_DIRECTORY / "metrics.json", content=metrics)


def write_json_file(*, location, content):
    with location.open("w") as file:
        json.dump(content, file, indent=4)


if __name__ == "__main__":
    raise SystemExit(main())
