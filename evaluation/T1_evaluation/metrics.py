"""ATM26 Track 1 airway-segmentation metrics."""

import numpy as np
from scipy import ndimage

try:
    from skimage.morphology import skeletonize as _skeletonize

    def skeletonize_3d(mask):
        return _skeletonize(mask > 0)

except ImportError:  # pragma: no cover - compatibility with older scikit-image
    from skimage.morphology import skeletonize_3d


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def keep_largest_component(mask, structure=CONNECTIVITY_26):
    """Return the largest 26-connected foreground component."""
    mask = (mask > 0).astype(np.uint8)
    components, component_count = ndimage.label(mask, structure=structure)
    if component_count <= 1:
        return mask

    counts = np.bincount(components.ravel())
    counts[0] = 0
    return (components == counts.argmax()).astype(np.uint8)


def num_components(mask, structure=CONNECTIVITY_26):
    """Return the foreground Betti-0 value under 26-connectivity."""
    _, component_count = ndimage.label(mask > 0, structure=structure)
    return int(component_count)


def dice_coefficient(prediction, label, smooth=1e-5):
    """Calculate volumetric Dice similarity."""
    prediction = prediction > 0
    label = label > 0
    intersection = np.count_nonzero(prediction & label)
    return float(
        (2.0 * intersection + smooth)
        / (np.count_nonzero(prediction) + np.count_nonzero(label) + smooth)
    )


def tree_length_detected(prediction, gt_skeleton, smooth=1e-5):
    """Return the fraction of the GT centerline covered by the prediction."""
    prediction = prediction > 0
    skeleton = gt_skeleton > 0
    return float(
        (np.count_nonzero(prediction & skeleton) + smooth)
        / (np.count_nonzero(skeleton) + smooth)
    )


def branch_detected(prediction, gt_parsing, gt_skeleton, threshold=0.8):
    """Return total branches, detected branches, and their ratio."""
    skeleton_mask = gt_skeleton > 0
    branch_ids = gt_parsing[skeleton_mask].astype(np.int64, copy=False)
    gt_counts = np.bincount(branch_ids)[1:]
    total = int(gt_counts.shape[0])
    covered_branch_ids = branch_ids[(prediction > 0)[skeleton_mask]]
    prediction_counts = np.bincount(covered_branch_ids)[1:]
    if prediction_counts.shape[0] < total:
        prediction_counts = np.pad(
            prediction_counts,
            (0, total - prediction_counts.shape[0]),
        )

    coverage = prediction_counts / np.maximum(gt_counts, 1)
    detected = int(np.count_nonzero(coverage >= threshold))
    ratio = detected / total if total else 0.0
    return total, detected, float(ratio)


def _centerline_overlap(volume, skeleton):
    volume = volume > 0
    skeleton = skeleton > 0
    skeleton_sum = np.count_nonzero(skeleton)
    if skeleton_sum == 0:
        return 0.0
    return float(np.count_nonzero(volume & skeleton) / skeleton_sum)


def cl_dice(prediction, label, gt_skeleton):
    """Calculate clDice using the supplied GT skeleton."""
    prediction = (prediction > 0).astype(np.uint8)
    label = (label > 0).astype(np.uint8)
    label_skeleton = (gt_skeleton > 0).astype(np.uint8)
    prediction_skeleton = (skeletonize_3d(prediction) > 0).astype(np.uint8)

    topology_sensitivity = _centerline_overlap(prediction, label_skeleton)
    topology_precision = _centerline_overlap(label, prediction_skeleton)
    denominator = topology_precision + topology_sensitivity
    if denominator == 0:
        return 0.0
    return float(2.0 * topology_precision * topology_sensitivity / denominator)


def betti0_error(prediction):
    """Calculate Betti-0 error against the fixed GT Betti-0 value of one."""
    return abs(num_components(prediction) - 1)
