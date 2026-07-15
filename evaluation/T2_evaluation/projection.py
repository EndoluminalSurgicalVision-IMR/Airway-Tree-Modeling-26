# -*- coding: utf-8 -*-
"""
Project a multi-class segmentation IMAGE onto the fixed GT branch-nodes.

The GT `airway_parsing.nii.gz` assigns every airway voxel a branch ID in
[1, node_num] (0 = background); parsing value v maps to graph node index v-1.
For each node we take the majority (mode) class label over its voxels.

This reuses the established repo idioms:
  - airway_parsing()         (feature_new_itk.py:62)  -> the parsing map itself
  - loc_trachea_vectorized() (feature_new_itk.py:94)  -> bincount/argmax voting
"""
import numpy as np
from scipy import ndimage

# 26-connectivity: airway is a thin tubular tree, so use full connectivity (under
# 6-connectivity even a clean mask splits into many pieces).
_CONN26 = np.ones((3, 3, 3), dtype=np.uint8)


def keep_largest_component(seg, structure=_CONN26):
    """Largest-connected-component extraction for a multi-class segmentation IMAGE.

    Binarizes the foreground (any nonzero class), keeps the largest connected
    component, and zeros out all class labels outside it. Removes disconnected
    false-positive blobs before projection / voxel metrics.
    """
    fg = (seg > 0).astype(np.uint8)
    cc, n = ndimage.label(fg, structure=structure)
    if n <= 1:
        return seg
    counts = np.bincount(cc.ravel())
    counts[0] = 0  # ignore background
    largest = counts.argmax()
    return np.where(cc == largest, seg, 0)


def project_image_to_nodes(seg_arr, parsing_arr, node_num, num_classes=None):
    """Majority-vote a segmentation image onto branch-nodes.

    seg_arr     : int array, voxelwise class labels (>=0).
    parsing_arr : int array (same shape), branch IDs 0=bg, 1..node_num.
    node_num    : number of graph nodes.
    num_classes : class count for the vote histogram (auto if None).

    Returns: int array (node_num,), label per node. Nodes with no voxels -> 0.
    """
    if seg_arr.shape != parsing_arr.shape:
        raise ValueError(
            f"seg/parsing shape mismatch: {seg_arr.shape} vs {parsing_arr.shape}"
        )
    fg = parsing_arr > 0
    node_idx = parsing_arr[fg].astype(np.int64) - 1  # -> [0, node_num)
    cls = seg_arr[fg].astype(np.int64)
    if cls.size and cls.min() < 0:
        raise ValueError("segmentation contains negative labels")

    if num_classes is None:
        num_classes = int(cls.max()) + 1 if cls.size else 1
    num_classes = max(num_classes, 1)

    counts = np.zeros((node_num, num_classes), dtype=np.int64)
    np.add.at(counts, (node_idx, cls), 1)

    labels = counts.argmax(axis=1).astype(np.int64)  # majority vote per node
    labels[counts.sum(axis=1) == 0] = 0  # empty node -> background
    return labels
