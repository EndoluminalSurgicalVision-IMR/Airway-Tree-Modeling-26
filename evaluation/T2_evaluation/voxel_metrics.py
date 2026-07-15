# -*- coding: utf-8 -*-
"""
Track-2 voxel-level multi-class metrics (complement the branch-node metrics).

These operate DIRECTLY on the voxelwise segmental-class images (airway_seg_cls),
not on the majority-vote node labels, so they catch what the branch-node metrics
(ACC/F1/SC/TD/TAcc) structurally cannot:
  - intra-branch voxel mixing -- a branch whose majority vote is correct hides any
    minority of mislabeled voxels from the node metrics; per-class Dice/clDice see it;
  - voxels labeled as the wrong class (incl. class voxels outside the GT airway,
    when scored over the full image).

multi_class_dice  : mean per-class volumetric Dice.
multi_class_cldice: mean per-class clDice (Shit et al., CVPR 2021), reusing the
    provided GT skeleton for topology precision and skeletonizing the prediction
    per class for topology sensitivity.

Per-class Dice mirrors track_1.metrics.dice_coefficient; clDice mirrors
track_1.metrics.cl_dice. Labels follow the official segmental scheme
(0 = background, 1..19 segmental classes, 19 = trachea). The reported mean is over
the classes present in the GT (background excluded), matching the macro-F1 pattern
in track_2.metrics.compute_acc_f1.
"""
import numpy as np
# `skeletonize` handles 3D via the Lee94 algorithm (replaces the deprecated
# `skeletonize_3d`); fall back to `skeletonize_3d` on older skimage. Same idiom as
# track_1/metrics.py.
try:
    from skimage.morphology import skeletonize as _skeletonize

    def skeletonize_3d(x):
        return _skeletonize(x > 0)
except ImportError:  # pragma: no cover
    from skimage.morphology import skeletonize_3d

BACKGROUND_LABEL = 0


def present_classes(gt_seg):
    """Sorted GT segmental classes (background excluded)."""
    return [int(c) for c in np.unique(gt_seg) if int(c) != BACKGROUND_LABEL]


def _dice(pred_c, gt_c, smooth=1e-5):
    """Volumetric Dice between two boolean masks."""
    inter = float(np.count_nonzero(pred_c & gt_c))
    return (2.0 * inter + smooth) / (float(pred_c.sum()) + float(gt_c.sum()) + smooth)


def _cl_score(v, s):
    """Topology overlap: fraction of skeleton voxels `s` lying inside volume `v`.

    Both are boolean masks. Port of track_1.metrics._cl_score.
    """
    s_sum = float(np.count_nonzero(s))
    return float(np.count_nonzero(v & s)) / s_sum if s_sum > 0 else 0.0


def multi_class_dice(pred_seg, gt_seg, smooth=1e-5):
    """Mean per-class volumetric Dice over GT-present segmental classes.

    Returns (mean_dice, {class: dice}). A GT class the prediction misses entirely
    scores ~0; mislabeled voxels of a class lower its Dice via FP/FN.
    """
    per_class = {}
    for c in present_classes(gt_seg):
        gt_c = gt_seg == c
        pred_c = pred_seg == c
        per_class[c] = float(_dice(pred_c, gt_c, smooth))
    mean = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean, per_class


def multi_class_cldice(pred_seg, gt_seg, gt_skeleton):
    """Mean per-class clDice over GT-present segmental classes.

    For class c (boolean masks gt_c, pred_c):
      Tprec = |pred_c ∩ skel_gt_c| / |skel_gt_c|, where skel_gt_c is the PROVIDED GT
              skeleton restricted to class c (reused, never derived -- same contract
              as Track-1's Tprec).
      Tsens = |gt_c ∩ skel(pred_c)| / |skel(pred_c)|, skeletonizing the prediction.
      clDice = harmonic mean of Tprec and Tsens.

    A GT class with no provided skeleton voxel is excluded from the mean (it has no
    measurable centerline). Returns (mean_cldice, {class: cldice}).
    """
    skel_fg = gt_skeleton > 0
    per_class = {}
    for c in present_classes(gt_seg):
        gt_c = gt_seg == c
        skel_gt_c = skel_fg & gt_c
        if not skel_gt_c.any():
            continue  # no centerline for this class -> not topology-measurable
        pred_c = pred_seg == c
        if not pred_c.any():
            per_class[c] = 0.0
            continue
        skel_pred_c = skeletonize_3d(pred_c) > 0
        tprec = _cl_score(pred_c, skel_gt_c)
        tsens = _cl_score(gt_c, skel_pred_c)
        per_class[c] = 0.0 if (tprec + tsens) == 0 else \
            float(2.0 * tprec * tsens / (tprec + tsens))
    mean = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean, per_class
