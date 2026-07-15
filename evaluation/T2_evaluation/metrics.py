# -*- coding: utf-8 -*-
"""
Track-2 branch-wise anatomical labeling metrics.

All metrics operate on per-node labels (official segmental scheme) projected
onto the fixed GT graph, plus the GT-derived `spd` and `mask_top`.

Metric definitions follow the baseline repo:
  airway_labeling_baseline_LCY_20260302/airwaynet/test_2stage.py
    - ACC / macro-F1  (L246-300)
    - SC  `top_seg`   (L260-311)
    - TD  compute_td  (L83-105)
TAcc (TreeAcc) augments accuracy with a penalty by the class-to-class distance in
an anatomical hierarchy of airway classes: score_i = 1 - D[pred_i, gt_i]/D_max.
"""
import numpy as np

TRACHEA_LABEL = 19  # official Track-2 segmental scheme: 19 = trachea
BACKGROUND_LABEL = 0


def compute_acc_f1(gt, pred, valid_mask):
    """ACC and macro-F1 over valid branch-nodes.

    Macro-F1 averages per-class F1 over classes present in the GT (background
    excluded), mirroring the `cal`-mask pattern in test_2stage.py.
    """
    g = gt[valid_mask]
    p = pred[valid_mask]
    n = g.shape[0]
    acc = float(np.mean(p == g)) if n else 0.0

    classes = [c for c in np.unique(g) if c != BACKGROUND_LABEL]
    f1s = {}
    for c in classes:
        tp = np.sum((g == c) & (p == c))
        fp = np.sum((g != c) & (p == c))
        fn = np.sum((g == c) & (p != c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s[int(c)] = float(f1)
    macro_f1 = float(np.mean(list(f1s.values()))) if f1s else 0.0
    return acc, macro_f1, f1s


def compute_sc(pred, mask_top, trachea_label=TRACHEA_LABEL):
    """Subtree Consistency: fraction of (non-trivial) subtrees whose nodes all
    share the same predicted segmental label. Port of `top_seg`.

    A subtree rooted at node i is counted (cal) when it has >1 member and i is
    not predicted as trachea; it is consistent (top) when every member shares
    pred[i].
    """
    node_num = pred.shape[0]
    cal = 0
    top = 0
    for i in range(node_num):
        if pred[i] == trachea_label:
            continue
        members = pred[mask_top[i] == 1]
        if members.shape[0] <= 1:
            continue
        cal += 1
        if np.all(members == pred[i]):
            top += 1
    return float(top) / cal if cal else 0.0


def compute_td(spd, gt, pred, valid_mask, td_fill=50.0):
    """Topological Distance (TD), per baseline test_2stage.py:compute_td.

    For each valid node i: d_i = min_{j: gt_j == pred_i} spd[i, j]; if no such j
    exists (predicted label absent from GT) or it is unreachable, d_i = `td_fill`
    (baseline sentinel 50). Finite distances are not clamped. TD = mean(d_i).
    """
    idx = np.where(valid_mask)[0]
    d = np.full(idx.shape[0], td_fill, dtype=np.float64)  # no-match / unreachable
    for k, i in enumerate(idx):
        targets = np.where(gt == pred[i])[0]
        if targets.size:
            dmin = np.min(spd[i, targets])
            if np.isfinite(dmin):
                d[k] = dmin
    return float(np.mean(d)) if d.size else 0.0


# --- Anatomical class hierarchy for TAcc (TreeAcc) -------------------------------
# Official Track-2 segmental scheme grouped by (side, lobe). The taxonomy is
#   trachea -> side (L/R main bronchus) -> lobe -> segment,
# with trachea (19) as an INTERNAL node (the root) and the segmental classes as
# leaves. Class-to-class distance = shortest-path hops in this tree. Background (0)
# and any out-of-scheme label are treated as maximally distant (full penalty).
#
# Traceability to the baseline (verified): the (side, lobe) grouping below
# reproduces, exactly, the baseline's hierarchy mappings under the empirically
# established label bijection `official = baseline + 1`:
#   - lobe  <- GCN/datasetgcn.py:seg2lobor  (baseline seg -> lobe 1..5:
#       1=LUL, 2=LLL, 3=RUL, 4=RML, 5=RLL; trachea -> no lobe; LB7 (baseline 19)
#       handled by its explicit `==19 -> lobe 2` (LLL) branch).
#   - side  <- airwaynet/test_2stage.py:Class2Anno (Anno[:,0]: 1=left, 2=right).
# Official scheme names per imr_atm26_track2_test_GPU_TIME/resources/dataset.json.
_SEG_GROUP = {
    1: ("L", "LUL"), 2: ("L", "LUL"), 3: ("L", "LUL"), 4: ("L", "LUL"),
    5: ("L", "LLL"), 6: ("L", "LLL"), 7: ("L", "LLL"), 8: ("L", "LLL"), 20: ("L", "LLL"),
    9: ("R", "RUL"), 10: ("R", "RUL"), 11: ("R", "RUL"),
    12: ("R", "RML"), 13: ("R", "RML"),
    14: ("R", "RLL"), 15: ("R", "RLL"), 16: ("R", "RLL"), 17: ("R", "RLL"), 18: ("R", "RLL"),
}


def _class_path(c):
    """Root->class path in the taxonomy; None if c is not in the hierarchy."""
    if c == TRACHEA_LABEL:
        return ("trachea",)
    if c in _SEG_GROUP:
        side, lobe = _SEG_GROUP[c]
        return ("trachea", side, lobe, c)
    return None  # background / unknown


def _hier_distance(a, b):
    """Hops between classes a and b in the taxonomy; None if either is off-tree."""
    if a == b:
        return 0
    pa, pb = _class_path(a), _class_path(b)
    if pa is None or pb is None:
        return None
    common = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        common += 1
    return (len(pa) - common) + (len(pb) - common)


def build_class_distance_matrix(max_class):
    """Symmetric class-to-class distance matrix D over labels 0..max_class.

    D[c, c'] = taxonomy hops; D[c, c] = 0. Off-tree labels (background / unknown)
    are set to D_max, so mislabeling a branch as background incurs full penalty.
    Returns (D, d_max). For the segmental scheme d_max = 6 (segments in opposite
    lungs: seg->lobe->side->trachea->side->lobe->seg).
    """
    n = max_class + 1
    hier = [c for c in range(n) if _class_path(c) is not None]
    d_max = max((_hier_distance(a, b) for a in hier for b in hier), default=1)
    D = np.full((n, n), float(d_max), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            d = _hier_distance(a, b)
            if d is not None:
                D[a, b] = d
    return D, float(d_max)


def compute_tacc(gt, pred, valid_mask, dist_matrix, d_max):
    """TAcc (TreeAcc): accuracy weighted by anatomical class-to-class distance.

    Per valid node i: score_i = 1 - D[pred_i, gt_i] / d_max (linear). Correct -> 1;
    a near-miss in the class hierarchy -> high partial credit; a far miss -> ~0.
    TAcc = mean_i score_i, and is always >= plain accuracy.
    """
    idx = np.where(valid_mask)[0]
    if idx.size == 0:
        return 0.0
    dvals = dist_matrix[pred[idx], gt[idx]]
    return float(np.mean(1.0 - dvals / d_max))
