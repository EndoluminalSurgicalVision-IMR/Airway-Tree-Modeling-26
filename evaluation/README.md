# Evaluation
The ATM'26 challenge aims at extracting accurate binary airway segmentation (Track-1) and accurate branch-wise airway anatomical labeling (Track-2). For each track, the final ranking is determined by the **mean rank position** of a submission across all the ranking metrics described below.

## Track-1: Binary Airway Segmentation

The following metrics are incorporated into the final ranking:

1. **Dice** — the Dice similarity coefficient measuring the volumetric overlap between the prediction and the reference.
2. **clDice** — the skeleton-based (centerline) Dice measuring the topological overlap between the prediction and the reference [1].
3. **TLD** — the tree length detected rate of the airway centerline.
4. **BD** — the branch detected rate of the airway branches.
5. **Betti Error** — the difference in the number of connected components between the prediction and the reference.

Specifically, the TLD is the fraction detected correctly relative to the total tree length in the reference label, and the BD is the percentage of branches that are detected correctly with respect to the total number of branches present in the reference label. clDice is the skeleton-based Dice metric which only measures the Dice metric after the skeletonisation of the prediction. Betti-Error measures the difference of betti-0 value (the number of connected components) between the prediction and ground truth.

## Track-2: Branch-wise Anatomical Labeling

The following metrics are incorporated into the final ranking:

1. **ACC** — the branch-wise accuracy of the per-branch anatomical labeling.
2. **F1** — the branch-wise, macro-averaged F1 score of the per-branch anatomical labeling.
3. **SC** — the Subtree Consistency, the proportion of subtrees with consistent segmental labeling.
4. **TD** — the Topological Distance between predicted branches and their matched ground-truth nodes.
5. **Dice** — the multi-class Dice similarity coefficient measuring the voxel-level labeling overlap between the prediction and the reference.
6. **clDice** — the multi-class skeleton-based (centerline) Dice measuring the voxel-level labeling overlap between the prediction and the reference [1].

The branch-wise ACC and F1 and the subtree-wise SC and TD are measured on the airway graph [2]; for a clinically meaningful airway extraction, only the largest connected component of each predicted segmentation is kept, which is then projected onto the reference airway graph. Dice and clDice are computed as voxel-level multi-class labeling overlap measures.

**Branch-wise metrics.** ACC and F1 evaluate the per-branch anatomical labeling against the reference.

**Subtree Consistency (SC).** SC evaluates the proportion of subtrees meeting the segmental consistency criterion, i.e., all nodes in a subtree share the same predicted segmental label. It is computed as the ratio SC = Ncs / Ns, where Ns is the total number of anatomical subtrees and Ncs is the number of consistently labeled subtrees.

**Topological Distance (TD).** TD measures the shortest path length between predicted branches and their corresponding ground-truth subtrees, i.e., the average graph distance between predicted and ground-truth matched nodes. For each of the N nodes in the airway graph, we take the shortest path length d(i, j) from node i to the nearest node j whose ground-truth label equals the predicted label of node i, and TD is the mean of these distances over all nodes.

## References

[1] Shit S, Paetzold J C, Sekuboyina A, et al. clDice — a novel topology-preserving loss function for tubular structure segmentation[C]. CVPR, 2021: 16560-16569.

[2] Li C, Zhang M, Zhang C, Gu Y. Reflecting topology consistency and abnormality via learnable attentions for airway labeling[J]. International Journal of Computer Assisted Radiology and Surgery, 2025, 20(7): 1315-1323.