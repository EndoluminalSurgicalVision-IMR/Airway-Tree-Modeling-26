# Track 3 scoring and ranking rules v1

This version ranks submissions by the mean rank over several metrics. The four metrics below are implemented as host-side Python
scoring; they were validated on small samples of both the validation and test
paths, and regression-checked against the existing full-test predictions.
The machine-readable policy is in [policy.json](policy.json); the label set is
in [label_schema.json](label_schema.json).

| Ranking metric | What it measures | Computation and aggregation | Direction |
| --- | --- | --- | --- |
| `frame_macro_f1` | recognition quality per anatomical class | confusion matrix pooled over all frames; arithmetic mean of the fixed 32-class F1 | higher is better |
| `frame_accuracy` | overall per-frame correctness | correct frames / all frames | higher is better |
| `sequence_edit` | order of the anatomical labels and spurious transitions | per-sequence edit score of the compressed label runs; averaged within a case, then equally over cases | higher is better |
| `segment_f1_50` | completeness and temporal extent of contiguous label segments | per-sequence segment F1 at IoU ≥ 0.5; averaged within a case, then equally over cases | higher is better |

All scores are stored in 0–1 and may be displayed ×100. Case-equal aggregation
applies only to the last two metrics; the first two are pooled over all frames,
keeping the direct meaning of frame accuracy.

## Per-frame computation

For class c, `F1_c = 2 TP_c / (2 TP_c + FP_c + FN_c)`. A zero denominator is
recorded as 0.
The leaderboard macro-F1 always divides by the fixed 32, independent of which
classes the participant predicted or appear in the current sample.
On a small demo that does not cover all classes, even a prediction that is
correct on every frame can score below 1 for the fixed 32-class macro-F1; a demo
score is not a full-test score.
Auxiliary results also report macro-F1 over the classes present in the ground
truth only, balanced accuracy, weighted-F1, per-class support and the confusion
matrix.
Checkpoint selection during training used `macro_f1_present`; the full
validation set contains all 32 classes, where both macro-F1 values are equal.

## Sequence edit score

Adjacent duplicate labels are merged first. For example, the ground truth
`100,100,101,101,0,0` compresses to `100,101,0`.
The prediction `100,101,100,101,0` adds two extra segments, so its edit score is
`1 - 2/5 = 0.6`.
The formula is `1 - Levenshtein(reference_runs, prediction_runs) / max(reference_run_count, prediction_run_count)`.
Substitutions, insertions and deletions all cost 1. It measures segment order
only and does not require any particular temporal model.

## Segment F1@50

Adjacent frames of the same class form a segment, using half-open frame
intervals. Each predicted segment is matched, in temporal order, with the
ground-truth segment of the same class that has the highest IoU; ties go to the
earliest ground-truth segment.
IoU ≥ 0.5 with an as-yet unmatched ground-truth segment counts as TP, otherwise
as FP; unmatched ground-truth segments count as FN.
`F1@50 = 2 TP / (2 TP + FP + FN)`. Each ground-truth segment is matched at most
once; no background class is excluded because this task has no background label.
This metric penalises splitting one real segment into several predicted
segments; the edit score constrains label order at the same time. Merely
reducing the number of transitions does not automatically yield a high score.
Both are common temporal-segmentation metrics; the definitions follow
[MS-TCN, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.pdf).

## Ranking and ties

1. Rank all valid submissions separately by each of the four metrics, descending.
2. Per-metric scores are rounded to 12 decimals to decide ties; tied entries
   occupy the average of the positions. For example, two tied first places each
   score 1.5 and the next entry is 3.
3. The four per-metric ranks are averaged with equal weight; a lower mean rank
   is better. Ranks are averaged, not the four raw scores.
4. Equal final mean ranks share the final rank, e.g. `1,1,3`. Submission IDs only
   stabilise the display order.
5. Whenever a participant is added, the mean ranks of all valid submissions must
   be recomputed. Reports from different challenge phases or different test sets
   must never be mixed.

The leaderboard only uses JSON produced by the official host-side scorer, never
metrics entered by participants. Validation and test are ranked
separately and their reports are never mixed.

## Output errors and auxiliary diagnostics

A missing sequence, missing frame, extra frame, duplicate frame, extra file,
illegal class, wrong row order or malformed CSV invalidates the whole submission
and excludes it from the ranking.
No zero-filling, no skipping bad frames, no scoring of only the successful
cases. `0=LB1+2` is a normal class; `18` is unused.

The evaluation report keeps per-sequence and per-case metrics plus the number of
prediction/ground-truth transitions; "fewer transitions is better" is not a
ranking rule. Counting illegal transitions needs a reviewed connectivity graph;
when it is absent the count is `null`, not 0. The template
`transition_graph.template.json` defaults to `reviewed=false` and must not be
enabled as is.
A reviewed graph must cover the 32 classes, forward/backward motion and missing
annotations at boundaries; this diagnostic does not participate in ranking.

The evaluator leaves predictions untouched: no smoothing, no Viterbi, no
ground-truth-guided correction. The baseline uses an independent per-frame
CLIP+MLP; that choice does not constrain the methods participants may use.
The current interface provides whole sequences, so future frames are visible;
evaluating strictly online prediction would require a separately defined causal
input, interface and metric, and cannot be inferred from the whole-sequence
output.
