# T3_evaluation: official host-side scoring

The role follows the Track-1/2 evaluation: read predictions and the private
ground truth, then write `metrics.json`. As requested by the organizers, this
version runs directly in host Python and has no evaluation-Docker dependency.
It is not a unit test; the unit tests live under `verification/` in the release
bundle.

```bash
python evaluate.py --predictions-root /path/to/docker_output --gt-root /path/to/private_gt --output-root /path/to/scores
python rank.py --reports /path/to/trusted_reports --output /path/to/leaderboard
```

Validation and test use the same script; only the host folder paths change. The
interface, the GT manifest, the four metrics and the tie rules are described in
the root README and in [SCORING.md](SCORING.md).

`prepare_data.py` prepares the existing ATM data; `run_submission.py` and
`run_from_config.py` are optional host-side Docker wrappers for the organizers —
participants never submit them. The four host paths in `paths.example.json` can
be edited directly; `/input` and `/output` inside the container are a fixed
interface, not host disk paths.
