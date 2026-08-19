#!/usr/bin/env bash
# Build the FinalTestPhase batch baseline image locally.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="imr_atm26_track1_leaderboard_baseline_finaltest_batch"

docker build --platform=linux/amd64 -t "${DOCKER_IMAGE_TAG}" "${SCRIPT_DIR}"
echo "Built ${DOCKER_IMAGE_TAG}"
