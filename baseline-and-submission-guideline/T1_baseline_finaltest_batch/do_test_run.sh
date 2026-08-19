#!/usr/bin/env bash
# Local smoke test for the dual-mode wrapper (single-case AND batch paths).
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="imr_atm26_track1_leaderboard_baseline_finaltest_batch"
OUTPUT_SLUG="binary-airway-segmentation"

source "${SCRIPT_DIR}/do_build.sh"

run_case_dir() {
  local input_dir="$1" output_dir="$2" expected="$3"
  mkdir -p "${output_dir}"
  docker run --rm --gpus all --network none     --volume "${input_dir}":/input:ro     --volume "${output_dir}":/output     "${DOCKER_IMAGE_TAG}"
  mapfile -t outputs < <(find "${output_dir}/images/${OUTPUT_SLUG}" -maxdepth 1 -type f -name '*.mha' | sort)
  if [ "${#outputs[@]}" -ne "${expected}" ]; then
    echo "Expected ${expected} .mha output(s), found ${#outputs[@]}" >&2
    exit 1
  fi
  printf 'wrote %s output(s)\n' "${#outputs[@]}"
}

if [ -d "${SCRIPT_DIR}/test/input/case1" ]; then
  echo "=+= Single-case mode"
  run_case_dir "${SCRIPT_DIR}/test/input/case1" "${SCRIPT_DIR}/test/output/case1" 1
fi

if [ -d "${SCRIPT_DIR}/test/input/batch" ]; then
  echo "=+= Batch mode"
  count=$(find "${SCRIPT_DIR}/test/input/batch/images/lung-ct" -maxdepth 1 -type f -name '*.mha' | wc -l)
  run_case_dir "${SCRIPT_DIR}/test/input/batch" "${SCRIPT_DIR}/test/output/batch" "${count}"
fi
