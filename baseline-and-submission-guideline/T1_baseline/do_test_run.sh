#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="imr_atm26_track1_leaderboard_baseline_1_fold"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"
INPUT_DIR="${SCRIPT_DIR}/test/input/case1"
OUTPUT_DIR="${SCRIPT_DIR}/test/output/case1"
VALIDATOR="${SCRIPT_DIR}/test/validate_output.py"

if [ ! -f "${INPUT_DIR}/images/lung-ct/ATM_261_0000.mha" ]; then
  echo "Missing manual test input: ${INPUT_DIR}/images/lung-ct/ATM_261_0000.mha" >&2
  exit 1
fi

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

mkdir -p "${OUTPUT_DIR}"
chmod -f o+rwX "${OUTPUT_DIR}"

docker run --rm \
  --platform=linux/amd64 \
  --quiet \
  --volume "${OUTPUT_DIR}":/output \
  --entrypoint /bin/sh \
  "${DOCKER_IMAGE_TAG}" \
  -c "rm -rf /output/*"

docker volume create "${DOCKER_NOOP_VOLUME}" > /dev/null
cleanup() {
  docker run --rm \
    --platform=linux/amd64 \
    --quiet \
    --volume "${OUTPUT_DIR}":/output \
    --entrypoint /bin/sh \
    "${DOCKER_IMAGE_TAG}" \
    -c "chmod -R -f o+rwX /output/* || true"
  docker volume rm "${DOCKER_NOOP_VOLUME}" > /dev/null
}
trap cleanup EXIT

echo "=+= Running ATM_261"
docker run --rm --gpus all \
  --platform=linux/amd64 \
  --network none \
  --volume "${INPUT_DIR}":/input:ro \
  --volume "${OUTPUT_DIR}":/output \
  --volume "${DOCKER_NOOP_VOLUME}":/tmp \
  "${DOCKER_IMAGE_TAG}"

OUTPUT_SOCKET="${OUTPUT_DIR}/images/binary-airway-segmentation"
mapfile -t outputs < <(find "${OUTPUT_SOCKET}" -maxdepth 1 -type f -name '*.mha' | sort)
if [ "${#outputs[@]}" -ne 1 ]; then
  echo "Expected exactly one .mha output, found ${#outputs[@]}" >&2
  exit 1
fi
echo "=+= Wrote ${outputs[0]}"

echo "=+= Validating ATM_261 output"
docker run --rm \
  --platform=linux/amd64 \
  --network none \
  --volume "${INPUT_DIR}":/input:ro \
  --volume "${OUTPUT_DIR}":/output:ro \
  --volume "${VALIDATOR}":/validate_output.py:ro \
  --entrypoint python \
  "${DOCKER_IMAGE_TAG}" \
  /validate_output.py --input /input --output /output
