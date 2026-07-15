#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="imr_atm26_track2_leaderboard_baseline_1_fold"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"
INPUT_DIR="${SCRIPT_DIR}/test/input/interf0"
OUTPUT_DIR="${SCRIPT_DIR}/test/output/interf0"

if [ ! -f "${INPUT_DIR}/inputs.json" ]; then
  echo "Missing Grand Challenge test input: ${INPUT_DIR}/inputs.json" >&2
  exit 1
fi

mapfile -t inputs < <(find "${INPUT_DIR}/images/lung-ct" -maxdepth 1 -type f -name '*.mha' | sort)
if [ "${#inputs[@]}" -ne 1 ]; then
  echo "Expected exactly one .mha in ${INPUT_DIR}/images/lung-ct, found ${#inputs[@]}" >&2
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

echo "=+= Running ${inputs[0]}"
docker run --rm --gpus all \
  --platform=linux/amd64 \
  --network none \
  --volume "${INPUT_DIR}":/input:ro \
  --volume "${OUTPUT_DIR}":/output \
  --volume "${DOCKER_NOOP_VOLUME}":/tmp \
  "${DOCKER_IMAGE_TAG}"

OUTPUT_SOCKET="${OUTPUT_DIR}/images/multi-class-airway-segmentation"
mapfile -t outputs < <(find "${OUTPUT_SOCKET}" -maxdepth 1 -type f -name '*.mha' | sort)
if [ "${#outputs[@]}" -ne 1 ]; then
  echo "Expected exactly one .mha output, found ${#outputs[@]}" >&2
  exit 1
fi
echo "=+= Wrote ${outputs[0]}"
