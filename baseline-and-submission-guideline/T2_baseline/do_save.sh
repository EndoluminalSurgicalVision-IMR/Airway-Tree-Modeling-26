#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="imr_atm26_track2_leaderboard_baseline_1_fold"

echo "= STEP 1 = (Re)build the image"
export DOCKER_QUIET_BUILD=1
source "${SCRIPT_DIR}/do_build.sh"

build_timestamp=$(docker inspect --format='{{ .Created }}' "${DOCKER_IMAGE_TAG}")
formatted_build_info=$(date -d "${build_timestamp}" +"%Y%m%d_%H%M%S")
output_filename="${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"
output_path="${SCRIPT_DIR}/${output_filename}"

echo "= STEP 2 = Save the image"
docker save "${DOCKER_IMAGE_TAG}" | gzip -c > "${output_path}"
echo "Saved as: ${output_path}"
