#!/usr/bin/env bash
# Save the image as a timestamped archive for upload to the ATM26
# Validation / Final Test phase.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="imr_atm26_track1_leaderboard_baseline_finaltest_batch"

echo "= STEP 1 = (Re)build the image"
source "${SCRIPT_DIR}/do_build.sh"

build_timestamp=$(docker inspect --format='{{ .Created }}' "${DOCKER_IMAGE_TAG}")
formatted_build_info=$(date -d "${build_timestamp}" +"%Y%m%d_%H%M%S")
output_filename="${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"
output_path="${SCRIPT_DIR}/${output_filename}"

echo "= STEP 2 = Save the image"
docker save "${DOCKER_IMAGE_TAG}" | gzip -c > "${output_path}"
echo "Saved as: ${output_path}"
