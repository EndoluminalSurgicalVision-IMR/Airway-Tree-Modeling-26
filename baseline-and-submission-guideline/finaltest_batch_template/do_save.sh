#!/usr/bin/env bash
# Save the image as a compressed archive for upload to the ATM26 test phase.
set -e
IMAGE_NAME="${IMAGE_NAME:-atm26-batch-template:latest}"
OUTPUT_NAME="${OUTPUT_NAME:-atm26-batch-template.tar.gz}"
docker save "${IMAGE_NAME}" | gzip -1 > "${OUTPUT_NAME}"
echo "Saved ${OUTPUT_NAME}"
