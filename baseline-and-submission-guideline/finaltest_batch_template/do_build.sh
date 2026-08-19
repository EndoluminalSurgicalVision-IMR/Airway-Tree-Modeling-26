#!/usr/bin/env bash
# Build the batch template image locally.
set -e
IMAGE_NAME="${IMAGE_NAME:-atm26-batch-template:latest}"
docker build -t "${IMAGE_NAME}" .
echo "Built ${IMAGE_NAME}"
