#!/usr/bin/env bash
# Local smoke test: run the container against ./test/input (one or more
# lung-ct .mha files) and write outputs to ./test/output.
set -e
IMAGE_NAME="${IMAGE_NAME:-atm26-batch-template:latest}"

mkdir -p test/output
docker run --rm \
    -v "$(pwd)/test/input":/input:ro \
    -v "$(pwd)/test/output":/output:rw \
    --shm-size=8g \
    "${IMAGE_NAME}"

echo "Outputs:"
find test/output -type f | sort
