#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: bash do_test_run.sh INPUT_ROOT OUTPUT_ROOT" >&2; exit 2; }
INPUT="$(cd -- "$1" && pwd)"
mkdir -p -- "$2"
OUTPUT="$(cd -- "$2" && pwd)"
[[ -z "$(ls -A -- "$OUTPUT")" ]] || { echo "Output must be empty" >&2; exit 1; }
chmod o+rwX "$OUTPUT"
OPTIONS=(--rm --network none --platform linux/amd64 --mount "type=bind,src=$INPUT,dst=/input,readonly" --mount "type=bind,src=$OUTPUT,dst=/output")
[[ "${DEVICE:-cuda}" != cuda ]] || OPTIONS+=(--gpus all)
docker run "${OPTIONS[@]}" "${IMAGE:-atm26-track3-baseline:v3}"
