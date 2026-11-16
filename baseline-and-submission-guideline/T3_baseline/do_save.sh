#!/usr/bin/env bash
set -euo pipefail
IMAGE="${IMAGE:-atm26-track3-baseline:v3}"
OUTPUT="${1:-atm26-track3-baseline-v3.tar.gz}"
[[ ! -e "$OUTPUT" ]] || { echo "Output exists" >&2; exit 1; }
docker save "$IMAGE" | gzip > "$OUTPUT"
