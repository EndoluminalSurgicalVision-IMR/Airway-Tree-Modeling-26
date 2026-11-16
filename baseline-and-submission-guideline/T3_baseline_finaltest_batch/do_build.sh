#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
docker build --platform linux/amd64 -t "${IMAGE:-atm26-track3-baseline-batch:v3}" .
