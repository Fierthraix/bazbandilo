#!/usr/bin/bash

ROOT="$(git rev-parse --show-toplevel)"

set -ex

DATA_DIR="${ROOT}/final"
GRAPHS_DIR="${DATA_DIR}/graphs"
FIG_DEST_DIR="${HOME}/dosieroj/rmc/tezo/papero/img/fig"


rsync -avx "${GRAPHS_DIR}"/ "${FIG_DEST_DIR}"
