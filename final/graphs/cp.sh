#!/usr/bin/bash

ROOT="$(git rev-parse --show-toplevel)"

set -ex

DATA_DIR="${ROOT}/final"
# GRAPHS_DIR="${DATA_DIR}/graphs"
GRAPHS_DIR="${DATA_DIR}/paper_graphs/paper_graphs"
# FIG_DEST_DIR="${HOME}/dosieroj/rmc/tezo/papero/img/fig"
FIG_DEST_DIR="${HOME}/papers/papero/img/fig"


rsync -avx "${GRAPHS_DIR}"/ "${FIG_DEST_DIR}"
