#!/usr/bin/env bash

ROOT="$(git rev-parse --show-toplevel)"
DATA_DIR="${ROOT}/final"
GRAPHS_DIR="${DATA_DIR}/graphs"

set -ex

fd -e png -E misc '' "$GRAPHS_DIR" -X rm -vf
