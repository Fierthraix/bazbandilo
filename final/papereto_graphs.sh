#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
PYTHON_DIR="${ROOT}/pitono"
OUT_DIR="${ROOT}/final/papereto_graphs"
PAPER_DIR="${PAPER_DIR:-/mnt/c/Users/Lucas.R/Documents/Papers/papereto/img/fig/papereto_graphs}"
PAPER_CFAR_DIR="${PAPER_CFAR_DIR:-/mnt/c/Users/Lucas.R/Documents/Papers/papereto/img/fig/cfar_ssca}"
BER_FILE="${BER_FILE:-}"

EXTRA_ARGS=()
case "${PAPERETO_MODE:-full}" in
  combo)
    BER_FILE="${BER_FILE:-${ROOT}/final/bers_snr_fig1_merged_100_000.json}"
    EXTRA_ARGS+=(--combo-only)
    ;;
  detector)
    EXTRA_ARGS+=(--detector-behavior-only)
    ;;
  full)
    BER_FILE="${BER_FILE:-${ROOT}/final/bers_snr_newrange_100_000.json}"
    ;;
  *)
    echo "Unknown PAPERETO_MODE=${PAPERETO_MODE}; expected full, combo, or detector" >&2
    exit 2
    ;;
esac

uv run --no-project --with numpy --with matplotlib --with ijson python3 "${PYTHON_DIR}/cfar_papereto.py" \
  --ssca-results "${ROOT}/final/results_ssca_merged_100_000.json" \
  --fam-results "${ROOT}/final/results_fam_merged_100_000.json" \
  --ber-file "${BER_FILE}" \
  --save-dir "${OUT_DIR}" \
  --paper-dir "${PAPER_DIR}" \
  --paper-cfar-dir "${PAPER_CFAR_DIR}" \
  "${EXTRA_ARGS[@]}"
