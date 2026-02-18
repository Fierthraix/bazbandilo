#!/usr/bin/env bash

set -ex

ROOT="$(git rev-parse --show-toplevel)"
PYTHON_DIR="${ROOT}/pitono"
DATA_DIR="${ROOT}/final"
GRAPHS_DIR="${DATA_DIR}/graphs"

FAM_DATA="${DATA_DIR}/results_fam_merged_100_000.json"
SSCA_DATA="${DATA_DIR}/results_ssca_merged_100_000.json"

BERS_SNR="${DATA_DIR}/bers_snr_newrange_100_000.json"
BERS_EBN0="${DATA_DIR}/bers_ebn0_newrange_100_000.json"

uv run python3 "$PYTHON_DIR"/cfar.py -p "$FAM_DATA" -s -d "$GRAPHS_DIR"/cfar_fam
uv run python3 "$PYTHON_DIR"/cfar.py -p "$SSCA_DATA" -s -d "$GRAPHS_DIR"/cfar_ssca

uv run python3 "$PYTHON_DIR"/cfar_grouped.py -p "$FAM_DATA" -s -d "$GRAPHS_DIR"/cfar_fam_grouped
uv run python3 "$PYTHON_DIR"/cfar_grouped.py -p "$SSCA_DATA" -s -d "$GRAPHS_DIR"/cfar_ssca_grouped

uv run python3 "$PYTHON_DIR"/cfar_pfa_multi.py -p "$FAM_DATA" -s -d "$GRAPHS_DIR"/cfar_multi_pfa_fam
uv run python3 "$PYTHON_DIR"/cfar_pfa_multi.py -p "$SSCA_DATA" -s -d "$GRAPHS_DIR"/cfar_multi_pfa_ssca

uv run python3 "$PYTHON_DIR"/pdfs.py -p "$FAM_DATA" -s -d "$GRAPHS_DIR"/cfar_fam
uv run python3 "$PYTHON_DIR"/pdfs.py -p "$SSCA_DATA" -s -d "$GRAPHS_DIR"/cfar_ssca

uv run python3 "$PYTHON_DIR"/tw.py -t "$ROOT"/tw.json -s -d "$GRAPHS_DIR"/tw

uv run python3 "$PYTHON_DIR"/youden_j.py -p "$FAM_DATA" -s -d "$GRAPHS_DIR"/youden_j_fam
uv run python3 "$PYTHON_DIR"/youden_j.py -p "$SSCA_DATA" -s -d "$GRAPHS_DIR"/youden_j_ssca

uv run python3 "$PYTHON_DIR"/youden_j_grouped.py -p "$FAM_DATA" -s -d "$GRAPHS_DIR"/youden_j_fam_grouped
uv run python3 "$PYTHON_DIR"/youden_j_grouped.py -p "$SSCA_DATA" -s -d "$GRAPHS_DIR"/youden_j_ssca_grouped

uv run python3 "$PYTHON_DIR"/bers.py -b "$BERS_SNR" -s -d "$GRAPHS_DIR"/bers
uv run python3 "$PYTHON_DIR"/bers.py --ebn0 -b "$BERS_EBN0" -s -d "$GRAPHS_DIR"/bers
uv run python3 "$PYTHON_DIR"/bers_grouped.py -b "$BERS_SNR" -s -d "$GRAPHS_DIR"/bers
uv run python3 "$PYTHON_DIR"/bers_grouped.py --ebn0 -b "$BERS_EBN0" -s -d "$GRAPHS_DIR"/bers
