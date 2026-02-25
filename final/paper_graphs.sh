#!/usr/bin/env bash

set -ex

ROOT="$(git rev-parse --show-toplevel)"
PYTHON_DIR="${ROOT}/pitono"
DATA_DIR="${ROOT}/final"
GRAPHS_DIR="${DATA_DIR}/paper_graphs"

FAM_DATA="${DATA_DIR}/results_fam_merged_100_000.json"
SSCA_DATA="${DATA_DIR}/results_ssca_merged_100_000.json"

BERS_SNR="${DATA_DIR}/bers_snr_newrange_100_000.json"
BERS_EBN0="${DATA_DIR}/bers_ebn0_newrange_100_000.json"

PD_SNR_DB_MIN="-25"
PD_SNR_DB_MAX="0"
BER_PD_SNR_DB_MIN="-25"
BER_PD_SNR_DB_MAX="10"
PD_RANGE_ARGS=(--snr-db-min "$PD_SNR_DB_MIN" --snr-db-max "$PD_SNR_DB_MAX")
BER_PD_RANGE_ARGS=(--snr-db-min "$BER_PD_SNR_DB_MIN" --snr-db-max "$BER_PD_SNR_DB_MAX")

generate_cfar_with_split_ranges() {
    local data_file="$1"
    local out_dir="$2"
    local tmp_dir

    uv run python3 "$PYTHON_DIR"/cfar.py -p "$data_file" -s -d "$out_dir" "${PD_RANGE_ARGS[@]}"

    tmp_dir="$(mktemp -d)"
    uv run python3 "$PYTHON_DIR"/cfar.py -p "$data_file" -s -d "$tmp_dir" "${BER_PD_RANGE_ARGS[@]}"
    cp "$tmp_dir"/ber_*.png "$out_dir"/
    rm -rf "$tmp_dir"
}

generate_youden_with_split_ranges() {
    local data_file="$1"
    local out_dir="$2"
    local tmp_dir

    uv run python3 "$PYTHON_DIR"/youden_j.py -p "$data_file" -s -d "$out_dir" "${PD_RANGE_ARGS[@]}"

    tmp_dir="$(mktemp -d)"
    uv run python3 "$PYTHON_DIR"/youden_j.py -p "$data_file" -s -d "$tmp_dir" "${BER_PD_RANGE_ARGS[@]}"
    cp "$tmp_dir"/ber_*.png "$out_dir"/
    rm -rf "$tmp_dir"
}

generate_cfar_with_split_ranges "$FAM_DATA" "$GRAPHS_DIR"/cfar_fam
generate_cfar_with_split_ranges "$SSCA_DATA" "$GRAPHS_DIR"/cfar_ssca

uv run python3 "$PYTHON_DIR"/cfar_grouped.py -p "$FAM_DATA" -s -d "$GRAPHS_DIR"/cfar_fam_grouped "${PD_RANGE_ARGS[@]}"
uv run python3 "$PYTHON_DIR"/cfar_grouped.py -p "$SSCA_DATA" -s -d "$GRAPHS_DIR"/cfar_ssca_grouped "${PD_RANGE_ARGS[@]}"

uv run python3 "$PYTHON_DIR"/cfar_pfa_multi.py -p "$FAM_DATA" -s -d "$GRAPHS_DIR"/cfar_multi_pfa_fam "${PD_RANGE_ARGS[@]}"
uv run python3 "$PYTHON_DIR"/cfar_pfa_multi.py -p "$SSCA_DATA" -s -d "$GRAPHS_DIR"/cfar_multi_pfa_ssca "${PD_RANGE_ARGS[@]}"

uv run python3 "$PYTHON_DIR"/pdfs.py -p "$FAM_DATA" -s -d "$GRAPHS_DIR"/cfar_fam "${PD_RANGE_ARGS[@]}"
uv run python3 "$PYTHON_DIR"/pdfs.py -p "$SSCA_DATA" -s -d "$GRAPHS_DIR"/cfar_ssca "${PD_RANGE_ARGS[@]}"

uv run python3 "$PYTHON_DIR"/tw.py -t "$ROOT"/tw.json -s -d "$GRAPHS_DIR"/tw "${PD_RANGE_ARGS[@]}"

generate_youden_with_split_ranges "$FAM_DATA" "$GRAPHS_DIR"/youden_j_fam
generate_youden_with_split_ranges "$SSCA_DATA" "$GRAPHS_DIR"/youden_j_ssca

uv run python3 "$PYTHON_DIR"/youden_j_grouped.py -p "$FAM_DATA" -s -d "$GRAPHS_DIR"/youden_j_fam_grouped "${PD_RANGE_ARGS[@]}"
uv run python3 "$PYTHON_DIR"/youden_j_grouped.py -p "$SSCA_DATA" -s -d "$GRAPHS_DIR"/youden_j_ssca_grouped "${PD_RANGE_ARGS[@]}"

uv run python3 "$PYTHON_DIR"/bers.py -b "$BERS_SNR" -s -d "$GRAPHS_DIR"/bers "${BER_PD_RANGE_ARGS[@]}"
uv run python3 "$PYTHON_DIR"/bers.py --ebn0 -b "$BERS_EBN0" -s -d "$GRAPHS_DIR"/bers "${BER_PD_RANGE_ARGS[@]}"
uv run python3 "$PYTHON_DIR"/bers_grouped.py -b "$BERS_SNR" -s -d "$GRAPHS_DIR"/bers "${BER_PD_RANGE_ARGS[@]}"
uv run python3 "$PYTHON_DIR"/bers_grouped.py --ebn0 -b "$BERS_EBN0" -s -d "$GRAPHS_DIR"/bers "${BER_PD_RANGE_ARGS[@]}"
