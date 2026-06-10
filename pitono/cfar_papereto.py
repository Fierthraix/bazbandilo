#!/usr/bin/env python3
"""Papereto compact CFAR summary plots.

This script intentionally generates only the replacement figures needed for the
paper revision, rather than regenerating the full historical plot set.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import gc
import json
import math
import mmap
from pathlib import Path
from typing import Iterable

import ijson
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


PAPERETO_DIR = Path("/mnt/c/Users/Lucas.R/Documents/Papers/papereto")
PAPERETO_FIG_DIR = PAPERETO_DIR / "img" / "fig" / "papereto_graphs"

DETECTORS = ("Energy", "MaxCut", "Dcs")
TARGET_BER = 1e-2
TARGET_PD = 0.5
PFA = 0.01
COMBO_SNR_DB_MIN = -25
COMBO_SNR_DB_MAX = 10

REPRESENTATIVE_MODS = [
    ("CDMA-QPSK-64", "CDMA-QPSK-64"),
    ("CDMA-BPSK-16", "CDMA-BPSK-16"),
    ("BPSK", "BPSK"),
    ("OFDM-BPSK-64", "OFDM-BPSK-64"),
    ("64QAM", "64-QAM"),
    ("CSK", "CSK"),
]

SUMMARY_MODS = [
    ("CDMA-QPSK-64", "CDMA-QPSK-64"),
    ("CDMA-BPSK-16", "CDMA-BPSK-16"),
    ("BPSK", "BPSK"),
    ("OFDM-BPSK-64", "OFDM-BPSK-64"),
    ("QPSK", "QPSK"),
    ("64QAM", "64-QAM"),
    ("BFSK-64", "BFSK"),
    ("CSS-64", "CSS"),
    ("CSK", "CSK"),
]

COMBO_BER_PD_MODS = [
    ("CDMA-QPSK-64", "CDMA-QPSK-64"),
    ("DCSK", "DCSK"),
]

MAXCUT_PD_SNR_MODS = [
    ("CSK", "CSK"),
    ("CDMA-QPSK-64", "CDMA-QPSK-64"),
    ("CDMA-BPSK-16", "CDMA-BPSK-16"),
    ("BPSK", "BPSK"),
    ("OFDM-BPSK-64", "OFDM-BPSK-64"),
    ("64QAM", "64-QAM"),
]

DCS_HIGH_BRANCH = ["BFSK-64", "CDMA-QPSK-64", "CSK"]
DCS_LOW_BRANCH = ["BPSK", "OFDM-BPSK-64", "64QAM", "CSS-64", "DCSK"]

STYLE_BY_MOD = {
    "CDMA-QPSK-64": ("#009E73", "-"),
    "CDMA-BPSK-16": ("#E69F00", "--"),
    "BPSK": ("#0072B2", "-"),
    "OFDM-BPSK-64": ("#D55E00", "-."),
    "64QAM": ("#CC79A7", ":"),
    "CSK": ("#000000", (0, (3, 1, 1, 1))),
    "BFSK-64": ("#56B4E9", "--"),
    "CSS-64": ("#F0E442", ":"),
    "DCSK": ("#999999", "-."),
}

MARKERS_BY_MOD = {
    "CDMA-QPSK-64": "o",
    "CDMA-BPSK-16": "s",
    "BPSK": "^",
    "OFDM-BPSK-64": "D",
    "64QAM": "v",
    "CSK": "P",
    "BFSK-64": "X",
    "CSS-64": "<",
    "DCSK": ">",
}

ALL_MODS = sorted(
    {
        name
        for name, _ in REPRESENTATIVE_MODS
        + SUMMARY_MODS
        + COMBO_BER_PD_MODS
        + MAXCUT_PD_SNR_MODS
    }
    | set(DCS_HIGH_BRANCH)
    | set(DCS_LOW_BRANCH)
)


@dataclass
class ParsedModulation:
    name: str
    snrs_db: np.ndarray
    pds: dict[str, np.ndarray]


def db(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.asarray(values, dtype=float))


def parse_args() -> Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = ArgumentParser()
    ap.add_argument(
        "--ssca-results",
        type=Path,
        default=root / "final" / "results_ssca_merged_100_000.json",
    )
    ap.add_argument(
        "--fam-results",
        type=Path,
        default=root / "final" / "results_fam_merged_100_000.json",
    )
    ap.add_argument(
        "--ber-file",
        type=Path,
        default=root / "final" / "bers_snr_newrange_100_000.json",
    )
    ap.add_argument(
        "-d",
        "--save-dir",
        type=Path,
        default=root / "final" / "papereto_graphs",
    )
    ap.add_argument(
        "--paper-dir",
        type=Path,
        default=PAPERETO_FIG_DIR,
        help="Destination in the paper repo. Use an empty string to skip copying.",
    )
    ap.add_argument(
        "--paper-cfar-dir",
        type=Path,
        default=PAPERETO_DIR / "img" / "fig" / "cfar_ssca",
        help="Destination for regenerated BER/PD combo figures. Use an empty string to skip copying.",
    )
    ap.add_argument(
        "--combo-only",
        action="store_true",
        help="Only regenerate the BER/PD combo figures used in the paper.",
    )
    ap.add_argument(
        "--detector-behavior-only",
        action="store_true",
        help="Only regenerate the selected detector behavior figure used in the paper.",
    )
    return ap.parse_args()


def load_ber_curves(path: Path, wanted: set[str]) -> dict[str, dict[str, np.ndarray]]:
    with path.open("r") as f:
        records = json.load(f)
    out = {}
    for record in records:
        if record["name"] in wanted:
            out[record["name"]] = {
                "snrs_db": db(record["snrs"]),
                "bers": np.asarray(record["bers"], dtype=float),
            }
    missing = sorted(wanted - set(out))
    if missing:
        raise RuntimeError(f"Missing BER curves in {path}: {missing}")
    return out


def load_selected_cfar(path: Path, wanted: set[str]) -> dict[str, ParsedModulation]:
    """Compute selected CFAR curves from top-level JSON records.

    The merged result files are several GB.  Scanning directly to the wanted
    top-level modulation records avoids parsing unrelated Monte Carlo trials.
    """
    print(f"Loading selected CFAR curves from {path}", flush=True)
    work: dict[str, dict[str, object]] = {}
    for current_mod, record in iter_named_records(path, wanted):
        print(f"  found {current_mod}", flush=True)
        snrs_db: np.ndarray | None = None
        pds_by_detector: dict[str, np.ndarray] = {}
        for detector in record["results"]:
            detector_kind = detector["kind"]
            if detector_kind not in DETECTORS:
                continue
            print(f"    {detector_kind}", flush=True)
            snrs_db = db(detector["snrs"])
            pds = []
            for h0_values, h1_values in zip(detector["h0_λs"], detector["h1_λs"]):
                threshold = float(np.quantile(np.asarray(h0_values, dtype=float), 1 - PFA))
                h1 = np.asarray(h1_values, dtype=float)
                pds.append(float(np.mean(h1 > threshold)))
            pds_by_detector[detector_kind] = np.asarray(pds, dtype=float)
        missing_detectors = sorted(set(DETECTORS) - set(pds_by_detector))
        if missing_detectors:
            raise RuntimeError(f"Missing detectors for {current_mod}: {missing_detectors}")
        if snrs_db is None:
            raise RuntimeError(f"Missing SNRs for {current_mod}")
        work[current_mod] = {"snrs_db": snrs_db, "pds": pds_by_detector}
        del record
        gc.collect()

    return {
        name: ParsedModulation(
            name=name,
            snrs_db=work[name]["snrs_db"],
            pds=work[name]["pds"],
        )
        for name in wanted
    }


def iter_named_records(path: Path, wanted: set[str]):
    found = set()
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for name in sorted(wanted):
            needle = f'"name": "{name}"'.encode()
            idx = mm.find(needle)
            if idx < 0:
                continue
            start = mm.rfind(b"{", 0, idx)
            end = matching_object_end(mm, start)
            found.add(name)
            yield name, json.loads(mm[start:end])
    missing = sorted(wanted - found)
    if missing:
        raise RuntimeError(f"Missing modulations in {path}: {missing}")


def matching_object_end(mm: mmap.mmap, start: int) -> int:
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(mm)):
        ch = mm[pos]
        if in_string:
            if escape:
                escape = False
            elif ch == 92:  # backslash
                escape = True
            elif ch == 34:  # quote
                in_string = False
            continue
        if ch == 34:
            in_string = True
        elif ch == 123:  # {
            depth += 1
        elif ch == 125:  # }
            depth -= 1
            if depth == 0:
                return pos + 1
    raise RuntimeError("Could not find end of JSON object")


def load_selected_cfar_streaming_backup(path: Path, wanted: set[str]) -> dict[str, ParsedModulation]:
    """Compute selected CFAR curves from the raw merged JSON without full loads."""
    print(f"Streaming selected CFAR curves from {path}", flush=True)
    work: dict[str, dict[str, object]] = {}

    current_mod = ""
    mod_selected = False
    detector_kind = ""
    active_detector = False
    snrs: list[float] = []
    thresholds: list[float] = []
    pds: list[float] = []
    h0_values: list[float] = []
    h1_threshold = math.nan
    h1_count = 0
    h1_total = 0
    collecting = ""

    h0_array = "item.results.item.h0_λs.item"
    h0_value = "item.results.item.h0_λs.item.item"
    h1_array = "item.results.item.h1_λs.item"
    h1_value = "item.results.item.h1_λs.item.item"

    def complete() -> bool:
        return all(
            name in work and set(work[name]["pds"]) >= set(DETECTORS)
            for name in wanted
        )

    with path.open("rb") as f:
        for prefix, event, value in ijson.parse(f, use_float=True):
            if prefix == "item" and event == "start_map":
                current_mod = ""
                mod_selected = False
                continue

            if prefix == "item.name" and event == "string":
                current_mod = str(value)
                mod_selected = current_mod in wanted
                if mod_selected:
                    print(f"  found {current_mod}", flush=True)
                continue

            if prefix == "item.results.item" and event == "start_map":
                detector_kind = ""
                active_detector = False
                snrs = []
                thresholds = []
                pds = []
                collecting = ""
                continue

            if prefix == "item.results.item.kind" and event == "string":
                detector_kind = str(value)
                active_detector = mod_selected and detector_kind in DETECTORS
                if active_detector:
                    print(f"    {detector_kind}", flush=True)
                continue

            if not active_detector:
                continue

            if prefix == "item.results.item.snrs.item" and event == "number":
                snrs.append(float(value))
                continue

            if prefix == h0_array and event == "start_array":
                collecting = "h0"
                h0_values = []
                continue
            if prefix == h0_value and event == "number":
                h0_values.append(float(value))
                continue
            if prefix == h0_array and event == "end_array":
                thresholds.append(float(np.quantile(np.asarray(h0_values), 1 - PFA)))
                h0_values = []
                collecting = ""
                continue

            if prefix == h1_array and event == "start_array":
                if len(pds) >= len(thresholds):
                    raise RuntimeError(
                        f"H1 values arrived before matching H0 threshold for "
                        f"{current_mod}/{detector_kind}"
                    )
                collecting = "h1"
                h1_threshold = thresholds[len(pds)]
                h1_count = 0
                h1_total = 0
                continue
            if prefix == h1_value and event == "number":
                h1_total += 1
                if float(value) > h1_threshold:
                    h1_count += 1
                continue
            if prefix == h1_array and event == "end_array":
                pds.append(h1_count / h1_total if h1_total else math.nan)
                collecting = ""
                continue

            if prefix == "item.results.item" and event == "end_map":
                if collecting:
                    raise RuntimeError(f"Unexpected open collection in {current_mod}")
                if len(snrs) != len(pds):
                    raise RuntimeError(
                        f"SNR/PD length mismatch for {current_mod}/{detector_kind}: "
                        f"{len(snrs)} != {len(pds)}"
                    )
                entry = work.setdefault(current_mod, {"snrs_db": db(snrs), "pds": {}})
                entry["pds"][detector_kind] = np.asarray(pds, dtype=float)
                if complete():
                    break

    missing = sorted(wanted - set(work))
    if missing:
        raise RuntimeError(f"Missing modulations in {path}: {missing}")

    out = {}
    for name in wanted:
        pds_by_detector = work[name]["pds"]
        missing_detectors = sorted(set(DETECTORS) - set(pds_by_detector))
        if missing_detectors:
            raise RuntimeError(f"Missing detectors for {name}: {missing_detectors}")
        out[name] = ParsedModulation(
            name=name,
            snrs_db=work[name]["snrs_db"],
            pds=pds_by_detector,
        )
    return out


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def common_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def interpolate_pd_on_ber(
    mod_name: str,
    pd_mod: ParsedModulation,
    detector: str,
    ber_curves: dict[str, dict[str, np.ndarray]],
    target_ber: float = TARGET_BER,
) -> tuple[float, bool]:
    ber, pd = pd_ber_curve(mod_name, pd_mod, detector, ber_curves)
    finite = np.isfinite(ber) & np.isfinite(pd) & (ber > 0)
    ber = ber[finite]
    pd = pd[finite]

    if np.nanmin(ber) > target_ber:
        return 1.0, False

    order = np.argsort(ber)
    ber = ber[order]
    pd = pd[order]
    log_ber = np.log10(ber)
    target_log = math.log10(target_ber)
    value = float(np.interp(target_log, log_ber, pd))
    return value, True


def transition_snr(
    pd_mod: ParsedModulation, detector: str, target_pd: float = TARGET_PD
) -> float:
    pds = pd_mod.pds[detector]
    snrs_db = pd_mod.snrs_db
    order = np.argsort(snrs_db)
    snrs_db = snrs_db[order]
    pds = pds[order]
    above = np.where(pds >= target_pd)[0]
    if len(above) == 0:
        return np.nan
    idx = int(above[0])
    if idx == 0:
        return float(snrs_db[0])
    return float(
        np.interp(
            target_pd,
            [pds[idx - 1], pds[idx]],
            [snrs_db[idx - 1], snrs_db[idx]],
        )
    )


def pd_ber_curve(
    mod_name: str,
    pd_mod: ParsedModulation,
    detector: str,
    ber_curves: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    ber_snrs = ber_curves[mod_name]["snrs_db"]
    ber = ber_curves[mod_name]["bers"]
    order = np.argsort(ber_snrs)
    ber_snrs = ber_snrs[order]
    ber = ber[order]

    pd_snrs = pd_mod.snrs_db
    pd = pd_mod.pds[detector]
    order = np.argsort(pd_snrs)
    pd_snrs = pd_snrs[order]
    pd = pd[order]

    in_range = (ber_snrs >= pd_snrs.min()) & (ber_snrs <= pd_snrs.max())
    return ber[in_range], np.interp(ber_snrs[in_range], pd_snrs, pd)


def plot_combo_ber_pd(
    ssca: dict[str, ParsedModulation],
    ber_curves: dict[str, dict[str, np.ndarray]],
    save_dir: Path,
) -> list[Path]:
    combo_dir = save_dir / "cfar_ssca"
    combo_dir.mkdir(parents=True, exist_ok=True)
    detector_labels = {
        "Energy": "Radiometer",
        "MaxCut": "Max-Cut",
        "Dcs": "DCS",
    }
    linestyles = {
        "Energy": "solid",
        "MaxCut": "dashed",
        "Dcs": "dashdot",
    }
    paths = []
    for mod_name, _ in COMBO_BER_PD_MODS:
        fig, ber_ax = plt.subplots(figsize=(5.2, 3.3))
        pd_ax = ber_ax.twinx()

        ber_snrs = ber_curves[mod_name]["snrs_db"]
        ber = ber_curves[mod_name]["bers"]
        order = np.argsort(ber_snrs)
        ber_snrs = ber_snrs[order]
        ber = ber[order]
        print(
            f"{mod_name} BER SNR span: {ber_snrs.min():.3f} to {ber_snrs.max():.3f} dB; "
            f"plot window {COMBO_SNR_DB_MIN} to {COMBO_SNR_DB_MAX} dB"
        )
        if ber_snrs.min() > COMBO_SNR_DB_MIN or ber_snrs.max() < COMBO_SNR_DB_MAX:
            print(f"warning: {mod_name} BER data do not cover the full plot window")
        ber_mask = (ber_snrs >= COMBO_SNR_DB_MIN) & (ber_snrs <= COMBO_SNR_DB_MAX)

        ber_ax.grid(True, which="both")
        ber_ax.plot(ber_snrs[ber_mask], ber[ber_mask], color="Red")
        ber_ax.set_yscale("log")
        ber_ax.set_ylim([1e-5, 0.55])
        ber_ax.tick_params(axis="y", colors="Red")
        ber_ax.set_ylabel("Bit Error Rate (BER)", color="Red")

        for detector in DETECTORS:
            pd_snrs = ssca[mod_name].snrs_db
            pd = ssca[mod_name].pds[detector]
            order = np.argsort(pd_snrs)
            pd_snrs = pd_snrs[order]
            pd = pd[order]
            pd_mask = (pd_snrs >= COMBO_SNR_DB_MIN) & (pd_snrs <= COMBO_SNR_DB_MAX)
            pd_ax.plot(
                pd_snrs[pd_mask],
                pd[pd_mask],
                color="Blue",
                linestyle=linestyles[detector],
                label=detector_labels[detector],
            )

        ber_ax.set_xlim(COMBO_SNR_DB_MIN, COMBO_SNR_DB_MAX)
        pd_ax.set_ylim([0, 1.025])
        pd_ax.tick_params(axis="y", colors="Blue")
        pd_ax.set_ylabel(r"Probability of Detection ($P_D$)", color="Blue")
        pd_ax.legend(loc="center left", fontsize=8)
        ber_ax.set_xlabel("Signal to Noise Ratio (dB)")

        path = combo_dir / f"ber_{mod_name}_pfa_{PFA}.png"
        save(fig, path)
        paths.append(path)
    return paths


def plot_representative_pd_ber(
    ssca: dict[str, ParsedModulation],
    ber_curves: dict[str, dict[str, np.ndarray]],
    save_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharey=True)
    for ax, detector, title in zip(
        axes, DETECTORS, ["Radiometer", "Max-Cut", "DCS"]
    ):
        for mod_name, label in REPRESENTATIVE_MODS:
            ber, pd = pd_ber_curve(mod_name, ssca[mod_name], detector, ber_curves)
            color, linestyle = STYLE_BY_MOD[mod_name]
            ax.plot(
                ber,
                pd,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=MARKERS_BY_MOD[mod_name],
                markevery=7,
                markersize=3.5,
                markerfacecolor="white",
                markeredgewidth=0.8,
                linewidth=1.7,
            )
        ax.set_xscale("log")
        ax.set_xlim(5e-1, 1e-5)
        ax.set_ylim(0, 1.03)
        ax.set_title(title)
        ax.set_xlabel("BER")
    axes[0].set_ylabel(r"$P_D$")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    path = save_dir / "papereto_representative_pd_ber.png"
    save(fig, path)
    return path


def plot_heatmap(
    ssca: dict[str, ParsedModulation],
    ber_curves: dict[str, dict[str, np.ndarray]],
    save_dir: Path,
) -> Path:
    values = np.zeros((len(SUMMARY_MODS), len(DETECTORS)))
    labels = [["" for _ in DETECTORS] for _ in SUMMARY_MODS]
    for i, (mod_name, _) in enumerate(SUMMARY_MODS):
        for j, detector in enumerate(DETECTORS):
            value, reached = interpolate_pd_on_ber(
                mod_name, ssca[mod_name], detector, ber_curves
            )
            values[i, j] = value
            labels[i][j] = f"{value:.2f}" if reached else "NR"

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(DETECTORS)), ["Radiometer", "Max-Cut", "DCS"])
    ax.set_yticks(np.arange(len(SUMMARY_MODS)), [label for _, label in SUMMARY_MODS])
    ax.set_title(r"$P_D$ at BER $=10^{-2}$")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > 0.55 else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$P_D$ (lower is better for Alice)")
    ax.text(
        0,
        len(SUMMARY_MODS) + 0.15,
        "NR: BER target not reached in simulated SNR range",
        fontsize=8,
        ha="left",
        va="top",
        transform=ax.transData,
    )
    path = save_dir / "papereto_pd_at_ber_heatmap.png"
    save(fig, path)
    return path


def plot_transition_snr(
    ssca: dict[str, ParsedModulation],
    fam: dict[str, ParsedModulation],
    save_dir: Path,
) -> Path:
    rows = SUMMARY_MODS
    y = np.arange(len(rows))
    marker_specs = {
        "Energy": ("Radiometer", "o", "#222222"),
        "MaxCut": ("Max-Cut", "s", "#d95f02"),
        "Dcs": ("DCS", "^", "#1b9e77"),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    energy_baseline = transition_snr(ssca["BPSK"], "Energy")
    ax.axvline(energy_baseline, color="#222222", linestyle="--", linewidth=1.2, alpha=0.65)
    for detector, (label, marker, color) in marker_specs.items():
        xs = []
        for mod_name, _ in rows:
            source = fam if detector == "Dcs" else ssca
            xs.append(transition_snr(source[mod_name], detector))
        ax.scatter(xs, y, label=label, marker=marker, color=color, s=42, zorder=3)
    ax.set_yticks(y, [label for _, label in rows])
    ax.invert_yaxis()
    ax.set_xlabel(r"SNR where $P_D=0.5$ (dB)")
    ax.set_title("Detection transition SNR")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.35)
    path = save_dir / "papereto_transition_snr.png"
    save(fig, path)
    return path


def branch_curve(
    mods: list[str], parsed: dict[str, ParsedModulation], detector: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snrs_db = parsed[mods[0]].snrs_db
    curves = np.vstack([parsed[name].pds[detector] for name in mods])
    return snrs_db, curves.mean(axis=0), curves.min(axis=0), curves.max(axis=0)


def plot_detector_behavior(
    ssca: dict[str, ParsedModulation],
    fam: dict[str, ParsedModulation],
    save_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), sharey=True)

    ax = axes[0]
    baseline = ssca["BPSK"]
    ax.plot(
        baseline.snrs_db,
        baseline.pds["Energy"],
        color="black",
        linestyle="--",
        linewidth=1.3,
        label="Radiometer",
    )
    for mod_name, label in MAXCUT_PD_SNR_MODS:
        color, linestyle = STYLE_BY_MOD[mod_name]
        ax.plot(
            ssca[mod_name].snrs_db,
            ssca[mod_name].pds["MaxCut"],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=MARKERS_BY_MOD[mod_name],
            markevery=8,
            markersize=3.6,
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.5,
        )
    ax.set_title("Max-Cut distinct behavior")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"$P_D$")
    ax.set_xlim(-25, 0)
    ax.set_ylim(0, 1.03)

    ax = axes[1]
    ax.plot(
        baseline.snrs_db,
        baseline.pds["Energy"],
        color="black",
        linestyle="--",
        linewidth=1.3,
        label="Radiometer",
    )
    for mods, label, color in [
        (DCS_HIGH_BRANCH, "High-detectability group", "#D55E00"),
        (DCS_LOW_BRANCH, "Low-detectability group", "#0072B2"),
    ]:
        snrs_db, mean, lower, upper = branch_curve(mods, fam, "Dcs")
        ax.fill_between(snrs_db, lower, upper, color=color, alpha=0.16, linewidth=0)
        ax.plot(snrs_db, mean, color=color, linewidth=1.8, label=label)
    ax.set_title("DCS group behavior")
    ax.set_xlabel("SNR (dB)")
    ax.set_xlim(-25, 0)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right", fontsize=8)

    h, l = axes[0].get_legend_handles_labels()
    axes[1].legend(h, l, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    path = save_dir / "papereto_detector_behavior.png"
    save(fig, path)
    return path


def copy_to_paper(paths: list[Path], paper_dir: Path | None) -> None:
    if paper_dir is None:
        return
    paper_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = paper_dir / path.name
        target.write_bytes(path.read_bytes())
        print(f"Copied {target}")


def main() -> None:
    args = parse_args()
    common_style()

    if args.combo_only:
        wanted = {mod_name for mod_name, _ in COMBO_BER_PD_MODS}
        ber_curves = load_ber_curves(args.ber_file, wanted)
        ssca = load_selected_cfar(args.ssca_results, wanted)
        args.save_dir.mkdir(parents=True, exist_ok=True)
        combo_paths = plot_combo_ber_pd(ssca, ber_curves, args.save_dir)
        copy_to_paper(combo_paths, args.paper_cfar_dir)
        print("Generated:")
        for path in combo_paths:
            print(f"  {path}")
        return

    if args.detector_behavior_only:
        wanted = (
            {mod_name for mod_name, _ in MAXCUT_PD_SNR_MODS}
            | set(DCS_HIGH_BRANCH)
            | set(DCS_LOW_BRANCH)
            | {"BPSK"}
        )
        ssca = load_selected_cfar(args.ssca_results, wanted)
        fam = load_selected_cfar(args.fam_results, wanted)
        args.save_dir.mkdir(parents=True, exist_ok=True)
        path = plot_detector_behavior(ssca, fam, args.save_dir)
        copy_to_paper([path], args.paper_dir)
        print("Generated:")
        print(f"  {path}")
        return

    wanted = set(ALL_MODS)
    ber_curves = load_ber_curves(args.ber_file, wanted)
    ssca = load_selected_cfar(args.ssca_results, wanted)
    fam = load_selected_cfar(args.fam_results, wanted)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_representative_pd_ber(ssca, ber_curves, args.save_dir),
        plot_heatmap(ssca, ber_curves, args.save_dir),
        plot_transition_snr(ssca, fam, args.save_dir),
        plot_detector_behavior(ssca, fam, args.save_dir),
    ]
    combo_paths = plot_combo_ber_pd(ssca, ber_curves, args.save_dir)
    copy_to_paper(paths, args.paper_dir)
    copy_to_paper(combo_paths, args.paper_cfar_dir)
    print("Generated:")
    for path in paths + combo_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
