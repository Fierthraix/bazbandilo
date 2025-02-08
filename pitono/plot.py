#!/usr/bin/env python
from util import db

from argparse import ArgumentParser
from cycler import cycler
import concurrent.futures
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import psutil
import re
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple

# FIG_SIZE = (16, 9)
FIG_SIZE: Tuple[float, float] = (12, 7)
# FIG_SIZE = (8, 4.5)

NCOLS: int = 2

BER_YLIM: List[float] = [1e-5, 0.55]

DETECTORS: List[str] = ["Energy", "MaxCut", "Dcs", "NormalTest"]

MODULATIONS = [
    "BPSK",
    "QPSK",
    "CDMA-BPSK-16",
    "CDMA-QPSK-16",
    "CDMA-QPSK-32",
    "CDMA-QPSK-64",
    "16QAM",
    "64QAM",
    "BFSK-16",
    "BFSK-32",
    "BFSK-64",
    "OFDM-BPSK-16",
    "OFDM-QPSK-16",
    "OFDM-BPSK-64",
    "OFDM-QPSK-64",
    "CSS-16",
    "CSS-64",
    "CSK",
    "DCSK",
    "QCSK",
    "FH-OFDM-DCSK",
]

GROUP_1: List[str] = [
    "BPSK",
    "QPSK",
    "16QAM",
    "64QAM",
    "CDMA-BPSK-16",
    "CDMA-QPSK-16",
    "CDMA-QPSK-32",
    "CDMA-QPSK-64",
]

GROUP_2: List[str] = [
    "BFSK-16",
    "BFSK-32",
    "BFSK-64",
    "OFDM-BPSK-16",
    "OFDM-QPSK-16",
    "OFDM-BPSK-64",
    "OFDM-QPSK-64",
]
GROUP_3: List[str] = [
    "CSS-16",
    "CSS-64",
    "CSK",
    "DCSK",
    "QCSK",
    "FH-OFDM-DCSK",
]

GROUPS: Dict[int, List[str]] = {
    1: GROUP_1,
    2: GROUP_2,
    3: GROUP_3,
}

GROUP_MARKERS: Dict[int, cycler] = {
    1: cycler(marker=["^", "v", "+", "x", "1", "2", "3", "4"])
    + cycler(color=["r", "r", "g", "g", "b", "b", "y", "orange"]),
    2: cycler(marker=["^", "v", "<", "1", "2", "3", "4"])
    + cycler(color=["g", "g", "g", "r", "b", "r", "b"]),
    3: cycler(marker=["^", "v", "1", "2", "3", "4"])
    + cycler(color=["b", "b", "r", "y", "orange", "g"]),
}


def get_cycles(num_lines: int) -> cycler:
    colours: List[str] = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    linestyles: List[str] = [
        "-",
        ":",
        "-.",
        "dashed",
        (0, (3, 1, 1, 1, 1, 1)),
        (0, (3, 5, 1, 5)),
    ]

    r: int = int(np.ceil(num_lines / len(colours)))

    return cycler(color=colours) * cycler(linestyle=linestyles[:r])


def plot_pd_with_multiple_modulations(
    pds: List[Tuple[str, List[float]]],
    snrs: List[float],
    cycles: Optional[cycler] = None,
    save_path: Optional[Path] = None,
):
    fig, ax = plt.subplots()
    ax.grid(True, which="both")
    ax.set_xlabel("SNR (db)")
    ax.set_ylabel(r"Probability of Detection ($\mathbb{P}_D$)")
    if cycles:
        ax.set_prop_cycle(cycles)

    snrs_db = db(snrs)
    ax.set_xlim(snrs_db.min(), snrs_db.max())
    ax.set_ylim([0, 1.025])

    for name, pd_curve in pds:
        ax.plot(db(snrs), pd_curve, label=name)

    ax.legend(loc="best", ncols=NCOLS)
    if save_path:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(save_path, bbox_inches="tight")
    ax.set_title(save_path.stem)


def plot_pd_vs_snr_cfar(
    modulation: Dict[str, object],
    kind: str,
    save=False,
    save_dir=Path("/tmp/"),
):
    """Plot $P_D$ versus SNR for multiple $P_{FA}$s."""
    fig, ax = plt.subplots()
    ax.grid(True, which="both")
    ax.set_xlabel("SNR (db)")
    ax.set_ylabel(r"Probability of Detection ($\mathbb{P}_D$)")
    # ax.set_prop_cycle(get_cycles(len(modulation_test_results)))
    snrs_db = db(modulation["snrs"])
    ax.set_xlim(snrs_db.min(), snrs_db.max())
    ax.set_ylim([0, 1.025])
    for pfa, pds in modulation[kind]["pfas"].items():
        ax.plot(snrs_db, pds, label=f"$P_{{FA}}={pfa}$")
    ax.legend(loc="best")
    if save:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(
            save_dir / f'cfar_pd_vs_snr_{kind}_{modulation["name"]}.png',
            bbox_inches="tight",
        )
    ax.set_title(f"{modulation["name"]} - {kind}")


def plot_pd_vs_ber(
    pds: List[Tuple[str, List[float]]],
    bers: List[float],
    snrs: List[float],
    save_path: Optional[Path] = None,
):
    fig, ax = plt.subplots()
    ax.set_xlim(db([snrs]).min(), db([snrs]).max())
    ber_ax = ax
    pd_ax = ax.twinx()
    ber_ax.grid(True, which="both")
    ber_ax.plot(db(snrs)[: len(bers)], bers, color="Red")
    ber_ax.set_yscale("log")
    ber_ax.set_ylim(BER_YLIM)
    ber_ax.tick_params(axis="y", colors="Red")
    ber_ax.set_ylabel("Bit Error Rate (BER)", color="Red")

    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    for (detector_name, pds), style in zip(pds, linestyles):
        try:
            pd_ax.plot(
                db(snrs),
                pds,
                color="Blue",
                linestyle=style,
                label=detector_name,
            )
        except KeyError:
            continue
    pd_ax.set_ylim([0, 1.025])
    pd_ax.tick_params(axis="y", colors="Blue")
    pd_ax.set_ylabel(r"Probability of Detection ($\mathbb{P}_D$)", color="Blue")
    pd_ax.legend(loc=6)
    ax.set_xlabel("Signal to Noise Ratio dB (SNR dB)")
    if save_path:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(save_path, bbox_inches="tight")
    ax.set_title(save_path.stem)


def plot_bers(
    bers: List[Dict[str, object]],
    ebn0: bool = False,
    cycles: Optional[cycler] = None,
    save_path: Optional[Path] = None,
):
    fig, ax = plt.subplots(1)
    if ebn0:
        ax.set_xlabel(r"$\frac{E_b}{N_0}$ (dB)")
    else:
        ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.grid(True, which="both")
    if cycles:
        ax.set_prop_cycle(cycles)
    for ber in bers:
        ax.plot(db(ber["snrs"]), ber["bers"], label=ber["name"])
    ax.legend(loc="best", ncols=NCOLS)
    ax.set_yscale("log")
    ax.set_ylim(BER_YLIM)
    ax.set_xlim([-20, 20])

    if save_path:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(save_path, bbox_inches="tight")
    if ebn0:
        ax.set_title(r"BER vs $\frac{E_b}{N_0}$" + f"{save_path.stem}")
    else:
        ax.set_title(f"BER vs SNR {save_path.stem}")


def plot_pd_vs_ber_metric(
    data: List[Tuple[str, List[float], List[float]]],
    cycles: Optional[cycler] = None,
    save_path: Optional[Path] = None,
):
    fig, ax = plt.subplots()
    ax.grid(True, which="both")
    if cycles:
        ax.set_prop_cycle(cycles)
    ax.set_xlabel(r"Probability of Detection ($\mathbb{P}_D$)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_yscale("log")
    ax.set_xlim([0, 1])
    ax.set_ylim([1e-2, 0.5])
    for name, pds, bers in data:
        r = min(len(pds), len(bers))
        ax.plot(pds[:r], bers[:r], label=name)
    ax.set_xlabel(r"Probability of Detection ($\mathbb{P}_D$)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.legend(loc="best", ncols=NCOLS)

    if save_path:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(save_path, bbox_inches="tight")
    ax.set_title(r" - BER vs $\mathbb{P}_D$" + save_path.stem)


def plot_pd_vs_pfa(
    data: List[Tuple[str, List[float], List[float]]],
    cycles: Optional[cycler] = None,
    save_path: Optional[Path] = None,
):
    fig, ax = plt.subplots()
    ax.grid(True, which="both")
    ax.set_prop_cycle(cycles)
    ax.set_xlabel(r"Probability of False Alarm ($\mathbb{P}_{FA}$)")
    ax.set_ylabel(r"Probability of Detection ($\mathbb{P}_D$)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    for mod_name, pds, pfas in data:
        ax.plot(pds, pfas, label=mod_name)
    ax.legend(loc="best", ncols=NCOLS)

    if save_path:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(save_path, bbox_inches="tight")
    ax.set_title(
        r"Detector - $\mathbb{P}_D$ vs $\mathbb{{P}}_{FA}$ - " + save_path.stem
    )


def plot_λ_vs_snr(
    λs: List[Tuple[str, List[float]]],
    snrs: List[float],
    cycles: Optional[cycler] = None,
    save_path: Optional[Path] = None,
):
    fig, ax = plt.subplots()
    ax.grid(True, which="both")
    if cycles:
        ax.set_prop_cycle(cycles)
    for name, λs in λs:
        ax.plot(db(snrs), λs, label=name)
        ax.set_xlim(db(snrs).min(), db(snrs).max())
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Threshold λ")
    ax.legend(loc="best", ncols=NCOLS)

    if save_path:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(save_path, bbox_inches="tight")
    ax.set_title(f"Threshold vs SNR ({save_path.stem})")


def filter_results(
    results_object: List[Dict[str, object]], pattern: re.Pattern
) -> List[Dict[str, object]]:
    return list(filter(lambda mod: pattern.match(mod["name"]), results_object))


def get_num_workers() -> int:
    ram_available: int = psutil.virtual_memory().available
    ram_used: int = psutil.Process(os.getpid()).memory_info().rss
    num_cpus: int = os.cpu_count()
    return min(ram_available // ram_used, num_cpus)


def multi_parse(
    results: List[Dict[str, object]],
    parse_fn: Callable[[List[Dict[str, object]]], List[Dict[str, object]]],
) -> List[Dict[str, object]]:

    num_workers: int = get_num_workers()
    if num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as p:
            return list(tqdm(p.map(parse_fn, results), total=len(results)))
    else:
        return list(tqdm(map(parse_fn, results), total=len(results)))


def multi_parse_grouped(
    results: List[List[Dict[str, object]]],
    parse_fn: Callable[[List[List[Dict[str, object]]]], List[List[Dict[str, object]]]],
) -> List[List[Dict[str, object]]]:

    num_workers: int = get_num_workers()
    if num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as p:
            return [
                list(tqdm(p.map(parse_fn, results_group), total=len(results_group)))
                for results_group in results
            ]
    else:
        return [
            list(tqdm(map(parse_fn, results_group), total=len(results_group)))
            for results_group in results
        ]


def base_parser() -> ArgumentParser:
    cwd: Path = Path(__file__).parent
    ap = ArgumentParser()
    ap.add_argument(
        "-b", "--ber-file", default=cwd.parent / "bers_curr.json", type=Path
    )
    ap.add_argument(
        "-p", "--pd-file", default=cwd.parent / "results_curr.json", type=Path
    )
    ap.add_argument("-s", "--save", action="store_true")
    ap.add_argument("-d", "--save-dir", type=Path, default=Path("/tmp/"))
    return ap


def load_json(ber_file: Path, filter: Optional[re.Pattern] = None) -> List[Dict]:
    with Path(ber_file).open("r") as f:
        bers = json.load(f)
    if filter:
        return filter_results(bers, filter)
    return bers
