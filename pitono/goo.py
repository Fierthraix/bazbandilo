#!/usr/bin/env python
from foo import (
    parse_results,
    FIG_SIZE,
    NCOLS,
)
from util import db, timeit

from argparse import ArgumentParser, Namespace
import concurrent.futures
from cycler import cycler
import gc
from functools import partial
import os
from pathlib import Path
import psutil
from typing import Dict, List


def plot_youden_j_with_multiple_modulations(
    modulation_test_results: List[Dict[str, object]],
    kind: str,
    save=False,
    save_dir=Path("/tmp/"),
    group_id: int = 0,
):
    fig, ax = plt.subplots()
    ax.grid(True, which="both")
    ax.set_xlabel("SNR (db)")
    ax.set_ylabel(r"Probability of Detection ($\mathbb{P}_D$)")
    ax.set_prop_cycle(GROUP_MARKERS[group_id])

    snrs_db = db(modulation_test_results[0]["snrs"])
    ax.set_xlim(snrs_db.min(), snrs_db.max())
    ax.set_ylim([0, 1.025])

    for modulation in modulation_test_results:
        snrs = modulation["snrs"]
        try:
            youden_js: List[float] = modulation[kind]["youden_js"]
        except KeyError:
            print(f"{kind} detector not found.")
            return
        ax.plot(db(snrs), youden_js, label=modulation["name"], linestyle="dotted")

    ax.legend(loc="best", ncols=NCOLS)
    if save:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(
            save_dir / f"Youden-J_{kind}_multiple_modulations_group_{group_id}.png",
            bbox_inches="tight",
        )
    fig.suptitle(kind)


def plot_pd_vs_ber_metric(
    modulation_test_results: List[Dict[str, object]],
    bers: List[Dict[str, object]],
    kind: str,
    save=False,
    save_dir=Path("/tmp/"),
    group_id: int = 0,
):
    fig, ax = plt.subplots()
    ax.grid(True, which="both")
    ax.set_prop_cycle(GROUP_MARKERS[group_id])
    ax.set_xlabel(r"Probability of Detection ($\mathbb{P}_D$)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_yscale("log")
    ax.set_xlim([0, 1])
    ax.set_ylim([1e-2, 0.5])
    for modulation in modulation_test_results:
        try:
            mod_ber = next(b for b in bers if b["name"] == modulation["name"])
        except TypeError:
            return
        except StopIteration:
            print(f'{modulation["name"]} BER not found.')
            return

        try:
            x = modulation[kind]["youden_js"]
        except KeyError:
            print(f"{kind} detector not found")
            return
        y = mod_ber["bers"]
        r = min(len(x), len(y))
        ax.plot(x[:r], y[:r], label=modulation["name"])
    ax.set_xlabel(r"Probability of Detection ($\mathbb{P}_D$)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.legend(loc="best", ncols=NCOLS)

    if save:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(
            save_dir / f"covert_metric_{kind}_group_{group_id}.png", bbox_inches="tight"
        )
    ax.set_title(f"{kind} Detector" + r" - BER vs $\mathbb{P}_D$")


def plot_pd_vs_pfa(
    results_object: List[Dict[str, object]],
    kind: str,
    save=False,
    save_dir=Path("/tmp/"),
    group_id: int = 0,
):
    fig, ax = plt.subplots()
    ax.grid(True, which="both")
    ax.set_prop_cycle(GROUP_MARKERS[group_id])
    ax.set_xlabel(r"Probability of False Alarm ($\mathbb{P}_{FA}$)")
    ax.set_ylabel(r"Probability of Detection ($\mathbb{P}_D$)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    for modulation in results_object:
        try:
            p = modulation[kind]["df"]
        except KeyError:
            print(f"{kind} detector not found.")
            return
        mid = len(p) // 2
        x = p[mid]["tpr"]
        y = p[mid]["fpr"]
        snr_db = db(modulation["snrs"][mid])
        ax.plot(x, y, label=f"{modulation["name"]}")
    ax.legend(loc="best", ncols=NCOLS)

    if save:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(
            save_dir / f"pd_vs_pfa_{kind}_{snr_db}dB_group_{group_id}.png",
            bbox_inches="tight",
        )
    ax.set_title(
        f"{kind}"
        + r"Detector - $\mathbb{P}_D$ vs $\mathbb{{P}}_{FA}$ - "
        + f"SNR={snr_db:.2f}"
    )


def plot_λ_vs_snr(
    results_object: List[Dict[str, object]],
    kind: str,
    save: bool = False,
    save_dir=Path("/tmp/"),
    group_id: int = 0,
):
    fig, ax = plt.subplots()
    ax.grid(True, which="both")
    ax.set_prop_cycle(GROUP_MARKERS[group_id])
    for modulation in results_object:
        try:
            p = modulation[kind]["df"]
        except KeyError:
            continue
        λ0s = []
        for i, snr in enumerate(modulation["snrs"]):
            try:
                λ0 = (
                    p[i]
                    .sort_values(by="youden_j", ascending=False, ignore_index=True)
                    .iloc[0]
                    .x
                )
            except IndexError:
                λ0 = 0
            λ0s.append(λ0)

        ax.plot(db(modulation["snrs"]), λ0s, label=f"{modulation['name']}")
        ax.set_xlim(db(modulation["snrs"]).min(), db(modulation["snrs"]).max())
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Threshold λ")
    ax.legend(loc="best", ncols=NCOLS)

    if save:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(
            save_dir / f"lambda_{kind}_group_{group_id}.png", bbox_inches="tight"
        )
    ax.set_title(f"Threshold vs SNR ({kind})")


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument(
        "-b", "--ber-file", default=CWD.parent / "bers_curr.json", type=Path
    )
    ap.add_argument(
        "-p", "--pd-file", default=CWD.parent / "results_curr.json", type=Path
    )
    ap.add_argument("-l", "--log-regressions", default=1, type=int)
    ap.add_argument("-s", "--save", action="store_true")
    ap.add_argument("-d", "--save-dir", type=Path, default=Path("/tmp/"))
    ap.add_argument("-g", "--group", type=int, choices=(1, 2, 3), default=None)
    ap.add_argument("--bers-only", action="store_true")
    return ap.parse_args()


GROUP_1: List[str] = [
    "BPSK",
    "QPSK",
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
    1: cycler(marker=["^", "v", "1", "2", "3", "4"])
    + cycler(color=["r", "r", "blue", "blue", "green", "orange"]),
    2: cycler(marker=["^", "v", "<", "1", "2", "3", "4"])
    + cycler(color=["g", "g", "g", "r", "b", "r", "b"]),
    3: cycler(marker=["^", "v", "1", "2", "3", "4"])
    + cycler(color=["b", "b", "r", "y", "orange", "g"]),
}


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt

    CWD: Path = Path(__file__).parent

    args = parse_args()

    # regex = re.compile(args.regex)
    if args.group:
        group_ids = [args.group]
    else:
        group_ids = [1, 2, 3]

    grouped_results: List[List[Dict]] = []

    with timeit("Loading Data") as _:
        results_file_size = args.pd_file.stat().st_size
        with Path(args.pd_file).open("r") as f:
            results = json.load(f)
        # results = filter_results(results, regex)
        grouped_results: List[List[Dict]] = [
            [r for r in results if r["name"] in GROUPS[group_id]]
            for group_id in group_ids
        ]
        del results
        gc.collect()

        with Path(args.ber_file).open("r") as f:
            bers = json.load(f)
        # bers = filter_results(bers, regex)

        gc.collect()

    # Parse and Log Regress results.
    parse = partial(parse_results, num_regressions=args.log_regressions)

    num_cpus: int = os.cpu_count()
    ram: int = psutil.virtual_memory().available
    num_workers = min(ram // results_file_size, num_cpus)

    with timeit("Logistic Regresstion") as _:
        if num_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as p:
                grouped_regress: List[List[Dict[str, object]]] = [
                    list(p.map(parse, group)) for group in grouped_results
                ]
        else:
            grouped_regress: List[List[Dict[str, object]]] = [
                list(map(parse, group)) for group in grouped_results
            ]
    del grouped_results
    gc.collect()

    DETECTORS = [k for k in grouped_regress[0][0].keys() if k not in ("name", "snrs")]

    with timeit("Plotting") as _:

        for group_id, regressed in zip(group_ids, grouped_regress):

            for detector in DETECTORS:
                if detector == "Energy":
                    continue
                plot_youden_j_with_multiple_modulations(
                    regressed,
                    detector,
                    save=args.save,
                    save_dir=args.save_dir,
                    group_id=group_id,
                )
            for detector in DETECTORS:
                plot_pd_vs_pfa(
                    regressed,
                    detector,
                    save=args.save,
                    save_dir=args.save_dir,
                    group_id=group_id,
                )

            for detector in DETECTORS:
                plot_λ_vs_snr(
                    regressed,
                    detector,
                    save=args.save,
                    save_dir=args.save_dir,
                    group_id=group_id,
                )

            for detector in DETECTORS:
                plot_pd_vs_ber_metric(
                    regressed,
                    bers,
                    detector,
                    save=args.save,
                    save_dir=args.save_dir,
                    group_id=group_id,
                )

    if not args.save:
        plt.show()
