#!/usr/bin/env python
from cfar import get_threshold, parse_results
from foo import (
    BER_YLIM,
    DETECTORS,
    FIG_SIZE,
    NCOLS,
)
from goo import (
    # GROUP_1,
    # GROUP_2,
    # GROUP_3,
    GROUPS,
    GROUP_MARKERS
)
from util import db, timeit

from argparse import ArgumentParser, Namespace
import concurrent.futures
import gc
from functools import partial
import os
from pathlib import Path
import psutil
from typing import Dict, List


def plot_cfar_with_multiple_modulations(
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
            pfa, pds = next(iter(modulation[kind]["pfas"].items()))
        except KeyError:
            print(f"{kind} detector not found.")
            return
        ax.plot(db(snrs), pds, label=modulation["name"], linestyle="dotted")

    ax.legend(loc="best", ncols=NCOLS)
    if save:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(
            save_dir / f"cfar_{kind}_pfa_{pfa}_group_{group_id}.png",
            bbox_inches="tight",
        )
    fig.suptitle(kind)


def plot_pd_vs_ber_metric(
    modulation_test_results: List[Dict[str, object]],
    bers: List[List[Dict[str, object]]],
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
            pfa, pds = next(iter(modulation[kind]["pfas"].items()))
            x = pds
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


# TODO: FIXME: BUG: XXX: Fix this function!
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
        λ0s = []
        for i, snr in enumerate(modulation["snrs"]):
            try:
                pfa = next(iter(modulation[kind]["pfas"]))
                λ0 = get_threshold(pfa, modulation[kind]["h0_λs"][i])
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


def plot_all_bers(
    bers: List[Dict[str, object]],
    save=False,
    save_dir=Path("/tmp/"),
    group_id: int = 0,
    ebn0: bool = False,
):
    fig, ax = plt.subplots(1)
    if ebn0:
        ax.set_xlabel(r"$\frac{E_b}{N_0}$ (dB)")
    else:
        ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.grid(True, which="both")
    ax.set_prop_cycle(GROUP_MARKERS[group_id])
    for ber in bers:
        ax.plot(db(ber["snrs"]), ber["bers"], label=ber["name"])
    ax.legend(loc="best", ncols=NCOLS)
    ax.set_yscale("log")
    ax.set_ylim(BER_YLIM)
    # ax.set_xlim([min(db(bers[0]["snrs"])), max(db(bers[0]["snrs"]))])
    ax.set_xlim([-20, 20])

    if save:
        fig.set_size_inches(*FIG_SIZE)
        if ebn0:
            fig.savefig(save_dir / f"bers_ebn0_group_{group_id}", bbox_inches="tight")
        else:
            fig.savefig(save_dir / f"bers_snr_group_{group_id}", bbox_inches="tight")
    if ebn0:
        ax.set_title(r"BER vs $\frac{E_b}{N_0}$ (All Modulations)")
    else:
        ax.set_title("BER vs SNR (All Modulations)")


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument(
        "-b", "--ber-file", default=CWD.parent / "bers_curr.json", type=Path
    )
    ap.add_argument(
        "-p", "--pd-file", default=CWD.parent / "results_curr.json", type=Path
    )
    ap.add_argument("-f", "--pfa", default=0.01, type=float)
    ap.add_argument("-s", "--save", action="store_true")
    ap.add_argument("-d", "--save-dir", type=Path, default=Path("/tmp/"))
    ap.add_argument("-g", "--group", type=int, choices=(1, 2, 3), default=None)
    ap.add_argument("--bers-only", action="store_true")
    ap.add_argument("--ebn0", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt

    CWD: Path = Path(__file__).parent

    args = parse_args()

    if args.group:
        group_ids = [args.group]
    else:
        group_ids = [1, 2, 3]

    with timeit("Loading Data") as _:
        if not args.bers_only:
            results_file_size = args.pd_file.stat().st_size
            with Path(args.pd_file).open("r") as f:
                results = json.load(f)
            grouped_results: List[List[Dict]] = [
                [r for r in results if r["name"] in GROUPS[group_id]]
                for group_id in group_ids
            ]
            del results
            gc.collect()

        with Path(args.ber_file).open("r") as f:
            bers = json.load(f)
        grouped_bers: List[List[Dict]] = [
            [b for b in bers if b["name"] in GROUPS[group_id]] for group_id in group_ids
        ]

        del bers
        gc.collect()

    if not args.bers_only:
        # Parse and interpret results
        parse = partial(parse_results, pfas=[args.pfa])

        num_cpus: int = os.cpu_count()
        ram: int = psutil.virtual_memory().available
        num_workers = min(ram // results_file_size, num_cpus)

        with timeit("CFAR Calculation") as _:
            if num_workers > 1:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers
                ) as p:
                    grouped_regress: List[List[Dict[str, object]]] = [
                        list(p.map(parse, group)) for group in grouped_results
                    ]
            else:
                grouped_regress: List[List[Dict[str, object]]] = [
                    list(map(parse, group)) for group in grouped_results
                ]
        del grouped_results
        gc.collect()

        DETECTORS = [
            k for k in grouped_regress[0][0].keys() if k not in ("name", "snrs")
        ]

        with timeit("Plotting") as _:

            for group_id, regressed, bers in zip(
                group_ids, grouped_regress, grouped_bers
            ):

                for detector in DETECTORS:
                    if detector == "Energy":
                        continue
                    plot_cfar_with_multiple_modulations(
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

    for group_id, bers in zip(group_ids, grouped_bers):
        plot_all_bers(
            bers,
            save=args.save,
            save_dir=args.save_dir,
            group_id=group_id,
            ebn0=args.ebn0,
        )

    if not args.save:
        plt.show()
