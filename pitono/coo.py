#!/usr/bin/env python
from util import db, timeit
from cfar import parse_results
from foo import filter_results, FIG_SIZE, NCOLS, BER_YLIM, DETECTORS, MODULATIONS

from argparse import ArgumentParser, Namespace
import concurrent.futures
from cycler import cycler
import gc
from functools import partial
import numpy as np
import os
from pathlib import Path
import psutil
import re
from typing import Dict, List


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


def plot_cfar_with_multiple_modulations(
    modulation_test_results: List[Dict[str, object]],
    kind: str,
    save=False,
    save_dir=Path("/tmp/"),
    title: str = "",
):
    fig, ax = plt.subplots()
    ax.grid(True, which="both")
    ax.set_xlabel("SNR (db)")
    ax.set_ylabel(r"Probability of Detection ($\mathbb{P}_D$)")
    ax.set_prop_cycle(get_cycles(len(modulation_test_results)))

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
        ax.plot(db(snrs), pds, label=modulation["name"])

    ax.legend(loc="best", ncols=NCOLS)
    if save:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(
            save_dir / f"cfar_{kind}_pfa_{pfa}.png",
            bbox_inches="tight",
        )
    if title:
        fig.suptitle(f"{kind} - {title}")
    else:
        fig.suptitle(kind)


def plot_pd_vs_ber(
    modulation: Dict[str, object],
    bers: List[Dict[str, object]],
    save=False,
    save_dir=Path("/tmp/"),
):
    try:
        mod_ber = next(b for b in bers if b["name"] == modulation["name"])
    except (TypeError, StopIteration):
        print(f"BER for {modulation["name"]} not found.")
        return
    fig, ax = plt.subplots()
    ax.set_xlim(db([modulation["snrs"]]).min(), db([modulation["snrs"]]).max())
    ber_ax = ax
    pd_ax = ax.twinx()
    # pd_ax.grid(True, which='both')
    ber_ax.grid(True, which="both")
    ber_ax.plot(db(mod_ber["snrs"]), mod_ber["bers"], color="Red")
    ber_ax.set_yscale("log")
    ber_ax.set_ylim(BER_YLIM)
    ber_ax.tick_params(axis="y", colors="Red")
    ber_ax.set_ylabel("Bit Error Rate (BER)", color="Red")

    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    for detector, style in zip(DETECTORS, linestyles):
        try:
            pfa, pds = next(iter(modulation[detector]["pfas"].items()))
            pd_ax.plot(
                db(modulation["snrs"]),
                pds,
                color="Blue",
                linestyle=style,
                label=detector,
            )
        except KeyError:
            continue
    pd_ax.set_ylim([0, 1.025])
    # pd_ax.plot(-12.0556, good_pd, "bo")
    # pd_ax.axhline(good_pd, color='Blue', ls='--', label=f'Acceptable ℙd ({good_pd})')
    pd_ax.tick_params(axis="y", colors="Blue")
    pd_ax.set_ylabel(r"Probability of Detection ($\mathbb{P}_D$)", color="Blue")
    pd_ax.legend(loc=6)
    ax.set_xlabel("Signal to Noise Ratio dB (SNR dB)")
    if save:
        fig.set_size_inches(*FIG_SIZE)
        fig.savefig(
            save_dir / f'ber_cfar_{modulation["name"]}_pfa_{pfa}.png',
            bbox_inches="tight",
        )
    ax.set_title(modulation["name"])


def plot_pd_vs_ber_metric(
    modulation_test_results: List[Dict[str, object]],
    bers: List[Dict[str, object]],
    kind: str,
    save=False,
    save_dir=Path("/tmp/"),
):
    fig, ax = plt.subplots()
    ax.grid(True, which="both")
    ax.set_prop_cycle(get_cycles(len(modulation_test_results)))
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
            save_dir / f"covert_metric_cfar_{kind}_pfa_{pfa}.png", bbox_inches="tight"
        )
    ax.set_title(f"{kind} Detector" + r" - BER vs $\mathbb{P}_D$")


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument(
        "-b", "--ber-file", default=CWD.parent / "bers_curr.json", type=Path
    )
    ap.add_argument(
        "-p", "--pd-file", default=CWD.parent / "results_curr.json", type=Path
    )
    ap.add_argument("-f", "--pfa", default=0.01, type=float)
    ap.add_argument("-r", "--regex", default="", type=str)
    ap.add_argument("-s", "--save", action="store_true")
    ap.add_argument("-d", "--save-dir", type=Path, default=Path("/tmp/"))
    ap.add_argument("--ebn0", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt

    CWD: Path = Path(__file__).parent

    args = parse_args()

    regex = re.compile(args.regex)

    with timeit("Loading Data") as _:
        # Load from JSON.
        results_file_size = args.pd_file.stat().st_size
        with Path(args.pd_file).open("r") as f:
            results = json.load(f)
        results = filter_results(results, regex)

        gc.collect()

        with Path(args.ber_file).open("r") as f:
            bers = json.load(f)
        bers = filter_results(bers, regex)

        gc.collect()

    # Parse and Log Regress results.
    parse = partial(parse_results, pfas=[args.pfa])

    num_cpus: int = os.cpu_count()
    ram: int = psutil.virtual_memory().available
    num_workers = min(ram // results_file_size, num_cpus)

    with timeit("Logistic Regresstion") as _:
        if num_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as p:
                regressed: List[Dict[str, object]] = list(p.map(parse, results))
        else:
            regressed: List[Dict[str, object]] = list(map(parse, results))
    del results
    gc.collect()

    with timeit("Plotting") as _:

        for detector in DETECTORS:
            plot_cfar_with_multiple_modulations(
                regressed,
                detector,
                save=args.save,
                save_dir=args.save_dir,
                title=args.pd_file.name,
            )

        for modulation in regressed:
            plot_pd_vs_ber(modulation, bers, save=args.save, save_dir=args.save_dir)

        for detector in DETECTORS:
            plot_pd_vs_ber_metric(
                regressed, bers, detector, save=args.save, save_dir=args.save_dir
            )

    if not args.save:
        plt.show()
