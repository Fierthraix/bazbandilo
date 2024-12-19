#!/usr/bin/env python
from foo import filter_results, parse_results
from util import db, timeit, undb

from argparse import ArgumentParser, Namespace
import concurrent.futures
from functools import partial
import gc
import numpy as np
import re
from scipy.stats import rv_histogram
from typing import Dict, List


def get_closest_index(vec: List[object], index: object) -> int:
    error = np.abs(np.array(vec) - index)
    return list(error).index(min(error))


def plot_specific_snrs(
    modulation: List[Dict[str, object]],
    snrs: List[float],
    n: int = 16,
    save: bool = False,
):
    n = len(snrs)
    m = int(np.sqrt(n))
    assert m**2 == n

    fig, axs = plt.subplots(m, m)

    binification = 128

    indices: List[int] = [get_closest_index(modulation["snrs"], snr) for snr in snrs]

    for (i, idx), snr in zip(enumerate(indices), snrs):
        h0s = modulation["Energy"]["h0_λs"][idx]
        h1s = modulation["Energy"]["h1_λs"][idx]

        xmin = min(h0s + h1s)
        xmax = max(h0s + h1s)
        x = np.linspace(xmin, xmax, binification)
        # x = np.linspace(35, 100, binification)
        h0_pdf = rv_histogram(np.histogram(h0s, bins=binification)).pdf(x)
        h1_pdf = rv_histogram(np.histogram(h1s, bins=binification)).pdf(x)

        ax = axs[i // m, i % m]
        ax.plot(x, h0_pdf, label="$H_0$ case")
        ax.plot(x, h1_pdf, label="$H_1$ case")

        df = modulation["Energy"]["df"][idx]
        cut_off = (
            df.sort_values(by="youden_j", ascending=False, ignore_index=True).iloc[0].x
        )
        ax.axvline(cut_off, color="k", ls="--")

        ax.set_title(f"SNR={int(db(snr))}dB")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\mathbb{P}(\lambda)$")
        ax.legend(loc="best")
        ax.grid(True)
    plt.tight_layout()

    if save:
        fig.set_size_inches(16, 9)
        fig.savefig(f"/tmp/pdfs_energy_{modulation["name"]}.png")


def plot_some_pdfs(
    modulation: List[Dict[str, object]], n: int = 16, save: bool = False
):
    m = int(np.sqrt(n))
    assert m**2 == n

    fig, axs = plt.subplots(m, m)

    binification = 128
    # Calculate the spacing index
    step = len(modulation["snrs"]) / n
    indices = [int(i * step) for i in range(n)]

    for i, idx in enumerate(indices):
        h0s = modulation["Energy"]["h0_λs"][idx]
        h1s = modulation["Energy"]["h1_λs"][idx]

        h0_mean = np.average(h0s)
        h1_mean = np.average(h1s)
        # import pdb; pdb.set_trace()
        mean_point = np.average([h0_mean, h1_mean])
        xmin = min(h0s + h1s)
        xmax = max(h0s + h1s)
        x = np.linspace(xmin, xmax, binification)
        # x = np.linspace(35, 100, binification)
        h0_pdf = rv_histogram(np.histogram(h0s, bins=binification)).pdf(x)
        h1_pdf = rv_histogram(np.histogram(h1s, bins=binification)).pdf(x)

        ax = axs[i // m, i % m]
        ax.plot(x, h0_pdf, label="$H_0$ case")
        ax.plot(x, h1_pdf, label="$H_1$ case")

        df = modulation["Energy"]["df"][idx]
        cut_off = (
            df.sort_values(by="youden_j", ascending=False, ignore_index=True).iloc[0].x
        )
        ax.axvline(cut_off, color="k", ls="--")
        ax.axvline(mean_point, color="r", ls="-.")

        snr = modulation["snrs"][idx]
        ax.set_title(f"SNR={db(snr)}dB")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\mathbb{P}(\lambda)$")
        ax.legend(loc="best")
        ax.grid(True)
    fig.suptitle(f"{modulation["name"]} - PDF of $H_0$ and $H_1$ cases")
    plt.tight_layout()

    if save:
        fig.set_size_inches(16, 9)
        fig.savefig(f"/tmp/pdfs_energy_some_snrs_{modulation["name"]}.png")


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument("-p", "--pd-file", default=CWD.parent / "results_curr.json")
    ap.add_argument("-l", "--log-regressions", default=1, type=int)
    ap.add_argument(
        "-n",
        "--num-plots",
        default=16,
        type=int,
        # type=lambda x: (
        #     int(x)
        #     if x == int(np.sqrt(int(x))) ** 2
        #     else ArgumentTypeError(f"{x} is not a square number")
        # ),
    )
    ap.add_argument("-r", "--regex", default="")
    ap.add_argument("-s", "--save", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    import json

    import matplotlib.pyplot as plt
    from pathlib import Path

    CWD: Path = Path(__file__).parent

    args = parse_args()

    assert (
        int(np.sqrt(args.num_plots)) ** 2 == args.num_plots
    ), f"{args.num_plots} needs to be a square number"

    regex = re.compile(args.regex)

    with timeit("Loading Data") as _:
        # Load from JSON.
        with Path(args.pd_file).open("r") as f:
            results = json.load(f)
            # results = umsgpack.load(f, raw=True)
        results = filter_results(results, regex)

    gc.collect()

    parse = partial(parse_results, num_regressions=args.log_regressions)
    # regressed: List[Dict[str, object]] = list(map(parse, results))
    with concurrent.futures.ProcessPoolExecutor() as p, timeit(
        "Logistic Regresstion"
    ) as _:
        regressed: List[Dict[str, object]] = list(p.map(parse, results))

    del results
    gc.collect()

    with timeit("Plotting") as _:
        for modulation in regressed:
            plot_some_pdfs(modulation, n=args.num_plots, save=args.save)
            # plot_specific_snrs(modulation, snrs=undb(np.array([6, -6, -18, -30])))

    gc.collect()
    if not args.save:
        plt.show()
