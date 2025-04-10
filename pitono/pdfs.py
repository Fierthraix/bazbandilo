#!/usr/bin/env python
from cfar import parse_results
from plot import FIG_SIZE, base_parser, load_json, multi_parse, save_figure
from util import db, timeit, undb

from argparse import Namespace
from functools import partial
import numpy as np
import re
from pathlib import Path
from scipy.stats import rv_histogram
from typing import Dict, List, Optional


def get_closest_index(vec: List[object], index: object) -> int:
    error = np.abs(np.array(vec) - index)
    return list(error).index(min(error))


def plot_specific_snrs(
    modulation: List[Dict[str, object]],
    snrs: List[float],
    pfa: float,
    n: int = 16,
    save_path: Optional[Path] = None,
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

        cut_off = modulation["Energy"]["λ0s"][pfa][idx]
        ax.axvline(cut_off, color="k", ls="--")

        ax.set_title(f"SNR={int(db(snr))}dB")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\mathbb{P}(\lambda)$")
        ax.legend(loc="best")
        ax.grid(True)
    plt.tight_layout()

    save_figure(fig, save_path)
    ax.set_title(save_path.stem)


def plot_some_pdfs(
    modulation: List[Dict[str, object]],
    pfa: float,
    n: int = 16,
    save_path: Optional[Path] = None,
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
        xmin = min(h0s + h1s)
        xmax = max(h0s + h1s)
        x = np.linspace(xmin, xmax, binification)
        h0_pdf = rv_histogram(np.histogram(h0s, bins=binification)).pdf(x)
        h1_pdf = rv_histogram(np.histogram(h1s, bins=binification)).pdf(x)

        ax = axs[i // m, i % m]
        ax.plot(x, h0_pdf, label="$H_0$ case")
        ax.plot(x, h1_pdf, label="$H_1$ case")

        cut_off = modulation["Energy"]["λ0s"][pfa][idx]
        ax.axvline(cut_off, color="k", ls="--")
        mean_point = np.average([h0_mean, h1_mean])
        ax.axvline(mean_point, color="r", ls="-.")

        snr = modulation["snrs"][idx]
        ax.set_title(f"SNR={db(snr)}dB")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\mathbb{P}(\lambda)$")
        ax.legend(loc="best")
        ax.grid(True)
    plt.tight_layout()

    save_figure(fig, save_path)
    fig.suptitle(f"{modulation["name"]} - PDF of $H_0$ and $H_1$ cases")


def parse_args() -> Namespace:
    ap = base_parser()
    ap.add_argument("-f", "--pfa", default=[0.01], type=float, nargs="+")
    ap.add_argument(
        "-n",
        "--num-plots",
        default=16,
        type=int,
    )
    ap.add_argument("-r", "--regex", default="")
    return ap.parse_args()


if __name__ == "__main__":
    import gc
    import matplotlib.pyplot as plt
    from pathlib import Path
    import sys

    CWD: Path = Path(__file__).parent

    args = parse_args()

    assert (
        int(np.sqrt(args.num_plots)) ** 2 == args.num_plots
    ), f"{args.num_plots} needs to be a square number"

    print("Starting Probability Distribution Function (PDF) Analysis...")

    with timeit("Loading Data") as _:
        regex: re.Pattern = re.compile(args.regex)
        results: List[Dict[str, object]] = load_json(args.pd_file, filter=regex)
        gc.collect()

    with timeit("CFAR Analysis") as _:
        parse_fn = partial(parse_results, pfas=args.pfa)
        regressed: List[Dict[str, object]] = multi_parse(results, parse_fn)
        del results
        gc.collect()

    with timeit("Plotting") as _:
        for pfa in args.pfa:
            for modulation in regressed:
                # plot_some_pdfs(
                #     modulation,
                #     pfa=pfa,
                #     n=args.num_plots,
                #     save_path=args.save_dir
                #     / f"pdfs_energy_some_snrs_{modulation["name"]}_pfa_{pfa}.png",
                # )
                plot_specific_snrs(
                    modulation,
                    snrs=undb(np.array([6, -6, -18, -30])),
                    pfa=pfa,
                    save_path=args.save_dir
                    / f"pdfs_energy_{modulation["name"]}_pfa_{pfa}.png",
                )

    if not args.save:
        plt.show()

    if not sys.flags.interactive:
        plt.close("all")
