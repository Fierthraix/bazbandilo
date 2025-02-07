#!/usr/bin/env python
from cfar import parse_results
from plot import (
    DETECTORS,
    FIG_SIZE,
    get_cycles,
    load_json,
    multi_parse,
)
from util import db, timeit

from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
import re
from typing import Dict, List


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
    ax.set_prop_cycle(get_cycles(len(modulation[kind]["pfas"])))
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


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument(
        "-b", "--ber-file", default=CWD.parent / "bers_curr.json", type=Path
    )
    ap.add_argument(
        "-p", "--pd-file", default=CWD.parent / "results_curr.json", type=Path
    )
    ap.add_argument("-r", "--regex", default="", type=str)
    ap.add_argument("-s", "--save", action="store_true")
    ap.add_argument("-d", "--save-dir", type=Path, default=Path("/tmp/"))
    ap.add_argument("--ebn0", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    import gc
    import matplotlib.pyplot as plt

    CWD: Path = Path(__file__).parent

    args = parse_args()

    with timeit("Loading Data") as _:
        regex = re.compile(args.regex)
        results: List[Dict[str, object]] = load_json(args.pd_file, filter=regex)
        gc.collect()

    # Parse and Log Regress results.
    with timeit("Logistic Regresstion") as _:
        parse_fn = partial(parse_results, pfas=[0.25, 0.15, 0.1, 0.05, 0.01])
        regressed: List[Dict[str, object]] = multi_parse(results, parse_fn)
        del results
        gc.collect()

    __dx = [k for k in regressed[0].keys() if k not in ("name", "snrs")]
    if sorted(__dx) != sorted(DETECTORS):
        DETECTORS = __dx

    with timeit("Plotting") as _:
        for modulation in regressed:
            for detector in DETECTORS:
                plot_pd_vs_snr_cfar(
                    modulation, detector, save=args.save, save_dir=args.save_dir
                )

    if not args.save:
        plt.show()
