#!/usr/bin/env python
from foo import (
    DETECTORS,
    FIG_SIZE,
    filter_results,
)
from util import db, timeit

from argparse import ArgumentParser, Namespace
from collections import defaultdict
import concurrent.futures
import gc
from functools import partial
import numpy as np
import os
from pathlib import Path
import psutil
import re
from typing import Dict, List


def calculate_pd(pfa: float, h0_λs: List[float], h1_λs: List[float]) -> float:
    assert 0 <= pfa <= 1
    λ0: float = np.quantile(h0_λs, 1 - pfa)
    return np.mean([λ > λ0 for λ in h1_λs])


def parse_results(
    modulation: Dict[str, object],
    pfas: List[float] = [0.1, 0.05, 0.01],
) -> Dict[str, object]:
    mod_res = {
        "name": modulation["name"],
        "snrs": modulation["snrs"],
    }
    for dx_result in modulation["results"]:
        dx = {
            "kind": dx_result["kind"],
            "h0_λs": dx_result["h0_λs"],
            "h1_λs": dx_result["h1_λs"],
        }
        pfa_map: Dict[float, List[float]] = defaultdict(list)

        for h0_λ, h1_λ in zip(dx["h0_λs"], dx["h1_λs"]):  # For each SNR.
            for pfa in pfas:
                pfa_map[pfa].append(calculate_pd(pfa, h0_λ, h1_λ))
        mod_res[dx_result["kind"]] = pfa_map
    gc.collect()
    return mod_res


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
    for pfa, pds in modulation[kind].items():
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

    # Parse and Log Regress results.
    parse = partial(parse_results, pfas=[0.25, 0.15, 0.1, 0.05, 0.01])

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

    # DETECTORS = [k for k in regressed[0].keys() if k not in ("name", "snrs")]

    with timeit("Plotting") as _:
        for modulation in regressed:
            for detector in DETECTORS:
                plot_pd_vs_snr_cfar(
                    modulation, detector, save=args.save, save_dir=args.save_dir
                )

    if not args.save:
        plt.show()
