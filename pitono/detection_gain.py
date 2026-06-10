#!/usr/bin/env python
from util import db, timeit
from foo import filter_results, get_cycles, log_regress, FIG_SIZE, parse_results

from argparse import ArgumentParser, Namespace
import concurrent.futures
import gc
from functools import partial
import numpy as np
import os
from pathlib import Path
import psutil
import re
from typing import Dict, List, Tuple


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument(
        "-b", "--ber-file", default=CWD.parent / "bers_curr.json", type=Path
    )
    ap.add_argument(
        "-p", "--pd-file", default=CWD.parent / "results_curr.json", type=Path
    )
    ap.add_argument("-l", "--log-regressions", default=1, type=int)
    ap.add_argument("-r", "--regex", default="", type=str)
    ap.add_argument("-s", "--save", action="store_true")
    ap.add_argument("-d", "--save-dir", type=Path, default=Path("/tmp/"))
    return ap.parse_args()


def detector_gain(
    modulation: Dict[str, object],
    bers: List[Dict[str, object]],
    pfa: float = 0.05,
    ber: float = 0.05,
    save=False,
    save_dir=Path("/tmp/"),
):
    try:
        mod_ber = next(b for b in bers if b["name"] == modulation["name"])
    except (TypeError, StopIteration):
        print(f"BER for {modulation["name"]} not found.")
        return

    # For a specified PFA and BER, plot Pd:
    # 1. Find the threshold for the desired PFA.
    # 2. Find the SNR with the required BER.

    # 1. Find the threshold for the desired PFA.
    foo = modulation["df"]
    foo.sort_values(by="fpr")


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
    parse = partial(parse_results, num_regressions=args.log_regressions)

    num_cpus: int = os.cpu_count()
    ram: int = psutil.virtual_memory().available
    num_workers = min(ram // results_file_size, num_cpus)

    # TODO: XXX:
    results = results[:1]
    # TODO: XXX:
    with timeit("Logistic Regresstion") as _:
        if num_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as p:
                regressed: List[Dict[str, object]] = list(p.map(parse, results))
        else:
            regressed: List[Dict[str, object]] = list(map(parse, results))
    del results
    gc.collect()

    DETECTORS = [k for k in regressed[0].keys() if k not in ("name", "snrs")]

    with timeit("Plotting") as _:
        ...

    if not args.save:
        plt.show()
