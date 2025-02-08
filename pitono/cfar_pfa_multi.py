#!/usr/bin/env python
from cfar import parse_results
from plot import (
    DETECTORS,
    base_parser,
    load_json,
    plot_pd_vs_snr_cfar,
    multi_parse,
)
from util import timeit

from argparse import Namespace
from functools import partial
from pathlib import Path
import re
from typing import Dict, List


def parse_args() -> Namespace:
    ap = base_parser()
    ap.add_argument("-r", "--regex", default="", type=str)
    ap.add_argument(
        "-f", "--pfa", default=[0.25, 0.15, 0.1, 0.05, 0.01], type=float, nargs="+"
    )
    ap.add_argument("--ebn0", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    import gc
    import matplotlib.pyplot as plt
    import sys

    CWD: Path = Path(__file__).parent

    args = parse_args()

    print("Starting CFAR Analysis...")

    with timeit("Loading Data") as _:
        regex = re.compile(args.regex)
        results: List[Dict[str, object]] = load_json(args.pd_file, filter=regex)
        gc.collect()

    # Parse and Log Regress results.
    with timeit("CFAR Analysis") as _:
        parse_fn = partial(parse_results, pfas=args.pfa)
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
                    modulation,
                    detector,
                    save_path=args.save_dir
                    / f'cfar_pd_vs_snr_{detector}_{modulation["name"]}.png',
                )

    if not args.save:
        plt.show()

    if not sys.flags.interactive:
        plt.close("all")
