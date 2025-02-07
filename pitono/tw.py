#!/usr/bin/env python
from cfar import calculate_pd, get_threshold
from util import timeit
from plot import (
    base_parser,
    get_cycles,
    load_json,
    multi_parse,
    plot_pd_with_multiple_modulations,
)

from argparse import Namespace
from collections import defaultdict
from functools import partial
import numpy as np
from pathlib import Path
from typing import Dict, List


def parse_results(
    modulation: Dict[str, object],
    pfas: List[float] = [0.1, 0.05, 0.01],
) -> Dict[str, object]:
    pow2 = int(modulation["num_samples"])
    mod_res = {
        "name": f"2^{int(np.log2(pow2))}",
        "snrs": modulation["snrs"],
    }
    dx = {
        "h0_λs": modulation["results"]["h0_λs"],
        "h1_λs": modulation["results"]["h1_λs"],
        "pfas": defaultdict(list),
        "λ0s": defaultdict(list),
    }
    for h0_λ, h1_λ in zip(dx["h0_λs"], dx["h1_λs"]):
        for pfa in pfas:
            dx["pfas"][pfa].append(calculate_pd(pfa, h0_λ, h1_λ))
            dx["λ0s"][pfa].append(get_threshold(pfa, h0_λ))
    mod_res["results"] = dx
    return mod_res


def parse_args() -> Namespace:
    ap = base_parser()
    ap.add_argument("-t", "--tw-file", default=CWD.parent / "tw.json", type=Path)
    ap.add_argument("-f", "--pfa", default=0.01, type=float)
    ap.add_argument("-l", "--log-regressions", default=1, type=int)
    return ap.parse_args()


if __name__ == "__main__":
    import gc
    import matplotlib.pyplot as plt
    import sys

    CWD: Path = Path(__file__).parent

    args: Namespace = parse_args()
    PFA: float = args.pfa

    with timeit("Loading Data") as _:
        results: List[Dict[str, object]] = load_json(args.tw_file)
        for modulation in results:
            pow2 = int(modulation["num_samples"])
            modulation["name"] = f"2^{int(np.log2(pow2))}"
        gc.collect()

    # Parse and Log Regress results.
    with timeit("CFAR Analysis") as _:
        parse_fn = partial(parse_results, pfas=[PFA])
        regressed = multi_parse(results, parse_fn)
        del results
        gc.collect()

    with timeit("Plotting") as _:
        plot_pd_with_multiple_modulations(
            [(mod["name"], mod["results"]["pfas"][PFA]) for mod in regressed],
            regressed[0]["snrs"],
            save_path=args.save_dir / "tw_snr_vs_pd.png",
            cycles=get_cycles(len(regressed)),
        )

    if not args.save:
        plt.show()

    if not sys.flags.interactive:
        plt.close("all")
