#!/usr/bin/env python3
from util import timeit
from plot import (
    DETECTORS,
    base_parser,
    get_cycles,
    load_json,
    multi_parse,
    plot_pd_with_multiple_modulations,
    plot_pd_vs_ber,
    plot_pd_vs_ber_metric,
    plot_λ_vs_snr,
)

from argparse import Namespace
from collections import defaultdict
from functools import partial
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple


def calculate_pd(pfa: float, h0_λs: List[float], h1_λs: List[float]) -> float:
    assert 0 <= pfa <= 1
    λ0: float = get_threshold(pfa, h0_λs)
    return np.mean([λ > λ0 for λ in h1_λs])


def get_threshold(pfa: float, h0_λs: List[float]):
    return np.quantile(h0_λs, 1 - pfa)


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
            "pfas": defaultdict(list),
            "λ0s": defaultdict(list),
        }
        for h0_λ, h1_λ in zip(dx["h0_λs"], dx["h1_λs"]):  # For each SNR.
            for pfa in pfas:
                dx["pfas"][pfa].append(calculate_pd(pfa, h0_λ, h1_λ))
                dx["λ0s"][pfa].append(get_threshold(pfa, h0_λ))
        mod_res[dx_result["kind"]] = dx
    return mod_res


def get_thresholds(
    pfa: float, regressed: List[Dict[str, object]], detector: str
) -> List[Tuple[str, List[float]]]:
    return [
        (modulation["name"], modulation[detector]["λ0s"][pfa])
        for modulation in regressed
    ]


def parse_args() -> Namespace:
    ap = base_parser()
    ap.add_argument("-f", "--pfa", default=[0.01], type=float, nargs="+")
    ap.add_argument("-r", "--regex", default="", type=str)
    return ap.parse_args()


if __name__ == "__main__":
    import gc
    import matplotlib.pyplot as plt

    CWD: Path = Path(__file__).parent

    args = parse_args()

    with timeit("Loading Data") as _:
        regex: re.Pattern = re.compile(args.regex)
        results: List[Dict[str, object]] = load_json(args.pd_file, filter=regex)
        gc.collect()

        bers: List[Dict[str, object]] = load_json(args.ber_file, filter=regex)
        gc.collect()

    with timeit("CFAR Analysis") as _:
        parse_fn = partial(parse_results, pfas=args.pfa)
        regressed: List[Dict[str, object]] = multi_parse(results, parse_fn)
        del results
        gc.collect()

    __dx = [k for k in regressed[0].keys() if k not in ("name", "snrs")]
    if sorted(__dx) != sorted(DETECTORS):
        DETECTORS = __dx

    with timeit("Plotting") as _:
        for pfa in args.pfa:
            for detector in DETECTORS:
                plot_pd_with_multiple_modulations(
                    [(mod["name"], mod[detector]["pfas"][pfa]) for mod in regressed],
                    regressed[0]["snrs"],
                    save_path=args.save_dir / f"cfar_{detector}_pfa_{pfa}.png",
                    cycles=get_cycles(len(regressed)),
                )

            for modulation in regressed:
                pds: List[Tuple[str, List[float]]] = [
                    (detector, modulation[detector]["pfas"][pfa])
                    for detector in DETECTORS
                ]
                plot_pd_vs_ber(
                    pds,
                    next(b["bers"] for b in bers if b["name"] == modulation["name"]),
                    modulation["snrs"],
                    save_path=args.save_dir / f'ber_{modulation["name"]}_pfa_{pfa}.png',
                )

            for detector in DETECTORS:
                lambdas: List[float] = get_thresholds(pfa, regressed, detector)
                plot_λ_vs_snr(
                    lambdas,
                    regressed[0]["snrs"],
                    save_path=args.save_dir / f"lambda_{detector}_pfa_{pfa}.png",
                    cycles=get_cycles(len(regressed)),
                )

            for detector in DETECTORS:
                plot_pd_vs_ber_metric(
                    [
                        (
                            mod["name"],
                            mod[detector]["pfas"][pfa],
                            next(b["bers"] for b in bers if b["name"] == mod["name"]),
                        )
                        for mod in regressed
                    ],
                    save_path=args.save_dir / f"covert_metric_{detector}_pfa_{pfa}.png",
                    cycles=get_cycles(len(regressed)),
                )

    if not args.save:
        plt.show()
