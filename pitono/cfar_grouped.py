#!/usr/bin/env python
from cfar import (
    get_thresholds,
    parse_results,
)
from plot import (
    DETECTORS,
    GROUPS,
    GROUP_MARKERS,
    base_parser,
    load_json,
    multi_parse_grouped,
    plot_pd_with_multiple_modulations,
    plot_pd_vs_ber_metric,
    plot_λ_vs_snr,
)
from util import timeit

from argparse import Namespace
from functools import partial
from typing import Dict, List


def parse_args() -> Namespace:
    ap = base_parser()
    ap.add_argument("-f", "--pfa", default=0.01, type=float)
    ap.add_argument("-g", "--group", type=int, choices=(1, 2, 3), default=None)
    return ap.parse_args()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import gc

    args = parse_args()
    if args.group:
        group_ids = [args.group]
    else:
        group_ids = [1, 2, 3]

    with timeit("Loading Data") as _:
        results: List[Dict[str, object]] = load_json(args.pd_file)
        grouped_results: List[List[Dict]] = [
            [r for r in results if r["name"] in GROUPS[group_id]]
            for group_id in group_ids
        ]
        del results
        gc.collect()

        bers: List[Dict[str, object]] = load_json(args.ber_file)
        grouped_bers: List[List[Dict]] = [
            [b for b in bers if b["name"] in GROUPS[group_id]] for group_id in group_ids
        ]
        del bers
        gc.collect()

    PFA: float = args.pfa

    # Parse and Log Regress results.
    with timeit("CFAR Analysis.") as _:
        parse_fn = partial(parse_results, pfas=[PFA])
        grouped_regress: List[List[Dict[str, object]]] = multi_parse_grouped(
            grouped_results, parse_fn
        )
        del grouped_results
        gc.collect()

    __dx = [k for k in grouped_regress[0][0].keys() if k not in ("name", "snrs")]
    if sorted(__dx) != sorted(DETECTORS):
        DETECTORS = __dx

    with timeit("Plotting") as _:

        for group_id, regressed, bers in zip(group_ids, grouped_regress, grouped_bers):

            for detector in DETECTORS:
                plot_pd_with_multiple_modulations(
                    [(mod["name"], mod[detector]["pfas"][PFA]) for mod in regressed],
                    regressed[0]["snrs"],
                    save_path=args.save_dir
                    / f"cfar_{detector}_pfa_{PFA}_group_{group_id}.png",
                    cycles=GROUP_MARKERS[group_id],
                )

            for detector in DETECTORS:
                lambdas: List[float] = get_thresholds(PFA, regressed, detector)
                plot_λ_vs_snr(
                    lambdas,
                    regressed[0]["snrs"],
                    save_path=args.save_dir / f"lambda_{detector}_group_{group_id}.png",
                    cycles=GROUP_MARKERS[group_id],
                )

            for detector in DETECTORS:
                plot_pd_vs_ber_metric(
                    [
                        (
                            mod["name"],
                            mod[detector]["pfas"][PFA],
                            next(b["bers"] for b in bers if b["name"] == mod["name"]),
                        )
                        for mod in regressed
                    ],
                    save_path=args.save_dir / f"covert_metric_{detector}.png",
                    cycles=GROUP_MARKERS[group_id],
                )

    if not args.save:
        plt.show()
