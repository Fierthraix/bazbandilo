#!/usr/bin/env python
from plot import (
    DETECTORS,
    base_parser,
    load_json,
    get_cycles,
    multi_parse,
    plot_pd_with_multiple_modulations,
    plot_pd_vs_ber,
    plot_pd_vs_ber_metric,
    plot_pd_vs_pfa,
    plot_λ_vs_snr,
    set_snr_xlim,
)
from util import db, timeit

from argparse import Namespace
from functools import partial
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from typing import Dict, List, Tuple


def log_regress(
    h0_λs: List[float], h1_λs: List[float], return_df: bool = False
) -> pd.DataFrame:
    x_var = np.concatenate((h0_λs, h1_λs)).reshape(-1, 1)
    y_var = np.concatenate((np.zeros(len(h0_λs)), np.ones(len(h1_λs))))
    x_train, x_test, y_train, y_test = train_test_split(
        x_var, y_var, test_size=0.5, random_state=0
    )

    log_regression = LogisticRegression(solver="liblinear")
    log_regression.fit(x_train, y_train)

    y_pred_proba = log_regression.predict_proba(x_test)[::, 1]
    # y_pred_proba = log_regression.decision_function(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, drop_intermediate=False)

    df = pd.DataFrame(
        {
            "x": x_test.flatten(),
            "y": y_test,
            "proba": y_pred_proba,
        }
    )

    # sort it by predicted probabilities
    # because thresholds[1:] = y_proba[::-1]
    df.sort_values(by="proba", inplace=True)
    if len(tpr) == len(h0_λs):
        df["tpr"] = tpr[::-1]
    elif len(tpr) == len(h0_λs) + 1:
        df["tpr"] = tpr[1:][::-1]
    else:
        print("New Case!")
        # assert False

    if len(fpr) == len(h0_λs):
        df["fpr"] = fpr[::-1]
    else:
        df["fpr"] = fpr[1:][::-1]
    df["youden_j"] = df.tpr - df.fpr

    return df


def parse_results(
    modulation: Dict[str, object], num_regressions: int = 1
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
        youden_js = []
        dfs = []
        for h0_λ, h1_λ in zip(dx["h0_λs"], dx["h1_λs"]):
            try:
                these_dfs = [log_regress(h0_λ, h1_λ) for _ in range(num_regressions)]
                # df = log_regress(h0_λ, h1_λ)
                df = these_dfs[0].copy()
                if num_regressions > 1:
                    for df_i in these_dfs[1:]:
                        df += df_i
                    df /= num_regressions
                youden_js.append(max(abs(df["youden_j"])))
                dfs.append(df)
            except ValueError:
                dfs.append(
                    pd.DataFrame(columns=["x", "y", "proba", "tpr", "fpr", "youden_j"])
                )
                # youden_js.append(float('nan'))
                # youden_js.append(0)
                youden_js.append(1)
        dx["youden_js"] = youden_js
        dx["df"] = dfs
        mod_res[dx_result["kind"]] = dx
    return mod_res


def get_thresholds(
    regressed: List[Dict[str, object]], detector: str
) -> Tuple[str, List[float]]:
    lambdas = []
    for modulation in regressed:
        λ0s = []
        for i, snr in enumerate(modulation["snrs"]):
            try:
                λ0 = (
                    modulation[detector]["df"][i]
                    .sort_values(
                        by="youden_j",
                        ascending=False,
                        ignore_index=True,
                    )
                    .iloc[0]
                    .x
                )
            except IndexError:
                λ0 = 0
            λ0s.append(λ0)
        lambdas.append((modulation["name"], λ0s))
    return lambdas


def get_pfa_and_pd(
    regressed: List[Dict[str, object]],
    detector: str,
) -> Tuple[float, List[Tuple[str, List[float], List[float]]]]:
    mid_idx = len(regressed[0]["snrs"]) // 2
    mid_snr = regressed[0]["snrs"][mid_idx]

    data: List[Tuple[str, List[float], List[float]]] = [
        (
            mod["name"],
            mod[detector]["df"][mid_idx]["tpr"],
            mod[detector]["df"][mid_idx]["fpr"],
        )
        for mod in regressed
    ]
    return mid_snr, data


def parse_args() -> Namespace:
    ap = base_parser()
    ap.add_argument("-l", "--log-regressions", default=1, type=int)
    ap.add_argument("-r", "--regex", default="", type=str)
    return ap.parse_args()


if __name__ == "__main__":
    import gc
    import matplotlib.pyplot as plt
    import sys

    args = parse_args()
    set_snr_xlim(args.snr_db_min, args.snr_db_max)

    print("Starting Youden's J-Index Analysis...")

    with timeit("Loading Data") as _:
        regex = re.compile(args.regex)
        results: List[Dict[str, object]] = load_json(args.pd_file, filter=regex)
        gc.collect()

        bers: List[Dict[str, object]] = load_json(args.ber_file, filter=regex)
        gc.collect()

    # Parse and Log Regress results.
    with timeit("Logistic Regression") as _:
        parse_fn = partial(parse_results, num_regressions=args.log_regressions)
        regressed = multi_parse(results, parse_fn)

    __dx = [k for k in regressed[0].keys() if k not in ("name", "snrs")]
    if sorted(__dx) != sorted(DETECTORS):
        DETECTORS = __dx

    with timeit("Plotting") as _:

        for detector in DETECTORS:
            plot_pd_with_multiple_modulations(
                [(mod["name"], mod[detector]["youden_js"]) for mod in regressed],
                regressed[0]["snrs"],
                save_path=args.save_dir
                / f"Youden-J_{detector}_multiple_modulations.png",
                cycles=get_cycles(len(regressed)),
            )

        for detector in DETECTORS:
            lambdas: List[float] = get_thresholds(regressed, detector)
            plot_λ_vs_snr(
                lambdas,
                regressed[0]["snrs"],
                save_path=args.save_dir / f"lambda_{detector}.png",
                cycles=get_cycles(len(regressed)),
            )

        for detector in DETECTORS:
            mid_snr, data = get_pfa_and_pd(regressed, detector)
            plot_pd_vs_pfa(
                data,
                save_path=args.save_dir / f"pd_vs_pfa_{detector}_{db(mid_snr)}dB.png",
                cycles=get_cycles(len(regressed)),
            )

        for modulation in regressed:
            pds: List[Tuple[str, List[float]]] = [
                (detector, modulation[detector]["youden_js"]) for detector in DETECTORS
            ]
            plot_pd_vs_ber(
                pds,
                next(b["bers"] for b in bers if b["name"] == modulation["name"]),
                modulation["snrs"],
                save_path=args.save_dir / f'ber_{modulation["name"]}.png',
            )

        for detector in DETECTORS:
            plot_pd_vs_ber_metric(
                [
                    (
                        mod["name"],
                        mod[detector]["youden_js"],
                        next(b["bers"] for b in bers if b["name"] == mod["name"]),
                    )
                    for mod in regressed
                ],
                save_path=args.save_dir / f"covert_metric_{detector}.png",
                cycles=get_cycles(len(regressed)),
            )

    if not args.save:
        plt.show()

    if not sys.flags.interactive:
        plt.close("all")
