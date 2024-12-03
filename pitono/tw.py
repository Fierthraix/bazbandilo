#!/usr/bin/env python
from util import db, timeit
from foo import get_cycles, log_regress

from argparse import ArgumentParser, Namespace
import concurrent.futures
from functools import partial
import numpy as np
import pandas as pd
from typing import Dict, List


def parse_results(
    modulation: Dict[str, object], num_regressions: int = 1
) -> Dict[str, object]:
    pow2 = int(modulation["num_samples"])
    mod_res = {
        "name": f"2^{int(np.log2(pow2))}",
        "snrs": modulation["snrs"],
    }
    dx = {
        "h0_λs": modulation["results"]["h0_λs"],
        "h1_λs": modulation["results"]["h1_λs"],
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
            # youden_js.append(float('nan'))
            dfs.append(
                pd.DataFrame(columns=["x", "y", "proba", "tpr", "fpr", "youden_j"])
            )
            youden_js.append(0)
    dx["youden_js"] = youden_js
    dx["df"] = dfs
    mod_res["results"] = dx
    return mod_res


def plot_all_tws(
    modulation_test_results: List[Dict[str, object]],
    save=False,
):
    fig, ax = plt.subplots()
    ax.set_xlabel("SNR (db)")
    ax.set_ylabel("Youden J")
    fig.suptitle("SNR vs PD - Different $TW$ products")

    ax.set_prop_cycle(get_cycles(len(modulation_test_results)))
    for modulation in modulation_test_results:
        snrs = modulation["snrs"]
        youden_js: List[float] = modulation["results"]["youden_js"]
        ax.plot(db(snrs), youden_js, label=modulation["name"])

    ax.legend(loc="best")
    if save:
        fig.set_size_inches(16, 9)
        fig.savefig("/tmp/Youden-J_different_TW_product.png")


def plot_all_tws_old(
    results_object: List[Dict[str, object]],
    save=False,
):
    fig, ax = plt.subplots()
    ax.set_prop_cycle(get_cycles(len(results_object)))
    ax.set_xlabel(r"Signal-to-Noise Ration (SNR) (dB)")
    ax.set_ylabel(r"Probability of Detection ($\mathbb{P}_D$)")
    ax.set_title(r"Energy Detector $\text{SNR}$ vs $\mathbb{{P}}_D$")
    for modulation in results_object:
        p = modulation["results"]["youden_j"]
        snr_db = db(modulation["snrs"])
        ax.plot(snr_db, p, label=f"{modulation["name"]}")
    ax.legend(loc="best")

    if save:
        fig.set_size_inches(16, 9)
        fig.savefig("/tmp/tw_snr_vs_pd.png")


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument("-t", "--tw-file", default=CWD.parent / "tw.json", type=Path)
    ap.add_argument("-l", "--log-regressions", default=1, type=int)
    ap.add_argument("-s", "--save", action="store_true")
    return ap.parse_args()


DETECTORS: List[str] = ["Energy", "MaxCut", "Dcs", "NormalTest"]

if __name__ == "__main__":
    import json

    import matplotlib.pyplot as plt
    from pathlib import Path

    CWD: Path = Path(__file__).parent

    args = parse_args()

    with timeit("Loading Data") as _:
        # Load from JSON.
        with Path(args.tw_file).open("r") as f:
            results = json.load(f)

    # Parse and Log Regress results.
    parse = partial(parse_results, num_regressions=args.log_regressions)
    # regressed: List[Dict[str, object]] = list(map(parse, results))
    with concurrent.futures.ProcessPoolExecutor() as p, timeit(
        "Logistic Regresstion"
    ) as _:
        regressed: List[Dict[str, object]] = list(p.map(parse, results))

    with timeit("Plotting") as _:

        plot_all_tws(regressed)

    plt.show()