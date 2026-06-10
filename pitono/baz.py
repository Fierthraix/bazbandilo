#!/usr/bin/env python
from util import db

from cycler import cycler
import multiprocessing
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import pandas as pd
from typing import Dict, List


def plot_youden_j_with_multiple_modulations(
    modulation_test_results: List[Dict[str, object]],
    kind: str,
):
    fig, ax = plt.subplots()

    cycles = cycler(color=["r", "g", "b", "c", "m", "y"]) * cycler(
        linestyle=["-", ":", "-."]
    )
    ax.set_prop_cycle(cycles)
    for modulation in modulation_test_results:
        snrs = modulation["snrs"]
        youden_js: List[float] = modulation[kind]["youden_js"]
        ax.plot(db(snrs), youden_js, label=modulation["name"])

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Youden J")
    ax.legend(loc="best")
    fig.suptitle(kind)


def parse_results(modulation: Dict[str, object]) -> Dict[str, object]:
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
        for h0_λ, h1_λ in zip(dx["h0_λs"], dx["h1_λs"]):
            try:
                youden_j = log_regress(h0_λ, h1_λ)
                youden_js.append(youden_j)
            except ValueError:
                # youden_js.append(float('nan'))
                youden_js.append(0)
        dx["youden_js"] = youden_js
        mod_res[dx_result["kind"]] = dx
    return mod_res


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path

    CWD: Path = Path(__file__).parent
    results_file: Path = CWD.parent / "results.json"

    DETECTORS: List[str] = [
        "Energy",
        "MaxCut",
        "Dcs",
    ]

    # Load from JSON.
    with results_file.open("r") as f:
        results = json.load(f)

    # Parse and Log Regress results.
    # regressed: List[Dict[str, object]] = list(map(parse_results, results))
    with multiprocessing.Pool() as p:
        regressed: List[Dict[str, object]] = p.map(parse_results, results)

    for detector in DETECTORS:
        plot_youden_j_with_multiple_modulations(regressed, detector)
    plt.show()
