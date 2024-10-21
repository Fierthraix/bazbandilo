#!/usr/bin/env python
from util import db, timeit

from cycler import cycler
import multiprocessing
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import pandas as pd
from typing import Dict, List


def log_regress(h0_λs: List[float], h1_λs: List[float]) -> float:
    x_var = np.concatenate((h0_λs, h1_λs)).reshape(-1, 1)
    y_var = np.concatenate((np.zeros(len(h0_λs)), np.ones(len(h1_λs))))
    x_train, x_test, y_train, y_test = train_test_split(
        x_var, y_var, test_size=0.5, random_state=0
    )

    log_regression = LogisticRegression()
    log_regression.fit(x_train, y_train)

    y_pred_proba = log_regression.predict_proba(x_test)[::, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, drop_intermediate=False)

    df_test = pd.DataFrame(
        {
            "x": x_test.flatten(),
            "y": y_test,
            "proba": y_pred_proba,
        }
    )

    # sort it by predicted probabilities
    # because thresholds[1:] = y_proba[::-1]
    df_test.sort_values(by="proba", inplace=True)
    if len(tpr) == len(h0_λs):
        df_test["tpr"] = tpr[::-1]
    else:
        df_test["tpr"] = tpr[1:][::-1]
    if len(fpr) == len(h0_λs):
        df_test["fpr"] = fpr[::-1]
    else:
        df_test["fpr"] = fpr[1:][::-1]
    df_test["youden_j"] = df_test.tpr - df_test.fpr

    return max(abs(df_test["youden_j"]))


CYCLES: cycler = cycler(color=["r", "g", "b", "c", "m", "y"]) * cycler(
    linestyle=[
        "-",
        ":",
        "-.",
        "dashed",
        (0, (3, 1, 1, 1, 1, 1)),
        (0, (3, 5, 1, 5)),
    ]
)


def plot_youden_j_with_multiple_modulations(
    modulation_test_results: List[Dict[str, object]], kind: str, save=False
):
    fig, ax = plt.subplots()

    ax.set_prop_cycle(CYCLES)
    for modulation in modulation_test_results:
        snrs = modulation["snrs"]
        youden_js: List[float] = modulation[kind]["youden_js"]
        ax.plot(db(snrs), youden_js, label=modulation["name"])

    ax.set_xlabel("SNR (db)")
    ax.set_ylabel("Youden J")
    ax.legend(loc="best")
    fig.suptitle(kind)
    if save:
        fig.set_size_inches(16, 9)
        fig.savefig(f"/tmp/Youden-J_{kind}_multiple_modulations.png")


def plot_pd_vs_ber(
    modulation: Dict[str, object], bers: List[Dict[str, object]], save=False
):
    try:
        mod_ber = next(b for b in bers if b["name"] == modulation["name"])
    except TypeError:
        return
    fig, ax = plt.subplots()
    ber_ax = ax
    pd_ax = ax.twinx()
    ber_ax.plot(db(mod_ber["snrs"]), mod_ber["bers"], color="Red")
    # ber_ax.plot(1.31225, good_ber, 'ro')
    # ber_ax.axhline(good_ber, color='Red', ls='--', label=f"Acceptable BER ({good_ber})")
    ber_ax.set_ylim([0, 0.51])
    ber_ax.tick_params(axis="y", colors="Red")
    ber_ax.set_ylabel("Bit Error Rate (BER)", color="Red")
    ber_ax.legend(loc="best")

    linestyles = ["solid", "dashed", "dashdot"]
    for detector, style in zip(DETECTORS, linestyles):
        pd_ax.plot(
            db(modulation["snrs"]),
            modulation[detector]["youden_js"],
            color="Blue",
            linestyle=style,
            label=detector,
        )
    pd_ax.set_ylim([0, 1.1])
    # pd_ax.plot(-12.0556, good_pd, "bo")
    # pd_ax.axhline(good_pd, color='Blue', ls='--', label=f'Acceptable ℙd ({good_pd})')
    pd_ax.tick_params(axis="y", colors="Blue")
    pd_ax.set_ylabel(r"Probability of Detection ($\mathcal{P}_D$)", color="Blue")
    pd_ax.legend(loc="best")
    ax.set_xlabel("Signal to Noise Ratio dB (SNR dB)")
    ax.set_title(modulation["name"])
    if save:
        fig.set_size_inches(16, 9)
        fig.savefig(f'/tmp/ber_{modulation["name"]}.png')


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


def plot_all_bers(bers: List[Dict[str, object]], save=False):
    fig, ax = plt.subplots(1)
    ax.set_prop_cycle(CYCLES)
    for ber in bers:
        ax.plot(db(ber["snrs"]), ber["bers"], label=ber["name"])
    ax.legend(loc="best")
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("BER vs SNR (All Modulations)")
    if save:
        fig.set_size_inches(16, 9)
        fig.savefig("/tmp/bers_multiple_modulations.png")


def plot_pd_vs_ber_metric(
    modulation_test_results: List[Dict[str, object]],
    bers: List[Dict[str, object]],
    kind: str,
    save=False,
):
    fig, ax = plt.subplots()
    ax.set_prop_cycle(CYCLES)
    for modulation in modulation_test_results:
        try:
            mod_ber = next(b for b in bers if b["name"] == modulation["name"])
        except TypeError:
            return

        x = modulation[kind]["youden_js"]
        y = mod_ber["bers"]
        r = min(len(x), len(y))
        ax.plot(x[:r], y[:r], label=modulation["name"])
    ax.set_xlabel(r"Probability of Detection ($\mathcal{P}_D$)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(f"{kind}" + r"Detector - BER vs $\mathcal{{P}}_D$")
    ax.legend(loc="best")

    if save:
        fig.set_size_inches(16, 9)
        fig.savefig(f"/tmp/covert_metric_{kind}.png")


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path

    DETECTORS: List[str] = [
        "Energy",
        "MaxCut",
        "Dcs",
    ]

    CWD: Path = Path(__file__).parent

    with timeit('Loading Data') as _:
        # Load from JSON.
        results_file: Path = CWD.parent / "results_curr.json"
        with results_file.open("r") as f:
            results = json.load(f)

        bers_file: Path = CWD.parent / "bers_curr.json"
        with bers_file.open("r") as f:
            bers = json.load(f)

    # Parse and Log Regress results.
    # regressed: List[Dict[str, object]] = list(map(parse_results, results))
    with multiprocessing.Pool() as p, timeit("Logistic Regresstion") as _:
        regressed: List[Dict[str, object]] = p.map(parse_results, results)

    save = True
    for detector in DETECTORS:
        plot_youden_j_with_multiple_modulations(regressed, detector, save=save)

    for modulation in regressed:
        plot_pd_vs_ber(modulation, bers, save=save)

    plot_all_bers(bers, save=save)

    for detector in DETECTORS:
        plot_pd_vs_ber_metric(regressed, bers, detector, save=save)

    plt.show()
