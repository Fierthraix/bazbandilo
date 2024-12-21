#!/usr/bin/env python
from util import db, timeit
from ber import ber_bpsk

from argparse import ArgumentParser, Namespace
import concurrent.futures
from cycler import cycler
import gc
from functools import partial
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import pandas as pd
import re
from typing import Dict, List


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


def get_cycles(num_lines: int) -> cycler:
    colours: List[str] = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    linestyles: List[str] = [
        "-",
        ":",
        "-.",
        "dashed",
        (0, (3, 1, 1, 1, 1, 1)),
        (0, (3, 5, 1, 5)),
    ]

    r: int = int(np.ceil(num_lines / len(colours)))

    return cycler(color=colours) * cycler(linestyle=linestyles[:r])


def plot_youden_j_with_multiple_modulations(
    modulation_test_results: List[Dict[str, object]],
    kind: str,
    save=False,
    title: str = "",
):
    fig, ax = plt.subplots()
    ax.set_xlabel("SNR (db)")
    ax.set_ylabel("Youden J")
    ax.set_prop_cycle(get_cycles(len(modulation_test_results)))
    for modulation in modulation_test_results:
        snrs = modulation["snrs"]
        try:
            youden_js: List[float] = modulation[kind]["youden_js"]
        except KeyError:
            print(f"{kind} detector not found.")
            return
        ax.plot(db(snrs), youden_js, label=modulation["name"])

    ax.legend(loc="best")
    if save:
        fig.set_size_inches(16, 9)
        fig.savefig(f"/tmp/Youden-J_{kind}_multiple_modulations.png")
    if title:
        fig.suptitle(f"{kind} - {title}")
    else:
        fig.suptitle(kind)


def plot_pd_vs_ber(
    modulation: Dict[str, object], bers: List[Dict[str, object]], save=False
):
    try:
        mod_ber = next(b for b in bers if b["name"] == modulation["name"])
    except (TypeError, StopIteration):
        print(f"BER for {modulation["name"]} not found.")
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

    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    for detector, style in zip(DETECTORS, linestyles):
        try:
            pd_ax.plot(
                db(modulation["snrs"]),
                modulation[detector]["youden_js"],
                color="Blue",
                linestyle=style,
                label=detector,
            )
        except KeyError:
            continue
    pd_ax.set_ylim([0, 1.1])
    # pd_ax.plot(-12.0556, good_pd, "bo")
    # pd_ax.axhline(good_pd, color='Blue', ls='--', label=f'Acceptable ℙd ({good_pd})')
    pd_ax.tick_params(axis="y", colors="Blue")
    pd_ax.set_ylabel(r"Probability of Detection ($\mathbb{P}_D$)", color="Blue")
    pd_ax.legend(loc="best")
    ax.set_xlabel("Signal to Noise Ratio dB (SNR dB)")
    if save:
        fig.set_size_inches(16, 9)
        fig.savefig(f'/tmp/ber_{modulation["name"]}.png')
    ax.set_title(modulation["name"])


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
            except ValueError as err:
                # youden_js.append(float('nan'))
                # import pdb; pdb.set_trace()
                dfs.append(
                    pd.DataFrame(columns=["x", "y", "proba", "tpr", "fpr", "youden_j"])
                )
                # youden_js.append(0)
                youden_js.append(1)
        dx["youden_js"] = youden_js
        dx["df"] = dfs
        mod_res[dx_result["kind"]] = dx
    gc.collect()
    return mod_res


def plot_all_bers(bers: List[Dict[str, object]], save=False):
    fig, ax = plt.subplots(1)
    ax.set_prop_cycle(get_cycles(len(bers)))
    for ber in bers:
        ax.plot(db(ber["snrs"]), ber["bers"], label=ber["name"])
    ax.legend(loc="best")
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")

    ax.plot(
        db(ber["snrs"]),
        ber_bpsk(ber["snrs"]),
        color="Blue",
        ls="--",
        label="BPSK (Theoretical)",
    )
    if save:
        fig.set_size_inches(16, 9)
        fig.savefig("/tmp/bers_multiple_modulations.png")
    ax.set_title("BER vs SNR (All Modulations)")


def plot_pd_vs_ber_metric(
    modulation_test_results: List[Dict[str, object]],
    bers: List[Dict[str, object]],
    kind: str,
    save=False,
):
    fig, ax = plt.subplots()
    ax.set_prop_cycle(get_cycles(len(modulation_test_results)))
    ax.set_xlabel(r"Probability of Detection ($\mathbb{P}_D$)")
    ax.set_ylabel("Bit Error Rate (BER)")
    for modulation in modulation_test_results:
        try:
            mod_ber = next(b for b in bers if b["name"] == modulation["name"])
        except TypeError:
            return
        except StopIteration:
            print(f'{modulation["name"]} BER not found.')
            return

        try:
            x = modulation[kind]["youden_js"]
        except KeyError:
            print(f"{kind} detector not found")
            return
        y = mod_ber["bers"]
        r = min(len(x), len(y))
        ax.plot(x[:r], y[:r], label=modulation["name"])
    ax.set_xlabel(r"Probability of Detection ($\mathbb{P}_D$)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.legend(loc="best")

    if save:
        fig.set_size_inches(16, 9)
        fig.savefig(f"/tmp/covert_metric_{kind}.png")
    ax.set_title(f"{kind} Detector" + r" - BER vs $\mathbb{P}_D$")


def plot_pd_vs_pfa(
    results_object: List[Dict[str, object]],
    kind: str,
    save=False,
):
    fig, ax = plt.subplots()
    ax.set_prop_cycle(get_cycles(len(results_object)))
    ax.set_xlabel(r"Probability of False Alarm ($\mathbb{P}_{FA}$)")
    ax.set_ylabel(r"Probability of Detection ($\mathbb{P}_D$)")
    for modulation in results_object:
        try:
            p = modulation[kind]["df"]
        except KeyError:
            print(f"{kind} detector not found.")
            return
        mid = len(p) // 2
        x = p[mid]["tpr"]
        y = p[mid]["fpr"]
        snr_db = db(modulation["snrs"][mid])
        ax.plot(x, y, label=f"{modulation["name"]}")
    ax.legend(loc="best")

    if save:
        fig.set_size_inches(16, 9)
        fig.savefig(f"/tmp/pd_vs_pfa_{kind}_{snr_db}dB.png")
    ax.set_title(
        f"{kind}"
        + r"Detector - $\mathbb{P}_D$ vs $\mathbb{{P}}_{FA}$ - "
        + f"SNR={snr_db:.2f}"
    )


def plot_λ_vs_snr(
    results_object: List[Dict[str, object]], kind: str, save: bool = False
):
    fig, ax = plt.subplots()
    ax.set_prop_cycle(get_cycles(len(results_object)))
    for modulation in results_object:
        try:
            p = modulation[kind]["df"]
        except KeyError:
            continue
        λ0s = []
        for i, snr in enumerate(modulation["snrs"]):
            try:
                λ0 = (
                    p[i]
                    .sort_values(by="youden_j", ascending=False, ignore_index=True)
                    .iloc[0]
                    .x
                )
            except IndexError:
                λ0 = 0
            λ0s.append(λ0)

        ax.plot(db(modulation["snrs"]), λ0s, label=f"{modulation['name']}")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Threshold λ")
    ax.legend(loc="best")

    if save:
        fig.set_size_inches(16, 9)
        fig.savefig(f"/tmp/lambda_{kind}.png")
    ax.set_title(f"Threshold vs SNR ({kind})")


def filter_results(
    results_object: List[Dict[str, object]], pattern: re.Pattern
) -> List[Dict[str, object]]:
    return list(filter(lambda mod: pattern.match(mod["name"]), results_object))


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
    ap.add_argument("--bers-only", action="store_true")
    return ap.parse_args()


DETECTORS: List[str] = ["Energy", "MaxCut", "Dcs", "NormalTest"]

if __name__ == "__main__":
    import json

    # import umsgpack
    import matplotlib.pyplot as plt
    from pathlib import Path

    CWD: Path = Path(__file__).parent

    args = parse_args()

    regex = re.compile(args.regex)

    with timeit("Loading Data") as _:
        # Load from JSON.
        if not args.bers_only:
            with Path(args.pd_file).open("r") as f:
                results = json.load(f)
                # results = umsgpack.load(f, raw=True)
            results = filter_results(results, regex)

        gc.collect()

        with Path(args.ber_file).open("r") as f:
            bers = json.load(f)
            # bers = umsgpack.load(f, raw=True)
        bers = filter_results(bers, regex)

        gc.collect()

    if not args.bers_only:
        # Parse and Log Regress results.
        parse = partial(parse_results, num_regressions=args.log_regressions)
        # regressed: List[Dict[str, object]] = list(map(parse, results))
        with concurrent.futures.ProcessPoolExecutor() as p, timeit(
            "Logistic Regresstion"
        ) as _:
            regressed: List[Dict[str, object]] = list(p.map(parse, results))
        del results
        gc.collect()

        DETECTORS = [k for k in regressed[0].keys() if k not in ("name", "snrs")]

        with timeit("Plotting") as _:

            for detector in DETECTORS:
                plot_youden_j_with_multiple_modulations(
                    regressed, detector, save=args.save, title=args.pd_file.name
                )
            for detector in DETECTORS:
                plot_pd_vs_pfa(regressed, detector, save=args.save)

            for modulation in regressed:
                plot_pd_vs_ber(modulation, bers, save=args.save)

            for detector in DETECTORS:
                plot_λ_vs_snr(regressed, detector, save=args.save)

            for detector in DETECTORS:
                plot_pd_vs_ber_metric(regressed, bers, detector, save=args.save)

    plot_all_bers(bers, save=args.save)

    if not args.save:
        plt.show()
