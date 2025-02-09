#!/usr/bin/env python3
from cfar import calculate_pd, get_threshold
from plot import save_figure

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import rv_histogram
from sklearn import metrics


def calc_tpr_fpr(pfa: float, h0_samps: np.ndarray, h1_samps: np.ndarray):
    low_end = min(h0_samps.min(), h1_samps.min())
    high_end = max(h0_samps.max(), h1_samps.max())

    lambdas = np.linspace(low_end, high_end, 500)

    tpr = [np.mean([λ > λ0 for λ in h1_samps]) for λ0 in lambdas]
    fpr = [np.mean([λ > λ0 for λ in h0_samps]) for λ0 in lambdas]

    return tpr, fpr


if __name__ == "__main__":
    binification = 128
    # Get Points
    num_iters: int = 9001

    GRAPHS_DIR = Path(__file__).parent.parent / "final" / "graphs"

    SAVE: bool = False

    # These values represent the output statistic
    # of the detector in the H0 and H1 scenarios.

    μ0: float = 86
    σ0: float = 2
    h0_samps: np.ndarray = np.random.normal(μ0, σ0, num_iters)

    h0 = rv_histogram(np.histogram(h0_samps, bins=binification))

    μ1: float = 91
    σ1: float = 2.5
    h1_samps: np.ndarray = np.random.normal(μ1, σ1, num_iters)

    h1 = rv_histogram(np.histogram(h1_samps, bins=binification))

    x = np.linspace(75, 100, binification)
    h0_pdf = h0.pdf(x)
    h1_pdf = h1.pdf(x)

    cfar_pfa = 0.1
    cfar_threshold = get_threshold(cfar_pfa, h0_samps)
    cfar_tpr = calculate_pd(cfar_pfa, h0_samps, h1_samps)

    # Plot the PDFs of the H0 and H1 cases.
    fig, ax = plt.subplots()
    ax.set_ylabel(r"$\mathbb{P}(\lambda)$")
    ax.set_xlabel(r"Detector Output $\lambda$")
    ax.plot(x, h0_pdf)
    ax.plot(x, h1_pdf)
    # ax.axvline(cut_off.x, color="k", ls="--")
    ax.axvline(
        cfar_threshold,
        color="k",
        ls="--",
        label=r"Threshold for $\mathbb{P}_{FA}=" f"{cfar_pfa}$",
    )
    ax.legend(loc="best")
    if SAVE:
        save_figure(fig, GRAPHS_DIR / "pd:example.png", fig_size=(8, 4.5))

    tpr, fpr = calc_tpr_fpr(cfar_pfa, h0_samps, h1_samps)
    auc = metrics.auc(fpr, tpr)

    # Plot the ROC curve.
    fig, ax = plt.subplots()
    metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot(ax=ax)
    ax.axline(xy1=(0, 0), slope=1, color="r", ls=":")
    ax.axvline(cfar_pfa, color="k", ls="--")
    ax.plot(
        cfar_pfa,
        cfar_tpr,
        "ko",
        ms=10,
        label=r"Threshold $\lambda_0="
        f"{cfar_threshold:.1f}$ "
        r"($\mathbb{P}_{FA}="
        f"{cfar_pfa}$)",
    )
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.legend(loc=4)
    if SAVE:
        save_figure(fig, GRAPHS_DIR / "roc:example.png")
    ax.set_title("ROC Curve")

    plt.show()
