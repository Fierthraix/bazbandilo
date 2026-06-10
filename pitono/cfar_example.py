#!/usr/bin/env python
from cfar import calculate_pd

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_histogram
from typing import Dict, List


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
        }
        pfa_map: Dict[float, List[float]] = defaultdict(list)

        for h0_λ, h1_λ in zip(dx["h0_λs"], dx["h1_λs"]):  # For each SNR.
            for pfa in pfas:
                pfa_map[pfa].append(calculate_pd(pfa, h0_λ, h1_λ))
        mod_res[dx_result["kind"]] = pfa_map
    return mod_res


def greater_than_threshold(threshold: float, λs: List[float]):
    return np.mean([λ > threshold for λ in λs])


def plot_pd_and_pfa_versus_threshold(h0_samps: List[float], h1_samps: List[float]):
    binification = 150

    lambda_min = min(min(i) for i in (h0_samps, h1_samps))
    lambda_max = max(max(i) for i in (h0_samps, h1_samps))
    lambdas = np.linspace(lambda_min, lambda_max, binification)

    #: Plot of P_D as a function of lambda_0
    pfa_vs_lambda = [
        greater_than_threshold(threshold, h0_samps) for threshold in lambdas
    ]
    pd_vs_lambda = [
        greater_than_threshold(threshold, h1_samps) for threshold in lambdas
    ]

    fig, ax = plt.subplots(2, 1)

    h0 = rv_histogram(np.histogram(h0_samps, bins=binification))
    h1 = rv_histogram(np.histogram(h1_samps, bins=binification))
    h0_pdf = h0.pdf(lambdas)
    h1_pdf = h1.pdf(lambdas)
    ax[0].plot(lambdas, h0_pdf, label="$H_0$ case")
    ax[0].plot(lambdas, h1_pdf, label="$H_1$ case")
    ax[0].legend(loc="best")
    ax[0].set_title("PDF of H0 and H1 Cases")
    ax[0].set_xlabel(r"Threshold ($\lambda_0$)")
    ax[0].set_ylabel("Probability")

    ax[1].plot(lambdas, pfa_vs_lambda, label="$P_{FA}$", color="r")
    ax[1].plot(lambdas, pd_vs_lambda, label="$P_D$", color="b")
    ax[1].legend(loc="best")
    ax[1].set_title(r"$P_D$ and $P_{FA}$ versus Threshold ($\lambda$)")
    ax[1].set_xlabel(r"Threshold ($\lambda_0$)")
    ax[1].set_ylabel("Probability")

    plt.show()


if __name__ == "__main__":
    num_iters: int = 100_000

    μ0: float = 86
    σ0: float = 2
    h0_samps: np.ndarray = np.random.normal(μ0, σ0, num_iters)

    μ1: float = 91
    σ1: float = 2.5
    h1_samps: np.ndarray = np.random.normal(μ1, σ1, num_iters)

    l_88 = 88
    pfa_88 = np.mean([i > l_88 for i in h0_samps])
    pd_88 = np.mean([i > l_88 for i in h1_samps])
    print(f"threshold is {l_88}")
    print(f"H0: λ>λ0: {pfa_88}")
    print(f"H1: λ>λ0: {pd_88}")
    print(f"PFA: {1 - pfa_88}")
    print("\n")

    pfas = [0.5, 0.1, 0.05, 0.01]
    for pfa in pfas:
        # Threshold where `1-pfa` samples in H0 are below
        lambda_0 = np.quantile(h0_samps, 1 - pfa)
        # Find the number of H0 samples above the threshold (PFA)
        pfa_check = np.mean([i > lambda_0 for i in h0_samps])
        # Find the number of H1 samples above the threshold (PD)
        pd_check = np.mean([i > lambda_0 for i in h1_samps])

        pd_calc = calculate_pd(pfa, h0_samps, h1_samps)

        print("\n")
        print(f"Threshold: {lambda_0}")
        print(f"PFA: {pfa} | ({pfa_check})")
        print(f"PD: {pd_check} (pd_calc)")

    pfa = 0.5
    lambda_n5 = np.quantile(h0_samps, 0.5)
    print(lambda_n5)

    plot_pd_and_pfa_versus_threshold(h0_samps, h1_samps)
