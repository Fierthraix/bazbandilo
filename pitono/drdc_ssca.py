#!/usr/bin/env python3
from bazbandilo import max_cut_detector, dcs_detector, energy_detector

import matplotlib.pyplot as plt
import numpy as np
from typing import List

from util import timeit
from ssca import plot_lambda, plot_ssca_diamond, plot_ssca_triangle, max_cut, dcs


def dcs2(sig: List[complex]) -> np.ndarray:
    # λ = np.sum(np.abs(sx[1:]) ** 2, axis=0) / np.abs(sx[0, :]) ** 2
    N = 2**16
    Np = 64
    sx = ssca(sig, N, Np, map_output=True)

    # _ = sx[sx.shape[0] // 2, :]

    # λ = np.sum(np.abs(sx) ** 2, axis=0) / np.abs(sx[sx.shape[0] // 2, :]) ** 2
    λ = np.sum(np.abs(sx) ** 2, axis=1) / np.abs(sx[:, sx.shape[1] // 2]) ** 2
    # λ = np.sum(np.abs(sx) ** 2, axis=1) * np.abs(sx[:, sx.shape[1] // 2]) ** 2

    return 10 * np.log10(λ)
    # return λ


if __name__ == "__main__":
    from bazbandilo import awgn, tx_bpsk, tx_qpsk, tx_cdma_bpsk, random_data, ssca

    NUM_BITS = 2**16
    data: List[bool] = random_data(NUM_BITS)
    samp_rate = 1e5  # 100 kHz

    baud_a = 4800  # 4800 Baud
    fc_a = 23e3  # 23 kHz
    t_a = np.linspace(0, NUM_BITS / baud_a, int(NUM_BITS * samp_rate / baud_a), endpoint=False)
    d_a = np.repeat(tx_bpsk(data), samp_rate / baud_a)
    sig_a = d_a * np.exp(2j * np.pi * fc_a * t_a)[:len(d_a)]

    baud_b = 1e4  # 4800 Baud
    fc_b = 23.5e3  # 23.5 kHz
    t_b = np.linspace(0, NUM_BITS / baud_b, int(NUM_BITS * samp_rate / baud_b))
    d_b = np.repeat(tx_bpsk(data), samp_rate / baud_b)
    sig_b = d_b * np.exp(2j * np.pi * fc_b * t_b)[:len(d_b)]

    baud_c = 9e3  # 4800 Baud
    fc_c = 29.5e3  # 29.5 kHz
    t_c = np.linspace(0, NUM_BITS / baud_c, int(NUM_BITS * samp_rate / baud_c))
    d_c = np.repeat(tx_bpsk(data), samp_rate / baud_c)
    sig_c = d_c * np.exp(2j * np.pi * fc_c * t_c)[:len(d_c)]

    N0 = 0.5
    sig_n0 = np.array(awgn(np.zeros(len(sig_a)), N0))

    N5 = 5
    sig_n5 = np.array(awgn(np.zeros(len(sig_a)), N5))

    signals = (sig_a, sig_b, sig_c, sig_n0, sig_n5)
    # signals = (bpsk_awgn,)

    # N = 2**16
    N = 32768
    Np = 64

    def do_signal(sig: List[complex]):
        # N = floor_power_of_2(len(sig) - Np)
        with timeit("SSCA") as _:
            sx = ssca(sig, N, Np, map_output=True)
            # sx = ssca(sig, N, Np, map_output=False)
            # mc = max_cut_detector(sig)
            # dc = dcs_detector(sig)
            # mc = max_cut(sx)
            # dc = dcs(sx)
            # dc2 = dcs2(sig)
            # dc3 = dcs_detector(sig)
        plot_ssca_triangle(sx)
        plot_ssca_diamond(sx)
        # plot_lambda(mc)
        # plot_lambda(dc)
        # plot_lambda(dc2)
        # plot_lambda(dc3)

        # plt.show()

    results = [do_signal(signal) for signal in signals]

    plt.show()
