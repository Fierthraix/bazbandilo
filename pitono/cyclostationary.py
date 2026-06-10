#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from typing import List

from util import timeit
from ssca import plot_ssca_triangle


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
    from bazbandilo import awgn, tx_bpsk, tx_cdma_bpsk, random_data, ssca

    NUM_BITS = 2**17

    fc = 0.05
    bit_rate = 1 / 10

    # samples_per_symbol = 10
    samples_per_symbol = 256
    t_max = NUM_BITS * bit_rate

    t = np.linspace(0, t_max, NUM_BITS * samples_per_symbol)

    N0 = 0.5
    data: List[bool] = random_data(NUM_BITS)

    bpsk: List[complex] = tx_bpsk(data)
    bpsk_sig: List[float] = (
        np.repeat(bpsk, samples_per_symbol) * np.exp(2j * np.pi * fc * t)
    ).real
    bpsk_awgn: List[float] = awgn(bpsk_sig, N0)
    assert len(bpsk_sig) == NUM_BITS * samples_per_symbol

    key_len = 16
    cdma: List[complex] = tx_cdma_bpsk(data, key_len)
    cdma_sig: List[float] = np.repeat(cdma, samples_per_symbol / key_len) * np.exp(2j * np.pi * fc * t)
    cdma_awgn: List[float] = awgn(cdma_sig, N0)
    assert len(cdma) == NUM_BITS * samples_per_symbol / key_len

    key_len = 64
    cdma: List[complex] = tx_cdma_bpsk(data, key_len)
    cdma_sig: List[float] = np.repeat(cdma, samples_per_symbol / key_len) * np.exp(2j * np.pi * fc * t)
    cdma_awgn_64: List[float] = awgn(cdma_sig, N0)
    # assert len(cdma) == NUM_BITS * samples_per_symbol / key_len

    noise = awgn(np.zeros(len(bpsk_sig)), N0)
    signals = (noise, bpsk_awgn, cdma_awgn, cdma_awgn_64)

    N = 2**16
    Np = 64

    def do_signal(sig: List[complex]):
        with timeit("SSCA") as _:
            sx = ssca(sig, N, Np, map_output=True)
        plot_ssca_triangle(sx)

    results = [do_signal(signal) for signal in signals]

    N = 4096
    Np = 64

    n0s = [1e-4, 1, 1e4]
    for n0 in n0s:
        sig = awgn(np.zeros(N+Np), n0)
        sxf = ssca(sig, N, Np, map_output=True)
        plot_ssca_triangle(sxf)

    plt.show()
