#!/usr/bin/env python3
from bazbandilo import (
    tx_bpsk,
    random_data,
    awgn,
)

import numpy as np
import matplotlib.pyplot as plt


##
# Goals:
#   1. draw the FFT output of various comms schemes
#   2. Find out what filter parameters I need.
##

if __name__ == "__main__":
    # NUM_BITS = 2**16
    NUM_BITS = 4096

    data = random_data(NUM_BITS)

    bpsk_1 = tx_bpsk(data)

    bpsk_16 = np.repeat(bpsk_1, 16)
    bpsk_32 = np.repeat(bpsk_1, 32)
    bpsk_64 = np.repeat(bpsk_1, 64)

    for bpsk_sig in (bpsk_1, bpsk_16, bpsk_32, bpsk_64):

        eb: float = sum(abs(s_i) ** 2 for s_i in bpsk_sig) / NUM_BITS
        ebn0: float = 6
        n0: float = np.sqrt(eb / (2 * ebn0))

        fft = np.fft.fft(awgn(bpsk_sig, n0), len(bpsk_sig))
        # fft = np.fft.fftshift(fft)

        Δ = NUM_BITS
        midpoint = len(fft) // 2

        if len(fft) > NUM_BITS:
            # fft[: midpoint - Δ] = fft[midpoint + Δ :] = 0
            fft[Δ:len(fft) - Δ] = 0

        fig, ax = plt.subplots()

        num = int(len(fft) / NUM_BITS)

        ax.plot(fft, label=f"len={len(fft)} ({NUM_BITS}) ({np.log2(len(fft))})")
        ax.legend(loc="best")
        fig.suptitle(f"BPSK-{num}")

    plt.show()
