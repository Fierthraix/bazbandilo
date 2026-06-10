#!/usr/bin/env python3
from bazbandilo import awgn, tx_bpsk, tx_qpsk, tx_cdma_bpsk, ssca
from util import timeit
from ssca import (
    ssca as ssca_py,
    # plot_ssca_diamond,
    plot_ssca_triangle,
    max_cut,
    dcs,
)

import numpy as np
import random
from typing import List


if __name__ == "__main__":

    NUM_BITS: int = int(1e6)

    def rand_data(num_bits: int) -> List[bool]:
        return [random.choice((True, False)) for _ in range(num_bits)]

    N0 = 0.5
    data: List[bool] = rand_data(NUM_BITS)
    bpsk: List[complex] = tx_bpsk(data)
    # bpsk_awgn: List[complex] = awgn(bpsk, N0)
    bpsk_awgn: List[complex] = bpsk

    cdma_bpsk_awgn = tx_cdma_bpsk(data)

    qpsk: List[complex] = tx_qpsk(data)
    qpsk_awgn: List[complex] = awgn(qpsk, N0)

    # N = 4096
    N = 32768

    Np = 256
    # Np = 64

    signals = (awgn(np.zeros(len(bpsk)), N0), bpsk_awgn, cdma_bpsk_awgn)
    # signals = (awgn(np.zeros(len(bpsk)), N0),)
    # signals = (bpsk_awgn,)
    map_output = True
    with timeit("Rust") as _:
        for sig in signals:
            sx = ssca(sig, N, Np, map_output=map_output)
            mc = max_cut(sx)
            dc = dcs(sx)
            # plot_ssca_diamond(sx)
            plot_ssca_triangle(sx)

    with timeit("Python") as _:
        for sig in signals:
            sx = ssca_py(sig, N, Np, map_output=map_output)
            mc = max_cut(sx)
            dc = dcs(sx)
            # plot_ssca_diamond(sx)
            plot_ssca_triangle(sx)

    import matplotlib.pyplot as plt

    plt.show()
