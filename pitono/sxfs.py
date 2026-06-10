#!/usr/bin/env python3
from typing import List

# from ssca import dcs, max_cut, plot_ssca_diamond, plot_ssca_triangle, plot_lambda
from bazbandilo import ssca, random_data
from ssca import plot_ssca_triangle
from util import timeit

if __name__ == "__main__":
    from bazbandilo import tx_fh_ofdm_dcsk

    # SAMPLE_RATE = 48_000
    # SYMBOL_RATE = 1000
    # CARRIER_FREQ = 2500
    NUM_BITS = 2**16

    N0 = 0.5
    data: List[bool] = random_data(NUM_BITS)

    fhofdmdcsk = tx_fh_ofdm_dcsk(data)

    N = 4096
    # N = 8192
    N = 32768  # 4096
    # Np = 64
    Np = 256

    # for sig in (awgn(np.zeros(len(bpsk)), N0), bpsk, bpsk_awgn, qpsk, qpsk_awgn):
    # for sig in (awgn(np.zeros(len(bpsk)), N0), bpsk, bpsk_awgn):
    signals = (fhofdmdcsk,)

    def do_signal(sig: List[float]):
        # Np = 256
        Np = 64
        # N = floor_power_of_2(len(sig) - Np)
        N = 4096
        with timeit("SSCA") as _:
            sx = ssca(sig, N, Np, map_output=True)
            # sx = ssca(sig, N, Np, map_output=False)
            plot_ssca_triangle(sx)
            # mc = max_cut(sx)
            # dc = dcs(sx)
            # plot_ssca_diamond(sx)
            # plot_lambda(mc)
            # plot_lambda(dc)
        # plt.show()

    results = [do_signal(signal) for signal in signals]
    # with Pool(8) as p:
    #     results = p.map(do_signal, signals)

    import matplotlib.pyplot as plt

    plt.show()
