#!/usr/bin/env python3

from bazbandilo import (
    awgn_complex,
    random_data,
    tx_bpsk,
    tx_bfsk,
    tx_cdma_bpsk,
    rx_bpsk,
    rx_bfsk,
    rx_cdma_bpsk,
)
from util import db, undb

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from scipy.special import erfc
from typing import List


def energy(signal: List[float]) -> float:
    return sum(abs(s_i) ** 2 for s_i in signal)


def test_1():
    data = [True, False]
    num_bits: int = len(data)

    bpsk = tx_bpsk(data)
    fsk = tx_bfsk(data, 100)

    energy_bpsk: float = energy(bpsk)
    energy_fsk: float = energy(fsk)

    samples_per_bit_bpsk = len(bpsk) / num_bits
    samples_per_bit_fsk = len(fsk) / num_bits

    eb_bpsk = energy_bpsk / num_bits
    eb_fsk = energy_fsk / num_bits

    samp_per_bit_bpsk = len(bpsk) / num_bits
    samp_per_bit_fsk = len(fsk) / num_bits

    eb_bpsk = energy_bpsk / num_bits
    eb_fsk = energy_fsk / num_bits

    ebs_bpsk = energy_bpsk / len(bpsk)
    ebs_fsk = energy_fsk / len(fsk)

    print(
        f"Eb BPSK: {eb_bpsk} | samples_per_bit: {samp_per_bit_bpsk} | Ebs: {ebs_bpsk}"
    )
    print(f"Eb FSK: {eb_fsk} | samples_per_bit: {samp_per_bit_fsk} | Ebs: {ebs_fsk}")


def calc_ber(
    tx_fn, rx_fn, n0s: List[float], num_errors: int = int(1e3), parallel: bool = True
):
    if parallel:
        with multiprocessing.Pool() as p:
            bers = p.starmap(
                get_ber,
                [(tx_fn, rx_fn, n0, num_errors) for n0 in n0s],
            )
    else:
        bers = [get_ber(tx_fn, rx_fn, n0, num_errors) for n0 in n0s]

    return bers


def get_ber(tx_fn, rx_fn, n0: float, errors: int) -> float:
    num_errors = 0
    num_total_bits = 0
    while num_errors < errors:
        num_bits = 9056
        num_total_bits += num_bits

        data = random_data(num_bits)
        rx_data: List[bool] = rx_fn(awgn_complex(tx_fn(data), n0))
        num_errors += sum(0 if tx_i == rx_i else 1 for tx_i, rx_i in zip(data, rx_data))

    return num_errors / num_total_bits


def ber_bpsk(eb_n0: float) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0))


def bpsk_tx(data: List[bool]) -> List[complex]:
    return np.repeat(tx_bpsk(data), 16)


def cdma_tx(data: List[bool]) -> List[complex]:
    return tx_cdma_bpsk(data)


def fsk_tx(data: List[bool]) -> List[complex]:
    return tx_bfsk(data, 16)


def bpsk_rx(signal: List[complex]) -> List[bool]:
    return rx_bpsk(signal[::16])


def cdma_rx(signal: List[complex]) -> List[bool]:
    return rx_cdma_bpsk(signal)


def fsk_rx(signal: List[complex]) -> List[bool]:
    return rx_bfsk(signal, 16)


def test_2():
    data = [True, False]
    num_bits: int = len(data)

    assert len(bpsk_tx(data)) == len(cdma_tx(data)) == len(fsk_tx(data))

    def calc_eb(signal: List[float]) -> float:
        energy = sum(abs(s_i) ** 2 for s_i in signal)
        return energy / num_bits

    eb_bpsk = calc_eb(bpsk_tx(data))
    eb_cdma = calc_eb(cdma_tx(data))
    eb_fsk = calc_eb(fsk_tx(data))

    assert eb_bpsk == eb_cdma == eb_fsk
    eb = eb_bpsk

    snrs = undb(np.linspace(-25, 6, 25))

    n0s = np.nan_to_num(np.sqrt(eb / (2 * snrs)))

    bpsk_bers = calc_ber(bpsk_tx, bpsk_rx, n0s)
    cdma_bers = calc_ber(cdma_tx, cdma_rx, n0s)
    fsk_bers = calc_ber(fsk_tx, fsk_rx, n0s)

    fig, ax = plt.subplots()
    ax.plot(db(snrs), [ber_bpsk(snr) for snr in snrs], label="BPSK Theory")
    ax.plot(db(snrs), bpsk_bers, label="BPSK")
    ax.plot(db(snrs), cdma_bers, label="CDMA")
    ax.plot(db(snrs), fsk_bers, label="FSK")
    ax.set_yscale("log")
    ax.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    # test_1()
    test_2()
