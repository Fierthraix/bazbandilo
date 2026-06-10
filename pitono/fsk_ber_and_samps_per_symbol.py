#!/usr/bin/env python3
from bazbandilo import (
    awgn,
    random_data,
    tx_bfsk,
    rx_bfsk,
)
from util import db, undb

from dataclasses import dataclass
from functools import partial
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from scipy.special import erfc
from typing import Callable, List, Optional


class Tx(Callable[[List[bool]], List[complex]]):
    """TypeClass for a Transmit Function"""

    ...


class Rx(Callable[[List[complex]], List[bool]]):
    """TypeClass for a Receive Function"""

    ...


@dataclass
class BitErrorTestResults:
    name: str
    tx_fn: Tx
    rx_fn: Rx
    snrs: List[float]
    bers: Optional[List[float]] = None
    theory_fn: Optional[Callable[[float], float]] = None

    def gen_tx_sig(self, num_bits: int = 9056) -> List[complex]:
        return self.tx_fn(random_data(num_bits))

    def get_n0s(self, num_bits: int = 9056) -> np.array:
        data: List[bool] = random_data(num_bits)
        tx_sig: List[complex] = self.tx_fn(data)
        num_bits_received: int = len(self.rx_fn(tx_sig))
        energy: float = sum(abs(s_i) ** 2 for s_i in tx_sig)
        samples_per_bit: int = len(tx_sig) / num_bits_received
        eb: float = energy / num_bits_received
        # return np.nan_to_num(np.sqrt(eb / (2 * self.snrs)) / samples_per_bit)  # CDMA hangs. FSK depends on samps/symb.
        # return np.nan_to_num(np.sqrt(eb / (2 * self.snrs)) * samples_per_bit)  # CDMA & FSK are WRONG.
        # return np.nan_to_num(np.sqrt(eb / (2 * self.snrs * samples_per_bit)))  # CDMA hangs. FSK is bad.
        # return np.nan_to_num(np.sqrt(eb / (2 * self.snrs / samples_per_bit)))  # CDMA & FSK are WRONG.

        # return np.nan_to_num(np.sqrt(eb / (2 * self.snrs * np.log2(samples_per_bit))))

        # return np.nan_to_num(eb / (2 * self.snrs))
        # return np.sqrt(np.nan_to_num(eb / (2 * self.snrs * samples_per_bit)))
        # return np.nan_to_num(eb / (2 * self.snrs)) / np.log2(samples_per_bit)
        # return np.nan_to_num(eb / (2 * self.snrs)) * samples_per_bit
        # return np.nan_to_num(eb / (2 * self.snrs * samples_per_bit))
        # return np.nan_to_num(eb / (2 * self.snrs * np.log2(samples_per_bit)))
        # """
        ebs: float = (energy / num_bits_received) * len(tx_sig)
        print(f"Eb: {eb} ({ebs}) || Samples per Bit: {samples_per_bit}")
        # return np.nan_to_num(np.sqrt(ebs / (2 * self.snrs)))

        return np.nan_to_num(np.sqrt(energy / (2 * len(tx_sig) * self.snrs)))

        # return np.nan_to_num(np.sqrt(eb / (2 * self.snrs)))
        # """

    def calc_ber(self, num_errors: int, parallel: bool = True):
        num_bits = 9056
        n0s: np.array = self.get_n0s(num_bits)

        if parallel:
            with multiprocessing.Pool() as p:
                self.bers = p.starmap(
                    get_ber,
                    [(self.tx_fn, self.rx_fn, n0, num_errors) for n0 in n0s],
                )
        else:
            self.bers = [get_ber(self.tx_fn, self.rx_fn, n0, num_errors) for n0 in n0s]

    def plot_graph(self):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.name} BER")
        ax.plot(db(self.snrs), self.bers, label="Empirical")
        if self.theory_fn:
            ax.plot(
                db(self.snrs),
                [self.theory_fn(eb_n0) for eb_n0 in self.snrs],
                label="Theory",
            )
        ax.set_yscale("log")
        ax.legend(loc="best")


def get_ber(tx_fn: Tx, rx_fn: Rx, n0: float, errors: int) -> float:
    num_errors = 0
    num_total_bits = 0
    while num_errors < errors:
        num_bits = 9056
        num_total_bits += num_bits

        data = random_data(num_bits)
        rx_data: List[bool] = rx_fn(awgn(tx_fn(data), n0))
        # rx_data: List[bool] = rx_fn(awgn2(tx_fn(data), n0))
        # assert len(data) == len(rx_data), f"Data: {len(data)} || Rx {len(rx_data)}"
        num_errors += sum(0 if tx_i == rx_i else 1 for tx_i, rx_i in zip(data, rx_data))

    return num_errors / num_total_bits


def ber_fsk(eb_n0: float) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0 / 2))


def ber_bpsk(eb_n0: float) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0))


def ber_qpsk(eb_n0: float) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0)) - 0.25 * erfc(np.sqrt(eb_n0)) ** 2


if __name__ == "__main__":
    # NUM_BITS: int = 9056
    # NUM_ERRORS: int = int(1e4)
    NUM_ERRORS: int = int(1e2)

    snrs = undb(np.linspace(-25, 6, 25))
    # snrs = undb(np.linspace(-45, 10, 25))

    # key_lens = [2**i for i in range(1, 9)]
    key_lens = [2**i for i in range(3, 7)]

    comms_schemes: List[BitErrorTestResults] = [
        BitErrorTestResults(
            f"FSK-{key_len}",
            partial(tx_bfsk, delta_f=key_len),
            partial(rx_bfsk, delta_f=key_len),
            snrs,
            theory_fn=ber_bpsk,
        )
        for key_len in key_lens
    ]

    for scheme in comms_schemes:
        print(f"Starting on {scheme.name}")
        scheme.calc_ber(NUM_ERRORS)
        # scheme.plot_graph()
        print(f"Finished with {scheme.name}")

    fig, ax = plt.subplots()
    ax.plot()
    ax.set_title("All BERs")
    ax.plot(db(snrs), [ber_bpsk(eb_n0) for eb_n0 in snrs], label="BPSK Theoretical")
    ax.plot(db(snrs), [ber_fsk(eb_n0) for eb_n0 in snrs], label="FSK Theoretical")

    for scheme in comms_schemes:
        # if scheme.bers is None:
        #     continue
        ax.plot(db(scheme.snrs), scheme.bers, label=f"{scheme.name} BER")

    ax.set_yscale("log")
    ax.legend(loc="best")

    plt.show()
