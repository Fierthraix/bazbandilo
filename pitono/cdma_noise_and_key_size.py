#!/usr/bin/env python3
from bazbandilo import (
    awgn,
    random_data,
    tx_cdma_bpsk,
    rx_cdma_bpsk,
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
        energy: float = sum(abs(s_i) ** 2 for s_i in tx_sig)

        nb: int = len(self.rx_fn(tx_sig))
        ns: int = len(tx_sig)
        spb: float = ns / nb  # BPSK: 1, QPSK: 0.5, CDMA: 16
        bps: float = nb / ns  # BPSK: 1, QPSK: 2, CDMA: 0.0625

        # return bps * np.nan_to_num(np.sqrt(energy / (2 * self.snrs))) # All BERS too high
        # return spb * np.nan_to_num(np.sqrt(energy / (2 * self.snrs))) # ALL BERS=0.5

        # return bps * np.nan_to_num(np.sqrt(energy / (2 * nb * self.snrs))) # All BERs too low
        # return spb * np.nan_to_num(np.sqrt(energy / (2 * nb * self.snrs))) # BERs too high
        # return spb * np.nan_to_num(np.sqrt(energy / (2 * ns * self.snrs))) # ALL BERS too high

        return np.nan_to_num(np.sqrt(energy / (2 * nb * self.snrs))) # all a little worse than  BPSK (GOOD)
        # return np.nan_to_num(np.sqrt(energy / (2 * ns * self.snrs))) # all better than BPSK
        # return np.nan_to_num(np.sqrt(energy * nb / (2 * self.snrs)))  # ALL BER=0.5 (BAD)
        # return np.nan_to_num(np.sqrt(energy * ns / (2 * self.snrs))) # ALL BER=0.5 (BAD)
        # return np.nan_to_num(np.sqrt(energy * nb / (2 * ns * self.snrs)))  # ALL BERs high (BAD)
        # return np.nan_to_num(np.sqrt(energy * ns / (2 * nb * self.snrs)))  # ALL BER=0.5 (BAD)
        # return np.nan_to_num(np.sqrt(energy / (2 * nb * ns * self.snrs)))  # CDMA Hangs;

        # s/\bn[bs]\b/np.log2(&)/
        # return np.nan_to_num(np.sqrt(energy / (2 * np.log2(nb) * self.snrs))) # ALL BERS BAD
        # return np.nan_to_num(np.sqrt(energy / (2 * np.log2(ns) * self.snrs))) # ALL BERS BAD
        # return np.nan_to_num(np.sqrt(energy * np.log2(nb) / (2 * self.snrs)))  # ALL BERS BAD
        # return np.nan_to_num(np.sqrt(energy * np.log2(ns) / (2 * self.snrs))) # ALL BERS BAD
        # return np.nan_to_num(np.sqrt(energy * np.log2(nb) / (2 * np.log2(ns) * self.snrs)))  # ALL BERS BAD
        # return np.nan_to_num(np.sqrt(energy * np.log2(ns) / (2 * np.log2(nb) * self.snrs)))  # ALL BERS BAD
        # return np.nan_to_num(np.sqrt(energy / (2 * np.log2(nb) * np.log2(ns) * self.snrs)))  # ALL BERS BAD
        # return np.nan_to_num(np.sqrt(energy / (2 * nb * np.log2(ns) * self.snrs)))  # HANGS FOREVER
        # return np.nan_to_num(np.sqrt(energy * np.log2(ns) / (2 * nb * np.log2(ns) * self.snrs)))  # HANGS

        # return np.nan_to_num(np.sqrt(energy / (2 * np.log2(nb * ns) * self.snrs)))  # All BERS bad

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

        if num_total_bits >= 1e6:
            break

    return num_errors / num_total_bits


def ber_bpsk(eb_n0: float) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0))


def ber_qpsk(eb_n0: float) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0)) - 0.25 * erfc(np.sqrt(eb_n0)) ** 2


if __name__ == "__main__":
    # NUM_BITS: int = 9056
    NUM_ERRORS: int = int(1e4)
    # NUM_ERRORS: int = 10

    snrs = undb(np.linspace(-25, 6, 25))
    # snrs = undb(np.linspace(-45, 10, 25))

    # key_lens = [2**i for i in range(1, 8)]
    key_lens = [2**i for i in range(1, 6)]

    comms_schemes: List[BitErrorTestResults] = [
        BitErrorTestResults(
            f"CDMA-BPSK-{key_len}",
            partial(tx_cdma_bpsk, key_len=key_len),
            partial(rx_cdma_bpsk, key_len=key_len),
            snrs,
            theory_fn=ber_bpsk,
        )
        for key_len in key_lens
    ]

    for scheme in comms_schemes:
        print(f"Starting on {scheme.name}")
        scheme.calc_ber(NUM_ERRORS)
        print(f"Finished with {scheme.name}")

    fig, ax = plt.subplots()
    ax.plot()
    ax.set_title("All BERs")
    ax.plot(db(snrs), [ber_bpsk(eb_n0) for eb_n0 in snrs], label="BPSK Theoretical")
    for scheme in comms_schemes:
        ax.plot(db(scheme.snrs), scheme.bers, label=f"{scheme.name} BER")
    ax.set_yscale("log")
    ax.legend(loc="best")

    plt.show()
