#!/usr/bin/env python3
from bazbandilo import (
    awgn,
    # awgn2,
    energy as calc_energy,
    random_data,
    tx_bfsk,
    rx_bfsk,
    tx_bpsk,
    rx_bpsk,
    tx_qam,
    rx_qam,
    tx_qpsk,
    rx_qpsk,
    tx_cdma_bpsk,
    rx_cdma_bpsk,
    tx_cdma_qpsk,
    rx_cdma_qpsk,
    tx_ofdm_qpsk,
    rx_ofdm_qpsk,
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
    sers: Optional[List[float]] = None
    theory_fn: Optional[Callable[[float], float]] = None
    __bits_per_sample = None
    __samples_per_symbol = None

    def __get_k_and_xi(self):
        # Find out how many samples that one bit produces
        one_bit_samples: int = len(self.tx_fn([True]))

        bits: int = 1
        while True:
            # Add another bit to see if that increases the number of samples.
            n_samples: int = len(self.tx_fn([True] * (bits + 1)))
            if n_samples > one_bit_samples:
                # Adding this extra bit added another frame; abort!
                break
            else:
                # The number of symbols didn't increase, so keep adding bits.
                bits += 1

        self.__bits = bits
        self.__samples_per_symbol = one_bit_samples

    @property
    def bits_per_sample(self) -> int:
        if self.__bits_per_sample is None:
            self.__get_k_and_xi()
        return self.__bits_per_sample

    @property
    def samples_per_symbol(self) -> int:
        if self.__samples_per_symbol is None:
            self.__get_k_and_xi()
        return self.__samples_per_symbol

    def gen_tx_sig(self, num_bits: int = 9056) -> List[complex]:
        return self.tx_fn(random_data(num_bits))

    def calc_ber(self, num_errors: int, parallel: bool = True):
        num_bits = 9056
        # num_bits = 2**24
        num_bits = 2**16
        data: List[bool] = random_data(num_bits)
        tx_sig: List[complex] = self.tx_fn(data)

        energy: float = calc_energy(tx_sig)

        num_bits_received: int = len(self.rx_fn(tx_sig))
        self.eb: float = energy / num_bits_received

        # Normalize the tx function.
        num_samples: int = len(tx_sig)
        num_symbols: int = num_samples / self.samples_per_symbol
        self.es: float = energy / num_symbols

        self.bits_per_symbol = num_bits_received / num_symbols

        # n0s: np.array = np.nan_to_num(np.sqrt(1 / (2 * self.snrs)))
        n0s: np.array = np.nan_to_num(
            # np.sqrt(self.es / (2 * self.bits_per_symbol * self.snrs))
            np.sqrt(self.eb / (2 * self.snrs))
        )

        if parallel:
            with multiprocessing.Pool() as p:
                self.sers = p.starmap(
                    get_ser,
                    [
                        (
                            self.tx_fn,
                            self.rx_fn,
                            n0,
                            self.es,
                            num_errors,
                            self.samples_per_symbol,
                        )
                        for n0 in n0s
                    ],
                )
        else:
            self.sers = [
                get_ser(
                    self.tx_fn,
                    self.rx_fn,
                    n0,
                    self.es,
                    num_errors,
                    self.samples_per_symbol,
                )
                for n0 in n0s
            ]

    def plot_graph(self):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.name} BER")
        ax.plot(db(self.snrs), self.sers, label="Empirical")
        if self.theory_fn:
            ax.plot(
                db(self.snrs),
                [self.theory_fn(eb_n0) for eb_n0 in self.snrs],
                label="Theory",
            )
        ax.set_yscale("log")
        ax.legend(loc="best")


def laborero(tx_fn: Tx, rx_fn: Rx, num_bits: int, es: float, n0) -> int:
    data = random_data(num_bits)
    # tx_sig: List[complex] = np.array(tx_fn(data)) / np.sqrt(es)
    tx_sig: List[complex] = tx_fn(data)
    rx_data: List[bool] = rx_fn(awgn(tx_sig, n0))

    num_errors = sum(0 if tx_i == rx_i else 1 for tx_i, rx_i in zip(data, rx_data))
    return num_errors


def get_ser(
    tx_fn: Tx,
    rx_fn: Rx,
    n0: float,
    es: float,
    errors: int,
    samples_per_symbol: int = 1,
) -> float:
    num_errors = 0
    num_total_bits = 0
    # total_energy = 0
    # total_symbols = 0

    # with multiprocessing.Pool() as p:
    while num_errors < errors:
        num_bits = 9056

        # new_errors: List[int] = p.starmap(
        #     laborero, [(tx_fn, rx_fn, num_bits, es, n0) for _ in range(p._processes)]
        # )

        # num_total_bits += num_bits * len(new_errors)

        # num_errors += sum(new_errors)
        num_errors += laborero(tx_fn, rx_fn, num_bits, es, n0)
        num_total_bits += num_bits

    # assert np.isclose(total_energy / total_symbols, 1, atol=0.2), f"{total_energy / total_symbols}"

    return num_errors / num_total_bits


def get_ser_old(
    tx_fn: Tx, rx_fn: Rx, n0: float, es: float, errors: int, samples_per_symbol: int = 1
) -> float:
    num_errors = 0
    num_total_bits = 0
    # total_energy = 0
    # total_symbols = 0

    while num_errors < errors:
        num_bits = 9056
        num_total_bits += num_bits

        data = random_data(num_bits)

        tx_sig: List[complex] = np.array(tx_fn(data)) / np.sqrt(es)
        rx_data: List[bool] = rx_fn(awgn(tx_sig, n0))

        # total_energy += calc_energy(tx_sig)
        # total_symbols += len(tx_sig) / samples_per_symbol

        num_errors += sum(0 if tx_i == rx_i else 1 for tx_i, rx_i in zip(data, rx_data))

    # assert np.isclose(total_energy / total_symbols, 1, atol=0.2), f"{total_energy / total_symbols}"

    return num_errors / num_total_bits


def ber_fsk(eb_n0: float) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0 / 2))


def ber_bpsk(eb_n0: float) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0))


def ber_qpsk(eb_n0: float) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0)) - 0.25 * erfc(np.sqrt(eb_n0)) ** 2


def ber_qam(eb_n0: float, m: int) -> float:
    assert np.emath.logn(4, m).is_integer()
    return (
        2
        / np.log2(m)
        * (1 - 1 / np.sqrt(m))
        * erfc((3 * eb_n0 * np.log2(m)) / (2 * (m - 1)))
    )


if __name__ == "__main__":
    # NUM_BITS: int = 9056
    # NUM_ERRORS: int = int(1e5)
    NUM_ERRORS: int = 1000

    # snrs = undb(np.linspace(-25, 6, 25))
    # snrs = undb(np.linspace(-45, 10, 25))
    # snrs = undb(np.linspace(-25, 6, 25))
    snrs = undb(np.linspace(0, 10, 25))

    # def ofdm_tx(data: List[bool]) -> np.array:
    #     subcarriers = 16
    #     pilots = int(subcarriers * 0.8)
    #     return tx_ofdm_qpsk(data, subcarriers, pilots)

    comms_schemes: List[BitErrorTestResults] = [
        BitErrorTestResults(
            "BPSK",
            tx_bpsk,
            rx_bpsk,
            snrs,
            theory_fn=ber_bpsk,
        ),
        BitErrorTestResults(
            "QPSK",
            tx_qpsk,
            rx_qpsk,
            snrs,
            # theory_fn=ber_qpsk,
            theory_fn=ber_bpsk,
        ),
        BitErrorTestResults(
            "CDMA-BPSK",
            tx_cdma_bpsk,
            rx_cdma_bpsk,
            snrs,
            theory_fn=ber_bpsk,
        ),
        BitErrorTestResults(
            "CDMA-QPSK",
            tx_cdma_qpsk,
            rx_cdma_qpsk,
            snrs,
            theory_fn=ber_qpsk,
        ),
        BitErrorTestResults(
            "4QAM",
            partial(tx_qam, m=4),
            partial(rx_qam, m=4),
            snrs,
            theory_fn=partial(ber_qam, m=4),
        ),
        BitErrorTestResults(
            "16QAM",
            partial(tx_qam, m=16),
            partial(rx_qam, m=16),
            snrs,
            theory_fn=partial(ber_qam, m=16),
        ),
        BitErrorTestResults(
            "64QAM",
            partial(tx_qam, m=64),
            partial(rx_qam, m=64),
            snrs,
            theory_fn=partial(ber_qam, m=64),
        ),
        BitErrorTestResults(
            "1024QAM",
            partial(tx_qam, m=1024),
            partial(rx_qam, m=1024),
            snrs,
            theory_fn=partial(ber_qam, m=1024),
        ),
        BitErrorTestResults(
            "BFSK",
            partial(tx_bfsk, delta_f=16),
            partial(rx_bfsk, delta_f=16),
            snrs,
            theory_fn=ber_fsk,
        ),
        BitErrorTestResults(
            "OFDM",
            partial(tx_ofdm_qpsk, subcarriers=16, pilots=14),
            partial(rx_ofdm_qpsk, subcarriers=16, pilots=14),
            snrs,
            theory_fn=ber_bpsk,
        ),
    ]

    # comms_schemes = [scheme for scheme in comms_schemes if "QAM" in scheme.name]

    for scheme in comms_schemes:
        print(f"{scheme.name}: Samples per Symbol: {scheme.samples_per_symbol}")

    for scheme in comms_schemes:
        print(f"Starting {scheme.name}")
        scheme.calc_ber(NUM_ERRORS)
        # scheme.plot_graph()
        print(f"Finished {scheme.name}")

    fig, ax = plt.subplots()
    ax.plot()
    ax.set_title("All SERs")
    ax.plot(db(snrs), [ber_bpsk(eb_n0) for eb_n0 in snrs], label="BPSK Theoretical")
    # ax.plot(db(snrs), [ber_fsk(eb_n0) for eb_n0 in snrs], label="FSK Theoretical")
    # ax.plot(db(snrs), [ber_qam(eb_n0, 4) for eb_n0 in snrs], label="4QAM Theoretical")
    # ax.plot(db(snrs), [ber_qam(eb_n0, 16) for eb_n0 in snrs], label="16QAM Theoretical")
    # ax.plot(db(snrs), [ber_qam(eb_n0, 64) for eb_n0 in snrs], label="64QAM Theoretical")
    # ax.plot(
    #     db(snrs), [ber_qam(eb_n0, 1024) for eb_n0 in snrs], label="1024QAM Theoretical"
    # )

    for scheme in comms_schemes:
        # if scheme.sers is None:
        #     continue
        print(f"{scheme.name}: Eb {scheme.eb} | Es: {scheme.es}")
        ax.plot(db(scheme.snrs), scheme.sers, label=f"{scheme.name} BER")

    ax.set_yscale("log")
    ax.legend(loc="best")

    plt.show()
