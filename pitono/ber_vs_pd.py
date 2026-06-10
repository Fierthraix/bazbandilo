#!/usr/bin/env python3

from util import db, undb

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from typing import List


def logistic_curve(x: float, x0=0, L=1, k=1) -> float:
    return L / (1 + np.exp(-k * (x - x0)))


def ber(eb_n0: float, offset=0) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0 - offset))




if __name__ == '__main__':
    snrs = undb(np.linspace(-25, 6, 100))
    uncovert_log = lambda x: logistic_curve(db(x), x0=-15)
    covert_log = lambda x: logistic_curve(db(x), x0=3)


    fig, ax = plt.subplots()

    ax.plot(covert_log(snrs), ber(snrs), label="More Covert Protocol")
    ax.plot(uncovert_log(snrs), ber(snrs), label="Less Covert Protocol")

    ax.set_xlabel(r"Probability of Detection ($\mathcal{P}_D$)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_ylim([0, 0.51])
    ax.legend(loc="best")

    plt.show()
