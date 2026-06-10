#!/usr/bin/env python3
from bazbandilo import tx_bpsk, awgn, tx_qpsk, random_data
from typing import Iterable, List
import matplotlib.pyplot as plt
from random import choice
import numpy as np
from scipy.stats import norm, normaltest


def avg_energy(signal: Iterable[float]) -> float:
    """Square, integrate, and average."""
    return sum(s_i**2 for s_i in signal) / len(signal)


def scale(vec: Iterable[float], scalar) -> List[float]:
    return [scalar * v_i for v_i in vec]


def neyman_pearson(signal: Iterable[float], n0: float, α: float = 0.05) -> bool:
    """Do a neyman-pearson test on the signal, where H0 is the N0"""
    mu0 = 0
    mu1 = np.mean(signal)
    sigma0 = n0
    sigma1 = np.std(signal)
    print(f"μ = {mu1} || σ = {sigma1}")

    # likelihood_ratio = np.prod(
    #     norm.pdf(signal, mu1, sigma1) / np.prod(norm.pdf(signal, mu0, sigma0))
    # )
    p0 = np.prod(norm.pdf(signal, mu0, sigma0))
    p1 = np.prod(norm.pdf(signal, mu1, sigma1))
    likelihood_ratio = p0 / p1

    threshold = norm.ppf(1 - α)
    print(f"p0: {p0} || p1: {p1} || {likelihood_ratio} || k = {threshold}")
    return likelihood_ratio > threshold


def p_value(signal: Iterable[float], n0: float) -> bool:
    res = normaltest(signal)
    print(f"p-val: {res.pvalue}")
    return res.pvalue < 0.05


def test_1():
    NUM_BITS = int(1e6)
    data = [choice((True, False)) for _ in range(NUM_BITS)]

    tx_sig = tx_bpsk(data)

    N0s = list(np.arange(1e-10, 5, 0.1))
    p_vals = []
    e_vals = []
    for N0 in N0s:
        # chan_sig = awgn(np.concatenate(([0] * 4000, tx_sig, [0] * 4000)), N0)
        chan_sig = awgn(tx_sig, N0)
        res = normaltest(chan_sig)
        p_vals.append(res.pvalue)

        asdf = awgn([0 for _ in range(len(tx_sig))], N0)
        res2 = normaltest(asdf)
        e_vals.append(res2.pvalue)

    plt.plot(N0s, p_vals)
    plt.plot(N0s, e_vals)
    # plt.plot(N0s, p_vals)
    plt.show()


if __name__ == "__main__":
    # test_1()

    # Norm test on complex data?

    num_bits = 2**18
    data = random_data(num_bits)

    qpsk_sig = np.array(tx_qpsk(data))

    res_re = normaltest(qpsk_sig.real)
    res_im = normaltest(qpsk_sig.imag)
    print(res_re, res_im)

    n0 = 10
    noise_sig = np.array(awgn(np.zeros(len(qpsk_sig)), 10))
    noise_re = normaltest(noise_sig.real)
    noise_im = normaltest(noise_sig.imag)
    print(noise_re, noise_im)
