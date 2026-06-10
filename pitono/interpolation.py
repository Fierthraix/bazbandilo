#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sample_rate = 100
    seconds = 1
    t1 = np.linspace(0, seconds, sample_rate * seconds)

    freq1 = 5
    sin1 = np.sin(2 * np.pi * freq1 * t1)

    plt.plot(t1, sin1, "ro")

    interp_factor = 2
    t2 = np.linspace(0, seconds, interp_factor * sample_rate * seconds)

    # sin2 = np.interp(t2, t1, sin1)
    # plt.plot(t2, sin2, 'bx')
    # plt.legend(loc='best')
    # plt.show()

    cc = [
        complex(real=np.cos(2 * np.pi * freq1 * t), imag=np.sin(2 * np.pi * freq1 * t))
        for t in t1
    ]

    sin3 = np.interp(t2, t1, cc)
    plt.plot(t2, sin3.real)  # , 'bx')
    plt.plot(t2, sin3.imag)  # , 'rx')
    plt.show()
