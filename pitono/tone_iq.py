#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal
from typing import List


def animate_iq(i: List[float], q: List[float], ms=300, lim=2):
    fig, ax = plt.subplots()

    iq = ax.scatter(i[0], q[0])
    ax.set(xlim=[-lim, lim], ylim=[-lim, lim], xlabel="I", ylabel="Q")

    def disp_iq(frame: int):
        x = i[frame]
        y = q[frame]
        iq.set_offsets([x, y])
        return iq

    num_frames: int = min(len(i), len(q))

    ani = animation.FuncAnimation(fig=fig, func=disp_iq, frames=num_frames, interval=ms)
    return ani


def lowpass(
    sig: List[float], cutoff: float, sample_freq: float, order=5
) -> List[float]:
    nyq = 0.5 * sample_freq
    high = cutoff / nyq
    sos = signal.butter(order, high, btype="low", output="sos")
    return signal.sosfilt(sos, sig)


def plot_line(signal: List[float], sample_rate: int, title: str = ""):
    fig, ax = plt.subplots()
    t = np.linspace(0, len(signal) / sample_rate, len(signal))
    ax.plot(t, s)
    ax.set_title(title)


def plot_fft(signal: List[float], sample_rate: int, title: str = ""):
    yf = fft(signal)
    xf = fftfreq(len(signal), 1 / sample_rate)
    fig, ax = plt.subplots()
    ax.plot(xf, np.abs(yf))
    ax.set_title(title)


if __name__ == "__main__":

    freq_high = 4e3
    freq_low = 2e3
    freq_mid = np.average([freq_high, freq_low])

    sample_rate = 2**20
    duration = 1

    # f1 = 3.141592e3
    # f1 = 2e3
    # f2 = 4e3
    f1 = 1e3
    f2 = 3e3

    num_samples = duration * sample_rate
    t = np.linspace(0, duration, num_samples)

    s1 = np.cos(2 * np.pi * f1 * t)
    s2 = np.cos(2 * np.pi * f2 * t)

    s = s1 + s2

    # Display FSK signal
    plot_line(s, sample_rate, "Signal")

    # Check FFT of signal.
    plot_fft(s, sample_rate, "Signal FFT")

    # Multiple by IF
    fm = np.average([f1, f2])
    s_if = s * np.cos(2 * np.pi * fm * t)
    plot_line(s_if, sample_rate, "IF'd Signal")

    # FFT of IF
    plot_fft(s_if, sample_rate, "IF'd Signal FFT")

    # Add in a low-pass filter.
    cutoff = np.abs(f1 - f2) * (3 / 4)
    s_if_lp = lowpass(s_if, cutoff, sample_rate)

    plot_fft(s_if_lp, sample_rate, "IF'd Signal FFT, Low Passed")

    r_i = s_if_lp.real
    r_q = s_if_lp.imag

    plt.show()
