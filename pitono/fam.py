#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from ssca import plot_ssca_triangle, plot_ssca_diamond


def get_bpsk_signal() -> np.array:
    N = 100000  # number of samples to simulate
    f_offset = 0.2  # Hz normalized
    sps = 20  # cyclic freq (alpha) will be 1/sps or 0.05 Hz normalized
    num_symbols = int(np.ceil(N / sps))
    symbols = np.random.randint(0, 2, num_symbols) * 2 - 1  # random 1's and -1's

    pulse_train = np.zeros(num_symbols * sps)
    pulse_train[::sps] = symbols  # easier explained by looking at an example output
    print(pulse_train[0:96].astype(int))

    # Raised-Cosine Filter for Pulse Shaping
    beta = 0.3  # roll-off parameter (avoid exactly 0.2, 0.25, 0.5, and 1.0)
    num_taps = 101  # somewhat arbitrary
    t = np.arange(num_taps) - (num_taps - 1) // 2
    h = (
        np.sinc(t / sps)
        * np.cos(np.pi * beta * t / sps)
        / (1 - (2 * beta * t / sps) ** 2)
    )  # RC equation
    bpsk = np.convolve(pulse_train, h, "same")  # apply the pulse shaping

    bpsk = bpsk[:N]  # clip off the extra samples
    bpsk = bpsk * np.exp(
        2j * np.pi * f_offset * np.arange(N)
    )  # Freq shift up the BPSK, this is also what makes it complex
    noise = np.random.randn(N) + 1j * np.random.randn(N)  # complex white Gaussian noise
    samples = bpsk + 0.1 * noise  # add noise to the signal
    return samples


def fam(samples, alphas=np.arange(0, 0.5, 0.001), psd: bool = False) -> np.ndarray:

    Nw = 256  # window length
    N = len(samples)  # signal length
    window = np.hanning(Nw)

    X = np.fft.fftshift(np.fft.fft(samples))  # FFT of entire signal

    num_freqs = int(np.ceil(N / Nw))  # freq resolution after decimation
    SCF = np.zeros((len(alphas), num_freqs), dtype=complex)
    for i, alpha in enumerate(alphas):
        shift = int(alpha * N / 2)
        SCF_slice = np.roll(X, -shift) * np.conj(np.roll(X, shift))
        # apply window and decimate by Nw
        SCF[i, :] = np.convolve(SCF_slice, window, mode="same")[::Nw]
    # SCF = np.abs(SCF)
    if not psd:
        # null out alpha=0 which is just the PSD of the signal, it throws off the dynamic range
        SCF[0, :] = 0
    return SCF


def tsm(samples, alphas=np.arange(0, 0.5, 0.001)):
    Nw = 256  # window length
    N = len(samples)  # signal length
    Noverlap = int(2 / 3 * Nw)  # block overlap
    num_windows = int((N - Noverlap) / (Nw - Noverlap))  # Number of windows
    window = np.hanning(Nw)

    SCF = np.zeros((len(alphas), Nw), dtype=complex)
    for ii in range(len(alphas)):  # Loop over cyclic frequencies
        neg = samples * np.exp(-1j * np.pi * alphas[ii] * np.arange(N))
        pos = samples * np.exp(1j * np.pi * alphas[ii] * np.arange(N))
        for i in range(num_windows):
            pos_slice = window * pos[i * (Nw - Noverlap) : i * (Nw - Noverlap) + Nw]
            neg_slice = window * neg[i * (Nw - Noverlap) : i * (Nw - Noverlap) + Nw]
            SCF[ii, :] += np.fft.fft(neg_slice) * np.conj(
                np.fft.fft(pos_slice)
            )  # Cross Cyclic Power Spectrum
    SCF = np.fft.fftshift(SCF, axes=1)  # shift the RF freq axis
    SCF = np.abs(SCF)
    # null out alpha=0 which is just the PSD of the signal, it throws off the dynamic range
    SCF[0, :] = 0
    return SCF


if __name__ == "__main__":
    alphas = np.arange(0, 0.5, 0.001)
    extent = (-0.5, 0.5, float(np.max(alphas)), float(np.min(alphas)))

    from bazbandilo import ssca, fam as fam_rs

    sig = get_bpsk_signal()
    SCF = fam(sig, alphas, psd=True)

    plot_ssca_diamond(SCF, log=False)

    scfrs = np.abs(fam_rs(sig, psd=True))

    plot_ssca_diamond(scfrs, log=False)

    sxf = ssca(sig, n=len(sig) - 256, np=256, map_output=True)
    plot_ssca_triangle(sxf, log=False)

    plt.show()
