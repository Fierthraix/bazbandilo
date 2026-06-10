#!/usr/bin/env python3

from bazbandilo import tx_qam, random_data

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List


def animate_iq(i: List[float], q: List[float], ms=300, lim=1):
    fig, ax = plt.subplots()

    iq = ax.scatter(i[0], q[0])
    ax.set(xlim=[-lim, lim], ylim=[-lim, lim], xlabel="I", ylabel="Q")
    ax.axhline(0, ls="--", color="Black")
    ax.axvline(0, ls="--", color="Black")

    def disp_iq(frame: int):
        x = i[frame]
        y = q[frame]
        iq.set_offsets([x, y])
        return iq

    num_frames: int = min(len(i), len(q))

    ani = animation.FuncAnimation(fig=fig, func=disp_iq, frames=num_frames, interval=ms)
    plt.show()


if __name__ == "__main__":
    num_bits = 1200
    data: List[bool] = random_data(num_bits)

    m = 16

    sample_rate = 1

    qam_signal: List[float] = np.array(tx_qam(data, m))

    animate_iq(qam_signal.real, qam_signal.imag, lim=1.5, ms=50)

    plt.show()
