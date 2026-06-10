#!/usr/bin/env python3

from bazbandilo import tx_qpsk, random_data

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List

if __name__ == "__main__":
    num_bits = 24
    data: List[bool] = random_data(num_bits)

    qpsk_symbols = tx_qpsk(data)

    fig, ax = plt.subplots()

    iq = ax.scatter(qpsk_symbols[0].real, qpsk_symbols[0].imag)
    lim = 2
    ax.set(xlim=[-lim, lim], ylim=[-lim, lim], xlabel="I", ylabel="Q")

    def update(frame: int):
        x = qpsk_symbols[frame].real
        y = qpsk_symbols[frame].imag
        iq.set_offsets([x, y])
        return iq

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=len(qpsk_symbols), interval=300
    )
    plt.show()
