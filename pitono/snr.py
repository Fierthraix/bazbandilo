#!/usr/bin/env python3
from bazbandilo import (
    tx_bpsk,
    tx_cdma_bpsk,
    tx_cdma_qpsk,
    tx_qpsk,
    # tx_ofdm_bpsk,
    tx_ofdm_qpsk,
    random_data,
    awgn,
)

import numpy as np
from typing import Dict, List

##
# Goals:
#   1. Check that I'm calculating the SNR correctly.
##

if __name__ == "__main__":
    NUM_BITS = 2**16

    data = random_data(NUM_BITS)

    sigs: Dict[str, List[complex]] = {
        "bpsk_1": tx_bpsk(data),
        "qpsk_1": tx_qpsk(data),
        "cdma_bpsk_16": tx_cdma_bpsk(data, 16),
        "cdma_bpsk_64": tx_cdma_bpsk(data, 64),
        "cdma_qpsk_16": tx_cdma_qpsk(data, 16),
        "cdma_qpsk_64": tx_cdma_qpsk(data, 64),
        "ofdm_qpsk_64": tx_ofdm_qpsk(data, 64, 0),
        "ofdm_qpsk_32": tx_ofdm_qpsk(data, 32, 0),
        "ofdm_qpsk_32_half": tx_ofdm_qpsk(data, 32, 16),
    }

    sigs["bpsk_16"] = np.repeat(sigs["bpsk_1"], 16)
    sigs["bpsk_32"] = np.repeat(sigs["bpsk_1"], 32)
    sigs["bpsk_64"] = np.repeat(sigs["bpsk_1"], 64)
    sigs["qpsk_16"] = np.repeat(sigs["qpsk_1"], 16)

    snr: float = 6

    ###
    # New test to calculate SNR correctly.
    ###
    for name, sig in sigs.items():
        num_samples = len(sig)

        E_s: float = sum(abs(s_i) ** 2 for s_i in sig)

        σ = np.sqrt(E_s / (2 * num_samples * snr))

        ω = awgn(np.zeros(num_samples), σ)

        r = np.array(sig) + np.array(ω)

        E_ω: float = sum(abs(ω_i) ** 2 for ω_i in ω)
        E_r: float = sum(abs(r_i) ** 2 for r_i in r)

        snr_calc = E_s / E_ω

        print(name)

        np.testing.assert_almost_equal(np.log10(E_s + E_ω), np.log10(E_r), decimal=1)

        np.testing.assert_almost_equal(snr_calc, snr, decimal=1)
