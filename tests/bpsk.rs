use std::ffi::CString;

use bazbandilo::psk::{rx_bpsk_signal, tx_bpsk_signal};
use bazbandilo::{awgn, Bit};

use num::complex::Complex;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

#[macro_use]
mod util;

#[test]
fn bpsk_graphs() {
    let data: Vec<Bit> = vec![
        false, false, true, false, false, true, false, true, true, true,
    ];

    let tx: Vec<Complex<f64>> = tx_bpsk_signal(data.iter().cloned()).collect();

    let rx_clean: Vec<Bit> = rx_bpsk_signal(tx.iter().cloned()).collect();

    let sigma = 2f64;
    let noisy_signal: Vec<Complex<f64>> = awgn(tx.iter().cloned(), sigma).collect();

    let rx_dirty: Vec<Bit> = rx_bpsk_signal(noisy_signal.iter().cloned()).collect();

    let t: Vec<f64> = (0..tx.len())
        .map(|idx| {
            // let time_step = symbol_rate as f64 / samp_rate as f64;
            idx as f64
        })
        .collect();

    plot!(t, tx, "/tmp/bpsk_tx.png");
    plot!(t, tx, noisy_signal, "/tmp/bpsk_tx_awgn.png");

    println!("ERROR: {}", error!(rx_clean, rx_dirty));
    assert!(error!(rx_clean, rx_dirty) <= 0.2);
}
