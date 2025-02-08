use std::ffi::CString;

use bazbandilo::{fh_ofdm_dcsk::tx_fh_ofdm_dcsk_signal, random_bits, Bit};

use num::Complex;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

#[macro_use]
mod util;

#[test]
fn baseband_plot() {
    let num_bits = 50 * 9002;
    let data_bits: Vec<Bit> = random_bits(num_bits);
    // let data_bits: Vec<Bit> = vec![true, true, false, false, true, false, true];

    let tx: Vec<Complex<f64>> = tx_fh_ofdm_dcsk_signal(data_bits.iter().cloned()).collect();

    println!("{:?}", tx);

    iq_plot!(tx, "/tmp/fh_ofdm_dcsk_baseband_IQ.png");
}
