use bazbandilo::fh_ofdm_dcsk::tx_fh_ofdm_dcsk_signal;
use bazbandilo::Bit;

use num::Complex;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;

#[macro_use]
mod util;

#[test]
fn baseband_plot() {
    let mut rng = rand::thread_rng();
    let num_bits = 50 * 9002;
    let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();
    // let data_bits: Vec<Bit> = vec![true, true, false, false, true, false, true];

    let tx: Vec<Complex<f64>> = tx_fh_ofdm_dcsk_signal(data_bits.iter().cloned()).collect();

    println!("{:?}", tx);

    iq_plot!(tx, "/tmp/fh_ofdm_dcsk_baseband_IQ.png");
}
