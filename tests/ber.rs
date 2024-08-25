#![allow(unused_variables, non_upper_case_globals)]
use bazbandilo::{
    bpsk::{rx_bpsk_signal, tx_bpsk_signal},
    // cdma::{rx_cdma_bpsk_signal, tx_cdma_bpsk_signal},
    db,
    erfc,
    linspace,
    ofdm::{rx_ofdm_signal, tx_ofdm_signal},
    qpsk::{rx_qpsk_signal, tx_qpsk_signal},
    undb,
    Bit,
};

use num::complex::Complex;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use rstest::*;

use util::not_inf;

#[macro_use]
mod util;

fn ber_bpsk(eb_nos: &[f64]) -> Vec<f64> {
    eb_nos
        .iter()
        .map(|eb_no| 0.5 * erfc(eb_no.sqrt()))
        .collect()
}

fn ber_qpsk(eb_nos: &[f64]) -> Vec<f64> {
    eb_nos
        .iter()
        .map(|eb_no| 0.5 * erfc(eb_no.sqrt()) - 0.25 * erfc(eb_no.sqrt()).powi(2))
        .collect()
}

// const NUM_BITS: usize = 160_000;
const NUM_BITS: usize = 9002;

fn get_data(num_bits: usize) -> Vec<Bit> {
    let mut rng = rand::thread_rng();
    (0..num_bits).map(|_| rng.gen::<Bit>()).collect()
}

#[fixture]
fn data_bits() -> Vec<Bit> {
    get_data(NUM_BITS)
}

#[fixture]
fn snrs() -> Vec<f64> {
    linspace(-26f64, 6f64, 100).map(undb).collect::<Vec<f64>>()
}

macro_rules! calculate_bers_baseband {
    ($data:expr, $tx_sig:expr, $rx:expr, $snrs:expr) => {{
        let bers: Vec<f64> = $snrs
            .par_iter()
            .map(|&snr| {
                let eb: f64 = $tx_sig
                    .iter()
                    .cloned()
                    .map(|s_i| s_i.norm_sqr())
                    .sum::<f64>()
                    / $data.len() as f64;

                let n0 = not_inf((eb / (2f64 * snr)).sqrt());
                let awgn_noise = Normal::new(0f64, n0).unwrap();

                let noisy_signal = $tx_sig
                    .iter()
                    .cloned()
                    .zip(awgn_noise.sample_iter(rand::thread_rng()))
                    .map(|(symb, noise)| symb + noise);

                $rx(noisy_signal)
                    .zip($data.iter())
                    .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                    .sum::<f64>()
                    / $data.len() as f64
            })
            .collect();
        bers
    }};
}

#[rstest]
fn bpsk_works(snrs: Vec<f64>, data_bits: Vec<Bit>) {
    // Transmit the signal.
    let bpsk_tx: Vec<Complex<f64>> = tx_bpsk_signal(data_bits.iter().cloned()).collect();

    let y = calculate_bers_baseband!(data_bits, bpsk_tx, rx_bpsk_signal, snrs);
    let y_theory: Vec<f64> = ber_bpsk(&snrs);
    let snrs_db: Vec<f64> = snrs.iter().cloned().map(db).collect();

    // ber_plot!(snrs_db, y, y_theory, "/tmp/ber_bpsk.png");
    ber_plot!(snrs, y, y_theory, "/tmp/ber_bpsk.png");

    let bpsk_rx: Vec<Bit> = rx_bpsk_signal(bpsk_tx.iter().cloned()).collect();
    assert_eq!(data_bits, bpsk_rx);
}

#[rstest]
fn qpsk_works(snrs: Vec<f64>, data_bits: Vec<Bit>) {
    // Transmit the signal.
    let qpsk_tx: Vec<Complex<f64>> = tx_qpsk_signal(data_bits.iter().cloned()).collect();

    let y = calculate_bers_baseband!(data_bits, qpsk_tx, rx_qpsk_signal, snrs);

    let y_theory: Vec<f64> = ber_qpsk(&snrs);

    ber_plot!(snrs, y, y_theory, "/tmp/ber_qpsk.png");

    let qpsk_rx: Vec<Bit> = rx_qpsk_signal(qpsk_tx.iter().cloned()).collect();
    assert_eq!(data_bits, qpsk_rx);
}

#[rstest]
fn ofdm_works(snrs: Vec<f64>, data_bits: Vec<Bit>) {
    let subcarriers = 64;
    let pilots = 12;
    let tx_sig: Vec<Complex<f64>> = tx_ofdm_signal(
        tx_bpsk_signal(data_bits.iter().cloned()),
        subcarriers,
        pilots,
    )
    .collect();

    fn rx<I: Iterator<Item = Complex<f64>>>(signal: I) -> impl Iterator<Item = Bit> {
        let subcarriers = 64;
        let pilots = 12;
        rx_bpsk_signal(rx_ofdm_signal(signal, subcarriers, pilots))
    }

    let y = calculate_bers_baseband!(data_bits, tx_sig, rx, snrs);

    let y_theory: Vec<f64> = ber_bpsk(&snrs);

    ber_plot!(snrs, y, y_theory, "/tmp/ber_ofdm.png");
}
