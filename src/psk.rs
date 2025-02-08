use std::f64::consts::PI;

use crate::{
    bit_to_nrz,
    iq_mod::{rx_iq_constellation, tx_iq_constellation},
    is_int, Bit,
};

use itertools::Itertools;
use num::complex::Complex;

pub fn tx_bpsk_signal<I: Iterator<Item = Bit>>(message: I) -> impl Iterator<Item = Complex<f64>> {
    message.map(|bit| Complex::<f64>::new(bit_to_nrz(bit), 0f64))
}

pub fn rx_bpsk_signal<I: Iterator<Item = Complex<f64>>>(message: I) -> impl Iterator<Item = Bit> {
    message.map(|sample| sample.re >= 0f64)
}

pub fn tx_qpsk_signal<I: Iterator<Item = Bit>>(message: I) -> impl Iterator<Item = Complex<f64>> {
    message.tuples().map(|(bit1, bit2)| match (bit1, bit2) {
        (true, true) => Complex::new(1f64 / 2f64.sqrt(), 1f64 / 2f64.sqrt()),
        (true, false) => Complex::new(1f64 / 2f64.sqrt(), -(1f64 / 2f64.sqrt())),
        (false, true) => Complex::new(-(1f64 / 2f64.sqrt()), 1f64 / 2f64.sqrt()),
        (false, false) => Complex::new(-(1f64 / 2f64.sqrt()), -(1f64 / 2f64.sqrt())),
    })
}

pub fn rx_qpsk_signal<I: Iterator<Item = Complex<f64>>>(message: I) -> impl Iterator<Item = Bit> {
    message.flat_map(|sample| [sample.re >= 0f64, sample.im >= 0f64].into_iter())
}

fn get_psk_symbols(m: usize) -> Vec<Complex<f64>> {
    assert!(is_int((m as f64).log(2f64)), "`m` must be a power of 2");
    (0..m)
        .map(|k| Complex::<f64>::new(0f64, (2f64 * PI * k as f64 / m as f64).exp()))
        .collect()
}

pub fn tx_mpsk_signal<I: Iterator<Item = Bit>>(
    message: I,
    m: usize,
) -> impl Iterator<Item = Complex<f64>> {
    let symbols = get_psk_symbols(m);
    tx_iq_constellation(message, m, symbols)
}

pub fn rx_mpsk_signal<I: Iterator<Item = Complex<f64>>>(
    signal: I,
    m: usize,
) -> impl Iterator<Item = Bit> {
    let symbols = get_psk_symbols(m);
    rx_iq_constellation(signal, m, symbols)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate itertools;
    use crate::random_bits;
    use rstest::rstest;

    #[test]
    fn bpsk() {
        let num_bits = 9001;
        let data_bits: Vec<Bit> = random_bits(num_bits);

        let bpsk_tx: Vec<Complex<f64>> = tx_bpsk_signal(data_bits.iter().cloned()).collect();
        let bpsk_rx: Vec<Bit> = rx_bpsk_signal(bpsk_tx.iter().cloned()).collect();
        assert_eq!(data_bits, bpsk_rx);
    }

    #[test]
    fn qpsk() {
        let num_bits = 9002;
        let data_bits: Vec<Bit> = random_bits(num_bits);

        let qpsk_tx: Vec<Complex<f64>> = tx_qpsk_signal(data_bits.iter().cloned()).collect();

        let qpsk_rx: Vec<Bit> = rx_qpsk_signal(qpsk_tx.iter().cloned()).collect();

        assert_eq!(data_bits, qpsk_rx);
    }

    #[rstest]
    #[case(4)]
    #[case(16)]
    #[case(64)]
    #[case(256)]
    #[case(1024)]
    #[case(4096)]
    fn test_mpsk_works(#[case] m: usize) {
        let num_bits = 10 * (m as f64).log2() as usize;
        let data_bits: Vec<Bit> = random_bits(num_bits);

        let mpsk_tx: Vec<Complex<f64>> = tx_mpsk_signal(data_bits.iter().cloned(), m).collect();
        let mpsk_rx: Vec<Bit> = rx_mpsk_signal(mpsk_tx.iter().cloned(), m).collect();
        println!(
            "num_mpsk: {} || num_bits: {} || num_symbols: {}",
            m,
            num_bits,
            mpsk_tx.len()
        );
        assert_eq!(mpsk_tx.len() as f64, (num_bits as f64) / (m as f64).log2());
        assert_eq!(data_bits.len(), mpsk_rx.len());
        assert_eq!(data_bits, mpsk_rx);
    }
}
