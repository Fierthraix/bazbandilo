use crate::{angle_diff, iter::Iter, linspace, Bit};
use std::f64::consts::PI;

use num::Zero;
use num_complex::Complex;

pub fn tx_bfsk_signal<I: Iterator<Item = Bit>>(
    message: I,
    samples_per_symbol: usize,
) -> impl Iterator<Item = Complex<f64>> {
    let degs_p: Vec<f64> = linspace(0f64, 2f64 * PI, samples_per_symbol)
        .take(samples_per_symbol)
        .collect();
    let degs_n: Vec<f64> = [2f64 * PI]
        .into_iter()
        .chain(degs_p.iter().cloned().rev())
        .take(samples_per_symbol)
        .collect();

    fn mm(d: f64) -> Complex<f64> {
        Complex::new(0f64, -d).exp()
    }

    message.flat_map(move |bit| {
        if bit {
            degs_n.clone().into_iter().map(mm)
        } else {
            degs_p.clone().into_iter().map(mm)
        }
    })
}

pub fn rx_bfsk_signal<I: Iterator<Item = Complex<f64>>>(
    signal: I,
    samples_per_symbol: usize,
) -> impl Iterator<Item = Bit> {
    signal.chunks(samples_per_symbol).map(|symbol| {
        symbol
            .iter()
            .zip(symbol[1..].iter())
            .map(|(&w1, &w2)| angle_diff(w1, w2))
            .sum::<f64>()
            .is_sign_positive()
    })
}

pub fn tx_mfsk_signal<I: Iterator<Item = Bit>>(
    message: I,
    samples_per_symbol: usize,
) -> impl Iterator<Item = Complex<f64>> {
    let _ = samples_per_symbol;
    message.map(|_| Complex::zero())
}

pub fn rx_mfsk_signal<I: Iterator<Item = Complex<f64>>>(
    signal: I,
    samples_per_symbol: usize,
) -> impl Iterator<Item = Bit> {
    let _ = samples_per_symbol;
    signal.map(|_| true)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    extern crate rand_distr;
    use crate::Rng;

    #[test]
    fn bfsk() {
        let mut rng = rand::thread_rng();
        let num_bits = 9001;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let samples_per_symbol = 1000;

        let bfsk_tx: Vec<Complex<f64>> =
            tx_bfsk_signal(data_bits.iter().cloned(), samples_per_symbol).collect();
        let bfsk_rx: Vec<Bit> =
            rx_bfsk_signal(bfsk_tx.iter().cloned(), samples_per_symbol).collect();
        assert_eq!(data_bits, bfsk_rx);
    }
}
