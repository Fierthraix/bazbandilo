use crate::{chaos::LogisticMap, Bit};

use num_complex::Complex;

pub fn tx_csk_signal<I: Iterator<Item = Bit>>(message: I) -> impl Iterator<Item = Complex<f64>> {
    let mu = 3.9;
    let x0_1 = 0.1;
    let x0_2 = 0.15;
    message
        .zip(LogisticMap::new(mu, x0_1))
        .zip(LogisticMap::new(mu, x0_2))
        .map(|((bit, chaos_1), chaos_2)| Complex::new(if bit { chaos_1 } else { chaos_2 }, 0f64))
}

pub fn rx_csk_signal<I: Iterator<Item = Complex<f64>>>(message: I) -> impl Iterator<Item = Bit> {
    let mu = 3.9;
    let x0_1 = 0.1;
    let x0_2 = 0.15;
    message
        .zip(LogisticMap::new(mu, x0_1))
        .zip(LogisticMap::new(mu, x0_2))
        .map(|((sample, chaos_1), chaos_2)| (sample - chaos_1).norm() < (sample - chaos_2).norm())
}

#[cfg(test)]
mod tests {

    use super::*;
    extern crate rand;
    use crate::Rng;

    #[test]
    fn csk() {
        let mut rng = rand::thread_rng();
        let num_bits = 9001;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let csk_tx: Vec<Complex<f64>> = tx_csk_signal(data_bits.iter().cloned()).collect();
        let csk_rx: Vec<Bit> = rx_csk_signal(csk_tx.iter().cloned()).collect();
        assert_eq!(data_bits, csk_rx);
    }
}
