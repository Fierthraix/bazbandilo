use crate::{bit_to_nrz, Bit};
use num::complex::Complex;

pub fn tx_bpsk_signal<I: Iterator<Item = Bit>>(message: I) -> impl Iterator<Item = Complex<f64>> {
    message.map(|bit| Complex::<f64>::new(bit_to_nrz(bit), 0f64))
}

pub fn rx_bpsk_signal<I: Iterator<Item = Complex<f64>>>(message: I) -> impl Iterator<Item = Bit> {
    message.map(|sample| sample.re >= 0f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate itertools;
    extern crate rand;
    extern crate rand_distr;
    use crate::Rng;

    #[test]
    fn baseband() {
        let mut rng = rand::thread_rng();
        let num_bits = 9001;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let bpsk_tx: Vec<Complex<f64>> = tx_bpsk_signal(data_bits.iter().cloned()).collect();
        let bpsk_rx: Vec<Bit> = rx_bpsk_signal(bpsk_tx.iter().cloned()).collect();
        assert_eq!(data_bits, bpsk_rx);
    }
}
