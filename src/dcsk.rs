use crate::{bit_to_nrz, chaos::LogisticMap, Bit};
use itertools::Itertools;
use num_complex::Complex;

pub fn tx_dcsk_signal<I: Iterator<Item = Bit>>(message: I) -> impl Iterator<Item = Complex<f64>> {
    message
        .zip(LogisticMap::new(3.9, 0.1))
        .flat_map(|(bit, reference)| {
            [reference, reference * bit_to_nrz(bit)]
                .into_iter()
                .map(|s_i| Complex::new(s_i, 0f64))
        })
}

pub fn rx_dcsk_signal<I: Iterator<Item = Complex<f64>>>(message: I) -> impl Iterator<Item = Bit> {
    message
        .tuples()
        .map(|(reference, information)| reference.re * information.re > 0f64)
}

#[cfg(test)]
mod tests {

    use super::*;
    extern crate rand;
    use crate::dcsk::tests::rand::Rng;

    #[test]
    fn dcsk_signal() {
        let mut rng = rand::thread_rng();
        let num_bits = 9001;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let dcsk_tx: Vec<Complex<f64>> = tx_dcsk_signal(data_bits.iter().cloned()).collect();
        let dcsk_rx: Vec<Bit> = rx_dcsk_signal(dcsk_tx.iter().cloned()).collect();
        assert_eq!(data_bits, dcsk_rx);
    }
}
