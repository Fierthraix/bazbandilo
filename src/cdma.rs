use crate::{bit_to_nrz, iter::Iter, Bit};

use num_complex::Complex;

/// Transmits a DS-CDMA signal.
/// Each bit of a data signal is multiplied by a key.
/// The output rate of this function is the bitrate times the keysize.
pub fn tx_cdma<'a, I: Iterator<Item = Bit> + 'a>(
    message: I,
    key: &'a [Bit],
) -> impl Iterator<Item = Bit> + 'a {
    message
        .inflate(key.len()) // Make each bit as long as the key.
        .zip(key.iter().cycle())
        .map(|(bit, key)| bit ^ key) // XOR the bit with the entire key.
}

/// Transmits a DS-CDMA signal.
pub fn rx_cdma<'a, I: Iterator<Item = Bit> + 'a>(
    bitstream: I,
    key: &'a [Bit],
) -> impl Iterator<Item = Bit> + '_ {
    // Multiply by key, and take the average.
    bitstream
        .zip(key.iter().cycle())
        .map(|(bit, key)| bit ^ key) // XOR the bit with the entire key.
        .chunks(key.len())
        .map(move |data_bit| {
            // Now take the average of the XOR'd part.
            let trueness: usize = data_bit
                .into_iter()
                .map(|bit| if bit { 1 } else { 0 })
                .sum();

            trueness * 2 > key.len()
        })
}

/// Transmits a DS-CDMA signal.
/// Each bit of a data signal is multiplied by a key.
/// The output rate of this function is the bitrate times the keysize.
pub fn tx_cdma_bpsk_signal<'a, I: Iterator<Item = Bit> + 'a>(
    message: I,
    key: &'a [Bit],
) -> impl Iterator<Item = Complex<f64>> + 'a {
    tx_cdma(message, key).map(|bit| Complex::new(bit_to_nrz(bit), 0f64))
}

/// Transmits a DS-CDMA signal.
pub fn rx_cdma_bpsk_signal<'a, I: Iterator<Item = Complex<f64>> + 'a>(
    signal: I,
    key: &'a [Bit],
) -> impl Iterator<Item = Bit> + '_ {
    rx_cdma(signal.map(|s_i| s_i.re < 0f64), key)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    extern crate rand_distr;
    use crate::cdma::tests::rand::Rng;
    use crate::hadamard::HadamardMatrix;
    use rstest::rstest;

    #[rstest]
    #[case(2)]
    #[case(4)]
    #[case(8)]
    #[case(16)]
    #[case(256)]
    fn cdma(#[case] matrix_size: usize) {
        let num_bits = 16960;

        let walsh_codes = HadamardMatrix::new(matrix_size);
        let key: Vec<Bit> = walsh_codes.key(0).clone();

        // Data bits.
        let mut rng = rand::thread_rng();
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        // TX CDMA.
        let cdma_tx: Vec<Bit> = tx_cdma(data_bits.clone().into_iter(), &key).collect();

        let cdma_rx: Vec<Bit> = rx_cdma(cdma_tx.clone().into_iter(), &key.clone()).collect();

        assert_eq!(data_bits, cdma_rx);
    }
}
