use crate::{iter::Iter, Bit};

use num::complex::Complex;

pub fn tx_iq_constellation<I: Iterator<Item = Bit>>(
    message: I,
    m: usize,
    symbols: Vec<Complex<f64>>,
) -> impl Iterator<Item = Complex<f64>> {
    let bits_per_symbol = (m as f64).log2() as usize;

    message.chunks(bits_per_symbol).map(move |chunk| {
        let idx = chunk.iter().fold(0, |acc, &bit| (acc << 1) | bit as usize);
        symbols[idx]
    })
}

pub fn rx_iq_constellation<I: Iterator<Item = Complex<f64>>>(
    signal: I,
    m: usize,
    symbols: Vec<Complex<f64>>,
) -> impl Iterator<Item = Bit> {
    let bits_per_symbol = (m as f64).log2() as usize;
    // This is half the separation between adjacent symbols.
    let min_distance = (symbols[0] - symbols[1]).norm() / 2f64;

    signal.flat_map(move |received_symbol| {
        // Find out which symbol was transmitted
        let idx = {
            let mut smallest_distance = f64::MAX;
            let mut best_index = 0;

            // Find the minumum euclidian distance between the symbols.
            for (index, &symbol) in symbols.iter().enumerate() {
                let distance = (received_symbol - symbol).norm();

                if distance < min_distance {
                    // Symbol cannot possibly be closer to any other symbol,
                    // so we short-circuit here.
                    best_index = index;
                    break;
                } else if distance < smallest_distance {
                    smallest_distance = distance;
                    best_index = index;
                }
            }
            best_index
        };

        // Convert symbol to bits.
        (0..bits_per_symbol)
            .rev()
            .map(move |i| match (idx >> i) & 1 {
                1 => true,
                0 => false,
                _ => unreachable!(),
            })
    })
}
