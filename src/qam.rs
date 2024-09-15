use crate::{is_int, iter::Iter, linspace, Bit};
use num_complex::Complex;

fn get_qam_symbols(m: usize) -> Vec<Complex<f64>> {
    assert!(is_int((m as f64).log(4f64)), "`m` must be a power of 4");

    let n = (m as f64).sqrt() as usize;
    linspace(-1f64, 1f64, n)
        .flat_map(|re| linspace(-1f64, 1f64, n).map(move |im| Complex::new(re, im)))
        .collect()
}

pub fn tx_qam_signal<I: Iterator<Item = Bit>>(
    message: I,
    m: usize,
) -> impl Iterator<Item = Complex<f64>> {
    let symbols = get_qam_symbols(m);
    let bits_per_symbol = (m as f64).log2() as usize;

    message.chunks(bits_per_symbol).map(move |chunk| {
        let idx = chunk.iter().fold(0, |acc, &bit| (acc << 1) | bit as usize);
        symbols[idx]
    })
}

pub fn rx_qam_signal<I: Iterator<Item = Complex<f64>>>(
    signal: I,
    m: usize,
) -> impl Iterator<Item = Bit> {
    let symbols = get_qam_symbols(m);
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

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    use crate::Rng;
    use rstest::rstest;

    #[rstest]
    #[case(4)]
    #[case(16)]
    #[case(64)]
    #[case(256)]
    #[case(1024)]
    #[case(4096)]
    fn test_qam_works(#[case] m: usize) {
        let mut rng = rand::thread_rng();
        let num_bits = 10 * (m as f64).log2() as usize;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let qam_tx: Vec<Complex<f64>> = tx_qam_signal(data_bits.iter().cloned(), m).collect();
        let qam_rx: Vec<Bit> = rx_qam_signal(qam_tx.iter().cloned(), m).collect();
        println!(
            "num_qam: {} || num_bits: {} || num_symbols: {}",
            m,
            num_bits,
            qam_tx.len()
        );
        assert_eq!(qam_tx.len() as f64, (num_bits as f64) / (m as f64).log2());
        assert_eq!(data_bits.len(), qam_rx.len());
        assert_eq!(data_bits, qam_rx);
    }
}
