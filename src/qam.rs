use crate::{
    iq_mod::{rx_iq_constellation, tx_iq_constellation},
    is_int, linspace, Bit,
};
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
    tx_iq_constellation(message, m, symbols)
}

pub fn rx_qam_signal<I: Iterator<Item = Complex<f64>>>(
    signal: I,
    m: usize,
) -> impl Iterator<Item = Bit> {
    let symbols = get_qam_symbols(m);
    rx_iq_constellation(signal, m, symbols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_bits;
    use rstest::rstest;

    #[rstest]
    #[case(4)]
    #[case(16)]
    #[case(64)]
    #[case(256)]
    #[case(1024)]
    #[case(4096)]
    fn test_qam_works(#[case] m: usize) {
        let num_bits = 10 * (m as f64).log2() as usize;
        let data_bits: Vec<Bit> = random_bits(num_bits);

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
