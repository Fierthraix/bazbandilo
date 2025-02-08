use crate::fftshift;
use crate::iter::Iter;
use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, FftPlanner};

#[inline]
fn get_data_subcarriers(num_subcarriers: usize, num_pilots: usize) -> Vec<usize> {
    if num_pilots.is_zero() {
        return (0..num_subcarriers).collect();
    }

    let m: usize = num_subcarriers / 2;
    let back = (num_pilots - 1) / 2;
    let front = num_pilots - 1 - back;

    [(front..m), ((m + 1)..(num_subcarriers - back))]
        .iter()
        .flat_map(|i| i.clone())
        .collect()
}

pub fn tx_ofdm_signal<I: Iterator<Item = Complex<f64>>>(
    symbols: I,
    subcarriers: usize,
    pilots: usize,
) -> impl Iterator<Item = Complex<f64>> {
    assert!(pilots < subcarriers);
    let num_data_subcarriers = subcarriers - pilots;
    // let cp_len = subcarriers / 4;

    let data_subcarriers: Vec<usize> = get_data_subcarriers(subcarriers, pilots);

    let mut fftp = FftPlanner::new();
    let fft = fftp.plan_fft_inverse(subcarriers);
    let mut fft_scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];

    let ofdm_symbols = symbols
        .wchunks(num_data_subcarriers) // S/P
        .flat_map(move |data_chunk| {
            // Insert data symbols into data carriers.
            let mut ofdm_symbol_data = vec![Complex::zero(); subcarriers];
            data_subcarriers
                .iter()
                .zip(data_chunk)
                .for_each(|(&carrier, datum)| ofdm_symbol_data[carrier] = datum);

            let mut ofdm_symbols = fftshift(&ofdm_symbol_data);
            fft.process_with_scratch(&mut ofdm_symbols, &mut fft_scratch); // IFFT

            // let cp = ofdm_symbols[subcarriers - cp_len..subcarriers]
            //     .iter()
            //     .cloned();

            // let cp_symbol: Vec<Complex<f64>> = cp
            //     .chain(ofdm_symbols.iter().cloned())
            let cp_symbol: Vec<Complex<f64>> = ofdm_symbols
                .into_iter()
                .map(|s_i| s_i / (subcarriers as f64) /*.sqrt()*/)
                .collect();

            cp_symbol
        });

    ofdm_symbols
}

pub fn rx_ofdm_signal<I: Iterator<Item = Complex<f64>>>(
    message: I,
    subcarriers: usize,
    pilots: usize,
) -> impl Iterator<Item = Complex<f64>> {
    let num_data_subcarriers = subcarriers - pilots;
    let cp_len = subcarriers / 4;

    let data_subcarriers: Vec<usize> = get_data_subcarriers(subcarriers, pilots);

    let mut fftp = FftPlanner::new();

    let fft = fftp.plan_fft_forward(subcarriers);
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];

    message
        // .wchunks(subcarriers + cp_len) // S/P
        .wchunks(subcarriers) // S/P
        .flat_map(move |ofdm_symbol_data| {
            // let mut buffer: Vec<Complex<f64>> = Vec::from(&ofdm_symbol_data[cp_len..]); // CP Removal
            let mut buffer: Vec<Complex<f64>> = ofdm_symbol_data; // CP Removal
            fft.process_with_scratch(&mut buffer, &mut scratch); // IFFT
            let demoded = fftshift(&buffer);

            let mut data_symbols = Vec::with_capacity(num_data_subcarriers);
            data_subcarriers
                .iter()
                .for_each(|&carrier| data_symbols.push(demoded[carrier]));

            // P/S
            data_symbols.into_iter()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        psk::{rx_qpsk_signal, tx_qpsk_signal},
        random_bits, Bit,
    };
    #[test]
    fn test_qpsk_ofdm() {
        let subcarriers = 64;
        let pilots = 12;
        // let pilots = 0;
        let num_bits = 103; // 64 - 12; //2080;
        let data_bits: Vec<Bit> = random_bits(num_bits);

        let tx_sig: Vec<Complex<f64>> = tx_ofdm_signal(
            tx_qpsk_signal(data_bits.iter().cloned()),
            subcarriers,
            pilots,
        )
        .collect();

        let rx_bits: Vec<Bit> =
            rx_qpsk_signal(rx_ofdm_signal(tx_sig.iter().cloned(), subcarriers, pilots)).collect();

        // assert_eq!(data_bits.len(), rx_bits.len());
        // assert_eq!(data_bits, rx_bits);
        assert_eq!(data_bits[..num_bits], rx_bits[..num_bits]);
    }
}
