use crate::{chaos::LogisticMap, fftshift, iter::Iter, no_nans, psk::tx_bpsk_signal, Bit};
use rand::prelude::*;
use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, FftPlanner};
use std::iter;

const NUM_SUBCARRIERS: usize = 8;
const CP_LEN: usize = NUM_SUBCARRIERS / 4;

const SEED: u64 = 64;

pub fn tx_fh_ofdm_dcsk_signal<I: Iterator<Item = Bit>>(
    message: I,
) -> impl Iterator<Item = Complex<f64>> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut fftp = FftPlanner::new();
    let fft = fftp.plan_fft_inverse(NUM_SUBCARRIERS);
    let mut fft_scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];

    let bpsk_symbols = tx_bpsk_signal(message);

    bpsk_symbols
        .wchunks(NUM_SUBCARRIERS - 1) // S/P
        // .zip(Chebyshev::new(0.5).chunks(NUM_SUBCARRIERS))
        .zip(LogisticMap::new(3.9, 0.1).chunks(NUM_SUBCARRIERS))
        .flat_map(move |(bpsk_symbols, chaotic_sequence)| {
            let pre_hop_symbols: Vec<Complex<f64>> = iter::once(chaotic_sequence[0])
                .map(Complex::from)
                .chain(
                    // Multiply the data symbols by the remaining chaotic sequence chunk.
                    chaotic_sequence[1..]
                        .iter()
                        .zip(bpsk_symbols.iter())
                        .map(|(&c_i, &s_i)| s_i * Complex::new(c_i, 0f64)),
                )
                .collect();
            debug_assert!(no_nans(&pre_hop_symbols), "{:?}", pre_hop_symbols);

            let freq_hopped_symbols: Vec<Complex<f64>> = {
                let mut symbols = Vec::with_capacity(NUM_SUBCARRIERS);
                let mut index: Vec<usize> = (0..NUM_SUBCARRIERS).collect();
                index.shuffle(&mut rng);
                for idx in index.into_iter() {
                    symbols.push(pre_hop_symbols[idx]);
                }
                symbols
            };
            debug_assert!(no_nans(&freq_hopped_symbols), "{:?}", freq_hopped_symbols);

            let mut ifft_symbols = fftshift(&freq_hopped_symbols);
            // let mut ifft_symbols = fftshift(&pre_hop_symbols);
            debug_assert_eq!(ifft_symbols.len(), NUM_SUBCARRIERS);
            fft.process_with_scratch(&mut ifft_symbols, &mut fft_scratch); // IFFT

            let cp = ifft_symbols[NUM_SUBCARRIERS - CP_LEN..NUM_SUBCARRIERS]
                .iter()
                .cloned();

            let cp_symbol: Vec<Complex<f64>> = cp
                .chain(ifft_symbols.iter().cloned())
                .map(|s_i| s_i / (NUM_SUBCARRIERS as f64))
                .collect();

            debug_assert!(no_nans(&cp_symbol), "{:?}", cp_symbol);

            cp_symbol
        })
}

pub fn rx_fh_ofdm_dcsk_signal<I: Iterator<Item = Complex<f64>>>(
    signal: I,
) -> impl Iterator<Item = Bit> {
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut fftp = FftPlanner::new();
    let fft = fftp.plan_fft_forward(NUM_SUBCARRIERS);
    let mut fft_scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];

    signal
        .chunks(NUM_SUBCARRIERS + CP_LEN)
        .flat_map(move |chunk| {
            let mut buffer: Vec<Complex<f64>> = Vec::from(&chunk[CP_LEN..]); // CP Removal

            fft.process_with_scratch(&mut buffer, &mut fft_scratch); // IFFT
            let demoded = fftshift(&buffer);

            let dehopped_symbols: Vec<Complex<f64>> = {
                let mut symbols = vec![Complex::zero(); NUM_SUBCARRIERS];
                let mut index: Vec<usize> = (0..NUM_SUBCARRIERS).collect();
                index.shuffle(&mut rng);
                for (&c_i, &idx) in demoded.iter().zip(index.iter()) {
                    symbols[idx] = c_i;
                }
                symbols
            };

            // TODO: Store in buffer.
            // TODO: Do something in buffer.

            let a = dehopped_symbols[0];
            // let a = demoded[0];
            let b = Vec::from(&dehopped_symbols[1..]);
            // let b = Vec::from(&demoded[1..]);

            (0..dehopped_symbols.len() - 1)
                // (0..demoded.len() - 1)
                .map(move |idx| (a.conj() * b[idx]).re.is_sign_positive())
        })
    // .skip(7)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    use crate::fh_ofdm_dcsk::tests::rand::Rng;

    #[test]
    #[ignore] // TODO: FIXME: this test.
    fn baseband() {
        let mut rng = rand::thread_rng();
        // let num_bits = 9002;
        // let num_bits = 98;
        // let num_bits = 7;
        let num_bits = 14;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let fh_ofdm_dcsk_tx: Vec<Complex<f64>> =
            tx_fh_ofdm_dcsk_signal(data_bits.iter().cloned()).collect();
        let fh_ofdm_dcsk_rx_orig: Vec<Bit> =
            rx_fh_ofdm_dcsk_signal(fh_ofdm_dcsk_tx.iter().cloned()).collect();
        let fh_ofdm_dcsk_rx: Vec<Bit> =
            Vec::from(&fh_ofdm_dcsk_rx_orig[..fh_ofdm_dcsk_rx_orig.len() - 7]);
        assert_eq!(data_bits.len(), fh_ofdm_dcsk_rx.len());
        assert_eq!(data_bits, fh_ofdm_dcsk_rx);
    }

    #[test]
    fn frequency_scrambling() {
        let mut txrng = StdRng::seed_from_u64(SEED);
        let mut rxrng = StdRng::seed_from_u64(SEED);
        let asdf: Vec<char> = "Hello World!".chars().cycle().take(12).collect();

        const LIM: usize = 12;

        let freq_hopped_symbols: Vec<char> = asdf
            .iter()
            .cloned()
            .chunks(LIM)
            .flat_map(|chunk| {
                let mut symbols: Vec<_> = Vec::with_capacity(LIM);
                let mut index: Vec<usize> = (0..LIM).collect();
                index.shuffle(&mut txrng);
                for &idx in index.iter() {
                    symbols.push(chunk[idx]);
                }
                symbols
            })
            .collect();

        let dehopped_symbols: Vec<char> = freq_hopped_symbols
            .iter()
            .cloned()
            .chunks(LIM)
            .flat_map(|chunk| {
                let mut symbols: Vec<_> = vec![' '; LIM];
                let mut index: Vec<usize> = (0..LIM).collect();
                index.shuffle(&mut rxrng);
                for (&c_i, &idx) in chunk.iter().zip(index.iter()) {
                    symbols[idx] = c_i;
                }
                symbols
            })
            .collect();

        assert_eq!(asdf, dehopped_symbols);
    }
}
