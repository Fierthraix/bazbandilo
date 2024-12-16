#![allow(unused_macros, dead_code)]
use std::f64::consts::PI;

use num_complex::Complex;
use numpy::ndarray::Axis;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};

mod bch;
pub mod cdma;
pub mod chaos;
pub mod csk;
pub mod css;
pub mod dcsk;
pub mod fh;
pub mod fh_ofdm_dcsk;
mod filters;
pub mod fsk;
pub mod hadamard;
mod iq_mod;
pub mod iter;
pub mod ofdm;
pub mod psk;
pub mod qam;
pub mod ssca;
mod util;

use crate::{
    cdma::{rx_cdma_bpsk_signal, rx_cdma_qpsk_signal, tx_cdma_bpsk_signal, tx_cdma_qpsk_signal},
    csk::{rx_csk_signal, tx_csk_signal},
    // css::tx_css_signal,
    dcsk::{rx_dcsk_signal, rx_qcsk_signal, tx_dcsk_signal, tx_qcsk_signal},
    fh_ofdm_dcsk::{rx_fh_ofdm_dcsk_signal, tx_fh_ofdm_dcsk_signal},
    fsk::{rx_bfsk_signal, tx_bfsk_signal},
    hadamard::HadamardMatrix,
    iter::Iter,
    ofdm::{rx_ofdm_signal, tx_ofdm_signal},
    psk::{
        rx_bpsk_signal, rx_mpsk_signal, rx_qpsk_signal, tx_bpsk_signal, tx_mpsk_signal,
        tx_qpsk_signal,
    },
    qam::{rx_qam_signal, tx_qam_signal},
    ssca::{ssca_base, ssca_mapped},
};

pub type Bit = bool;

#[inline]
fn angle_diff(a: Complex<f64>, b: Complex<f64>) -> f64 {
    // Check the diff from point a to point b.
    let diff: f64 = b.arg() - a.arg();

    if diff > PI {
        diff - 2f64 * PI
    } else if diff < -PI {
        diff + 2f64 * PI
    } else {
        diff
    }
}

fn bool_to_u8(bools: &[bool]) -> u8 {
    let mut out: u8 = 0x0;
    for (i, &bool) in bools.iter().enumerate().take(std::cmp::min(bools.len(), 8)) {
        if bool {
            out |= 1 << i
        }
    }
    out
}

fn u8_to_bools(data: u8) -> [bool; 8] {
    let mut out: [bool; 8] = [false; 8];
    let data: u8 = data;
    for (i, bool) in out.iter_mut().enumerate() {
        *bool = (data & (1 << i)) != 0
    }
    out
}

fn bools_to_u8s<I: Iterator<Item = Bit>>(bools: I) -> impl Iterator<Item = u8> {
    bools.chunks(8).map(|chunk| {
        // Pad the last chunk, if neccessary.
        if chunk.len() != 8 {
            let mut last_chunk = Vec::with_capacity(8);
            last_chunk.extend_from_slice(&chunk);

            while !last_chunk.len() < 8 {
                last_chunk.push(false);
            }
            bool_to_u8(&last_chunk)
        } else {
            bool_to_u8(&chunk)
        }
    })
}

fn u8s_to_bools<I: Iterator<Item = u8>>(bytes: I) -> impl Iterator<Item = bool> {
    bytes.flat_map(u8_to_bools)
}

fn is_int(num: f64) -> bool {
    num == (num as u64) as f64
}

/// Denotes types that take on values at infinity, negative infinity, or other Not-A-Number situarions.
trait NanAble {
    fn is_nan(&self) -> bool;
}

impl NanAble for f64 {
    fn is_nan(&self) -> bool {
        self.is_subnormal()
    }
}

impl NanAble for Complex<f64> {
    fn is_nan(&self) -> bool {
        self.re.is_subnormal() || self.im.is_subnormal()
    }
}

fn no_nans<T: NanAble>(v: &[T]) -> bool {
    !v.iter().any(|v_i| v_i.is_nan())
}

#[inline]
pub fn db(x: f64) -> f64 {
    10f64 * x.log10()
}

#[inline]
pub fn undb(x: f64) -> f64 {
    10f64.powf(x / 10f64)
}

#[inline]
pub fn linspace(start: f64, stop: f64, num: usize) -> impl Iterator<Item = f64> {
    let step = (stop - start) / ((num - 1) as f64);
    (0..num).map(move |i| start + step * (i as f64))
}

#[inline]
pub fn bit_to_nrz(bit: Bit) -> f64 {
    if bit {
        1_f64
    } else {
        -1_f64
    }
}

pub fn fftshift<T: Clone>(x: &[T]) -> Vec<T> {
    let mut v = Vec::with_capacity(x.len());
    let pivot = x.len().div_ceil(2);
    v.extend_from_slice(&x[pivot..]);
    v.extend_from_slice(&x[..pivot]);
    v
}

fn fftshift_mut<T: Copy>(x: &mut [T]) {
    let pivot = x.len().div_ceil(2);
    let v: Vec<T> = Vec::from(&x[..pivot]);
    for (i, j) in (pivot..x.len()).enumerate() {
        x[i] = x[j];
    }
    for (i, item) in (0..pivot).zip(v.into_iter()) {
        x[i] = item;
    }
}

#[inline]
pub fn erf(x: f64) -> f64 {
    let t: f64 = 1f64 / (1f64 + 0.5 * x.abs());
    let tau = t
        * (-x.powi(2) - 1.26551223
            + 1.00002368 * t
            + 0.37409196 * t.powi(2)
            + 0.09678418 * t.powi(3)
            - 0.18628806 * t.powi(4)
            + 0.27886807 * t.powi(5)
            - 1.13520398 * t.powi(6)
            + 1.48851587 * t.powi(7)
            - 0.82215223 * t.powi(8)
            + 0.17087277 * t.powi(9))
        .exp();
    if x >= 0f64 {
        1f64 - tau
    } else {
        tau - 1f64
    }
}

#[inline]
pub fn erfc(x: f64) -> f64 {
    1f64 - erf(x)
}

pub fn hamming_window(length: usize) -> impl Iterator<Item = f64> {
    (0..length).map(move |i| 0.54 - 0.46 * (2f64 * PI * i as f64 / length as f64).cos())
}

pub fn hamming_window_complex(length: usize) -> impl Iterator<Item = Complex<f64>> {
    hamming_window(length).map(|h_i| Complex::new(h_i, 0f64))
}

#[inline]
pub fn ber(tx: &[Bit], rx: &[Bit]) -> f64 {
    let len: usize = std::cmp::min(tx.len(), rx.len());
    let errors: usize = tx
        .iter()
        .zip(rx.iter())
        .map(|(&ti, &ri)| if ti == ri { 0 } else { 1 })
        .sum();
    (errors as f64) / (len as f64)
}

pub fn awgn<I: Iterator<Item = Complex<f64>>>(
    signal: I,
    sigma: f64,
) -> impl Iterator<Item = Complex<f64>> {
    signal
        .zip(
            Normal::new(0f64, sigma)
                .unwrap()
                .sample_iter(rand::thread_rng()),
        )
        .zip(
            Normal::new(0f64, sigma)
                .unwrap()
                .sample_iter(rand::thread_rng()),
        )
        .map(|((sample, n_1), n_2)| sample + Complex::new(n_1, n_2))
}

#[pyfunction]
#[pyo3(name = "energy_real", signature = (signal, avg=false))]
pub fn energy_real(signal: Vec<f64>, avg: bool) -> f64 {
    let energy: f64 = signal.iter().map(|&sample| sample.abs().powi(2)).sum();
    if avg {
        energy / signal.len() as f64
    } else {
        energy
    }
}

#[pyfunction]
#[pyo3(name = "energy", signature = (signal, avg=false))]
pub fn energy_complex(signal: Vec<Complex<f64>>, avg: bool) -> f64 {
    let energy: f64 = signal.iter().map(|&sample| sample.norm_sqr()).sum();
    if avg {
        energy / signal.len() as f64
    } else {
        energy
    }
}

#[inline]
/// Calculates the energy per sample.
pub fn avg_energy(signal: &[Complex<f64>]) -> f64 {
    signal.iter().map(|&sample| sample.norm_sqr()).sum::<f64>() / signal.len() as f64
}

#[cfg(test)]
mod tests {

    use super::*;
    use rand::Rng;

    #[test]
    fn it_works() {
        let fs = 44100;
        let baud = 900;
        let nbits = 4000;
        let _f0 = 1800;
        let ns = fs / baud;
        let _n = nbits * ns;
    }

    #[test]
    fn bitstream_conversions() {
        let mut rng = rand::thread_rng();
        let num_bits = 33; // Ensure there will be padding.
        let start_data: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let u8s: Vec<u8> = bools_to_u8s(start_data.iter().cloned()).collect();
        assert_eq!(u8s.len(), 40 / 8); // Check for padding as well...

        let bits: Vec<Bit> = u8s_to_bools(u8s.iter().cloned()).collect();
        assert_eq!(start_data, bits[..num_bits])
    }
}

#[pyfunction]
fn dcs_detector(signal: Vec<Complex<f64>>) -> Vec<f64> {
    let np = 64;
    let n = 4096;
    // let sx = ssca_base(&signal, n, np);
    let sxf = ssca_mapped(&signal, n, np);

    let top = sxf.map(Complex::<f64>::norm_sqr).sum_axis(Axis(1));
    let middle: usize = sxf.shape()[1] / 2;
    let bot = sxf.column(middle).map(Complex::<f64>::norm_sqr);

    let x = top / bot;
    let y = 10f64 * x.map(|i| 10f64 * i.log10());
    y.to_vec()

    // let lambda = (top / bot)
    //     .into_iter()
    //     .max_by(|a, b| a.partial_cmp(b).unwrap())
    //     .unwrap();

    // 10f64 * lambda.log10()
}

#[pyfunction]
fn dcs_detector_sxf(sxf: PyReadonlyArray2<'_, Complex<f64>>) -> f64 {
    let sx = sxf.as_array();

    let top = sx.map(Complex::<f64>::norm_sqr).sum_axis(Axis(0));
    let bot = sx.row(0).map(Complex::<f64>::norm_sqr);

    let lambda = (top / bot)
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    10f64 * lambda.log10()
}

#[pyfunction]
fn max_cut_detector(signal: Vec<Complex<f64>>) -> f64 {
    let np = 64;
    let n = 4096;
    let lambda: f64 = ssca_base(&signal, n, np)
        .iter()
        .map(|&s_i| s_i.norm_sqr())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    10f64 * lambda.log10()
}

#[pyfunction]
fn max_cut_detector_sxf(sxf: PyReadonlyArray2<'_, Complex<f64>>) -> f64 {
    let lambda = sxf
        .as_array()
        .iter()
        .map(|s_i| s_i.norm_sqr())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    10f64 * lambda.log10()
}

#[pyfunction]
#[pyo3(name = "ssca", signature=(s, n=4096, np=64, map_output=false))]
pub fn ssca_py(
    py: Python<'_>,
    s: Vec<Complex<f64>>,
    n: usize,
    np: usize,
    map_output: bool,
) -> Bound<'_, PyArray2<Complex<f64>>> {
    if map_output {
        ssca_mapped(&s, n, np).into_pyarray(py)
    } else {
        ssca_base(&s, n, np).into_pyarray(py)
    }
}

#[pyfunction]
pub fn random_data(py: Python<'_>, num_bits: usize) -> Bound<'_, PyArray1<Bit>> {
    let mut rng = rand::thread_rng();
    PyArray1::from_iter(py, (0..num_bits).map(|_| rng.gen::<Bit>()))
}

#[pyfunction]
fn energy_detector(signal: Vec<Complex<f64>>) -> f64 {
    10f64 * (signal.into_iter().map(|s_i| s_i.norm_sqr()).sum::<f64>()).log10()
}

#[pyfunction]
#[pyo3(name = "awgn")]
pub fn awgn_py(signal: Vec<Complex<f64>>, sigma: f64) -> Vec<Complex<f64>> {
    signal
        .into_iter()
        .zip(
            Normal::new(0f64, sigma)
                .unwrap()
                .sample_iter(rand::thread_rng()),
        )
        .zip(
            Normal::new(0f64, sigma)
                .unwrap()
                .sample_iter(rand::thread_rng()),
        )
        .map(|((sample, n_1), n_2)| sample + Complex::new(n_1, n_2))
        .collect()
}

#[pyfunction]
fn awgn2(signal: Vec<Complex<f64>>, sigma: f64) -> Vec<Complex<f64>> {
    signal
        .into_iter()
        .zip(
            Normal::new(0f64, sigma / 2f64)
                .unwrap()
                .sample_iter(rand::thread_rng()),
        )
        .zip(
            Normal::new(0f64, sigma / 2f64)
                .unwrap()
                .sample_iter(rand::thread_rng()),
        )
        .map(|((sample, n_1), n_2)| sample + Complex::new(n_1, n_2))
        .collect()
}

#[pyfunction]
fn pure_awgn(size: usize, sigma: f64) -> Vec<Complex<f64>> {
    Normal::new(0f64, sigma / 2f64)
        .unwrap()
        .sample_iter(rand::thread_rng())
        .zip(
            Normal::new(0f64, sigma / 2f64)
                .unwrap()
                .sample_iter(rand::thread_rng()),
        )
        .map(|(n_1, n_2)| Complex::new(n_1, n_2))
        .take(size)
        .collect()
}

#[pymodule]
#[pyo3(name = "bazbandilo")]
fn module_with_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    /// Transmit a baseband signal.
    #[pyfunction]
    fn tx_bpsk(message: Vec<Bit>) -> Vec<Complex<f64>> {
        tx_bpsk_signal(message.into_iter()).collect()
    }

    #[pyfunction]
    fn rx_bpsk(signal: Vec<Complex<f64>>) -> Vec<Bit> {
        rx_bpsk_signal(signal.into_iter()).collect()
    }

    #[pyfunction]
    fn tx_qpsk(message: Vec<Bit>) -> Vec<Complex<f64>> {
        if message.len() % 2 != 0 {
            tx_qpsk_signal(message.into_iter().chain([Bit::default()])).collect()
        } else {
            tx_qpsk_signal(message.into_iter()).collect()
        }
    }
    #[pyfunction]
    fn rx_qpsk(signal: Vec<Complex<f64>>) -> Vec<Bit> {
        rx_qpsk_signal(signal.into_iter()).collect()
    }

    #[pyfunction]
    fn tx_mpsk(message: Vec<Bit>, m: usize) -> Vec<Complex<f64>> {
        tx_mpsk_signal(message.into_iter(), m).collect()
    }

    #[pyfunction]
    fn rx_mpsk(signal: Vec<Complex<f64>>, m: usize) -> Vec<Bit> {
        rx_mpsk_signal(signal.into_iter(), m).collect()
    }

    #[pyfunction]
    fn tx_ofdm_qpsk(message: Vec<Bit>, subcarriers: usize, pilots: usize) -> Vec<Complex<f64>> {
        tx_ofdm_signal(tx_qpsk_signal(message.into_iter()), subcarriers, pilots).collect()
    }
    #[pyfunction]
    fn rx_ofdm_qpsk(signal: Vec<Complex<f64>>, subcarriers: usize, pilots: usize) -> Vec<Bit> {
        rx_qpsk_signal(rx_ofdm_signal(signal.into_iter(), subcarriers, pilots)).collect()
    }

    #[pyfunction]
    #[pyo3(signature=(message, key_len=16))]
    fn tx_cdma_bpsk(message: Vec<Bit>, key_len: usize) -> Vec<Complex<f64>> {
        let walsh_codes = HadamardMatrix::new(key_len);
        let key: Vec<Bit> = walsh_codes.key(0).clone();
        tx_cdma_bpsk_signal(message.into_iter(), &key).collect()
    }
    #[pyfunction]
    #[pyo3(signature=(signal, key_len=16))]
    fn rx_cdma_bpsk(signal: Vec<Complex<f64>>, key_len: usize) -> Vec<Bit> {
        let walsh_codes = HadamardMatrix::new(key_len);
        let key: Vec<Bit> = walsh_codes.key(0).clone();
        rx_cdma_bpsk_signal(signal.into_iter(), &key).collect()
    }

    #[pyfunction]
    #[pyo3(signature=(message, key_len=16))]
    fn tx_cdma_qpsk(message: Vec<Bit>, key_len: usize) -> Vec<Complex<f64>> {
        let walsh_codes = HadamardMatrix::new(key_len);
        let key: Vec<Bit> = walsh_codes.key(0).clone();
        tx_cdma_qpsk_signal(message.into_iter(), &key).collect()
    }
    #[pyfunction]
    #[pyo3(signature=(signal, key_len=16))]
    fn rx_cdma_qpsk(signal: Vec<Complex<f64>>, key_len: usize) -> Vec<Bit> {
        let walsh_codes = HadamardMatrix::new(key_len);
        let key: Vec<Bit> = walsh_codes.key(0).clone();
        rx_cdma_qpsk_signal(signal.into_iter(), &key).collect()
    }

    #[pyfunction]
    fn tx_bfsk(data: Vec<Bit>, delta_f: usize) -> Vec<Complex<f64>> {
        tx_bfsk_signal(data.into_iter(), delta_f).collect()
    }
    #[pyfunction]
    fn rx_bfsk(signal: Vec<Complex<f64>>, delta_f: usize) -> Vec<Bit> {
        rx_bfsk_signal(signal.into_iter(), delta_f).collect()
    }

    #[pyfunction]
    fn tx_qam(data: Vec<Bit>, m: usize) -> Vec<Complex<f64>> {
        tx_qam_signal(data.into_iter(), m).collect()
    }
    #[pyfunction]
    fn rx_qam(signal: Vec<Complex<f64>>, m: usize) -> Vec<Bit> {
        rx_qam_signal(signal.into_iter(), m).collect()
    }

    #[pyfunction]
    fn tx_csk(data: Vec<Bit>) -> Vec<Complex<f64>> {
        tx_csk_signal(data.into_iter()).collect()
    }
    #[pyfunction]
    fn rx_csk(signal: Vec<Complex<f64>>) -> Vec<Bit> {
        rx_csk_signal(signal.into_iter()).collect()
    }

    #[pyfunction]
    fn tx_dcsk(data: Vec<Bit>) -> Vec<Complex<f64>> {
        tx_dcsk_signal(data.into_iter()).collect()
    }
    #[pyfunction]
    fn rx_dcsk(signal: Vec<Complex<f64>>) -> Vec<Bit> {
        rx_dcsk_signal(signal.into_iter()).collect()
    }
    #[pyfunction]
    fn tx_qcsk(data: Vec<Bit>) -> Vec<Complex<f64>> {
        tx_qcsk_signal(data.into_iter()).collect()
    }
    #[pyfunction]
    fn rx_qcsk(signal: Vec<Complex<f64>>) -> Vec<Bit> {
        rx_qcsk_signal(signal.into_iter()).collect()
    }

    #[pyfunction]
    fn tx_fh_ofdm_dcsk(data: Vec<Bit>) -> Vec<Complex<f64>> {
        tx_fh_ofdm_dcsk_signal(data.into_iter()).collect()
    }
    #[pyfunction]
    fn rx_fh_ofdm_dcsk(signal: Vec<Complex<f64>>) -> Vec<Bit> {
        rx_fh_ofdm_dcsk_signal(signal.into_iter()).collect()
    }

    m.add_function(wrap_pyfunction!(tx_bpsk, m)?)?;
    m.add_function(wrap_pyfunction!(rx_bpsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_qpsk, m)?)?;
    m.add_function(wrap_pyfunction!(rx_qpsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_mpsk, m)?)?;
    m.add_function(wrap_pyfunction!(rx_mpsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_cdma_bpsk, m)?)?;
    m.add_function(wrap_pyfunction!(rx_cdma_bpsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_cdma_qpsk, m)?)?;
    m.add_function(wrap_pyfunction!(rx_cdma_qpsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_ofdm_qpsk, m)?)?;
    m.add_function(wrap_pyfunction!(rx_ofdm_qpsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_bfsk, m)?)?;
    m.add_function(wrap_pyfunction!(rx_bfsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_qam, m)?)?;
    m.add_function(wrap_pyfunction!(rx_qam, m)?)?;
    m.add_function(wrap_pyfunction!(tx_csk, m)?)?;
    m.add_function(wrap_pyfunction!(rx_csk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_dcsk, m)?)?;
    m.add_function(wrap_pyfunction!(rx_dcsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_qcsk, m)?)?;
    m.add_function(wrap_pyfunction!(rx_qcsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_fh_ofdm_dcsk, m)?)?;
    m.add_function(wrap_pyfunction!(rx_fh_ofdm_dcsk, m)?)?;
    m.add_function(wrap_pyfunction!(random_data, m)?)?;
    m.add_function(wrap_pyfunction!(awgn_py, m)?)?;
    m.add_function(wrap_pyfunction!(awgn2, m)?)?;
    m.add_function(wrap_pyfunction!(pure_awgn, m)?)?;
    m.add_function(wrap_pyfunction!(energy_detector, m)?)?;
    m.add_function(wrap_pyfunction!(energy_real, m)?)?;
    m.add_function(wrap_pyfunction!(energy_complex, m)?)?;
    m.add_function(wrap_pyfunction!(ssca_py, m)?)?;
    m.add_function(wrap_pyfunction!(max_cut_detector, m)?)?;
    m.add_function(wrap_pyfunction!(max_cut_detector_sxf, m)?)?;
    m.add_function(wrap_pyfunction!(dcs_detector, m)?)?;
    m.add_function(wrap_pyfunction!(dcs_detector_sxf, m)?)?;
    Ok(())
}
