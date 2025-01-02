use crate::{fftshift, hamming_window_complex, iter::Iter};

use ndrustfft::Zero;
use num_complex::Complex;
use numpy::ndarray::{range, s, Array1, Array2};
use rustfft::FftPlanner;

#[inline]
pub fn fam(s: &[Complex<f64>], n: usize, np: usize) -> Array2<Complex<f64>> {
    assert!(n > np, "{} > {}", n, np);
    let mut fftp = FftPlanner::new();
    let fft_n = fftp.plan_fft_forward(n);

    let alphas = Array1::from_iter(range(0f64, 0.5, 0.001));

    let window = Array1::from_iter(hamming_window_complex(np));

    let mut x: Vec<Complex<f64>> = s.to_vec();
    fft_n.process(&mut x);
    let a = Array1::from_vec(fftshift(&x));

    let num_freqs = (n as f64 / np as f64).ceil() as usize;
    let mut sx: Array2<Complex<f64>> = Array2::zeros((alphas.len(), num_freqs));

    {
        let window: Vec<Complex<f64>> = Vec::from(window.as_slice().unwrap());
        let scf_slice = Array1::from_iter(
            a.iter()
                .map(|a_i| a_i * a_i.conj())
                .convolve(window)
                .take_every(np)
                .skip(1),
        );
        sx.slice_mut(s![0, ..]).assign(&scf_slice);
    }

    for (i, alpha) in alphas.into_iter().enumerate().skip(1) {
        let shift = (alpha * n as f64 / 2f64) as isize;
        let shift_right = {
            let mut b = Array1::uninit(a.dim());
            // println!("alen: {} | blen: {} | shift: {}", a.len(), b.len(), shift);
            a.slice(s![-shift..]).assign_to(b.slice_mut(s![..shift]));
            a.slice(s![..-shift]).assign_to(b.slice_mut(s![shift..]));
            unsafe { b.assume_init() }
        };

        let shift_left = {
            let mut b = Array1::uninit(a.dim());
            a.slice(s![shift..])
                .map(|a_i| a_i.conj())
                .assign_to(b.slice_mut(s![..-shift]));
            a.slice(s![..shift])
                .map(|a_i| a_i.conj())
                .assign_to(b.slice_mut(s![-shift..]));
            unsafe { b.assume_init() }
        };

        let scf_slice = shift_right * shift_left;

        let window: Vec<Complex<f64>> = Vec::from(window.as_slice().unwrap());
        let scf_slice = Array1::from_iter(
            scf_slice
                .into_iter()
                .convolve(window)
                .take_every(np)
                .skip(1),
        );
        sx.slice_mut(s![i, ..]).assign(&scf_slice);
    }
    sx
}

#[inline]
pub fn fam_sans_psd(s: &[Complex<f64>], n: usize, np: usize) -> Array2<Complex<f64>> {
    let mut sxf = fam(s, n, np);
    // Null out alpha=0.
    sxf.slice_mut(s![0, ..]).fill(Complex::zero());
    sxf
}
