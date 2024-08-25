use crate::{linspace, Bit};
use std::f64::consts::PI;

use num_complex::Complex;

pub fn tx_bfsk_signal<I: Iterator<Item = Bit>>(
    message: I,
    delta_f: usize,
) -> impl Iterator<Item = Complex<f64>> {
    let degs_p: Vec<f64> = linspace(0f64, 2f64 * PI, delta_f).take(delta_f).collect();
    let degs_n: Vec<f64> = [2f64 * PI]
        .into_iter()
        .chain(degs_p.iter().cloned().rev())
        .take(delta_f)
        .collect();

    fn mm(d: f64) -> Complex<f64> {
        Complex::new(0f64, -d).exp()
    }

    message.flat_map(move |bit| {
        if bit {
            degs_p.clone().into_iter().map(mm)
        } else {
            degs_n.clone().into_iter().map(mm)
        }
    })
}
