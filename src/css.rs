use crate::{angle_diff, iter::Iter, linspace, Bit};
use num_complex::Complex;
use std::f64::consts::PI;

pub fn tx_css_signal<I: Iterator<Item = Bit>>(
    message: I,
    samples_per_symbol: usize,
) -> impl Iterator<Item = Complex<f64>> {
    assert!(samples_per_symbol >= 8);

    let delta_theta_max: f64 = 0.75 * PI;
    let delta_thetas: Vec<f64> =
        linspace(-delta_theta_max, delta_theta_max, samples_per_symbol - 1).collect();

    let mut theta: f64 = 0f64;
    message
        .flat_map(move |bit| {
            let mut out = Vec::with_capacity(samples_per_symbol);
            out.push(theta);

            for delta in delta_thetas.iter() {
                if bit {
                    theta += delta;
                } else {
                    theta -= delta;
                }
                out.push(theta);
            }
            out.into_iter()
        })
        .map(|angle| Complex::new(0f64, angle).exp())
}

pub fn rx_css_signal<I: Iterator<Item = Complex<f64>>>(
    message: I,
    samples_per_symbol: usize,
) -> impl Iterator<Item = Bit> {
    assert!(samples_per_symbol >= 8);

    message.chunks(samples_per_symbol).map(|symbol| {
        let angular_velocities: Vec<f64> = symbol
            .iter()
            .zip(symbol[1..].iter())
            .map(|(&w1, &w2)| angle_diff(w1, w2))
            .collect();

        let angular_accelerations = angular_velocities
            .iter()
            .zip(angular_velocities[1..].iter())
            .map(|(&v1, &v2)| v2 - v1);

        // Check average angular velocity.
        angular_accelerations.sum::<f64>().is_sign_positive()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    extern crate rand_distr;
    use crate::Rng;

    #[test]
    fn css() {
        let mut rng = rand::thread_rng();
        let num_bits = 9001;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let samples_per_symbol = 1000;

        let css_tx: Vec<Complex<f64>> =
            tx_css_signal(data_bits.iter().cloned(), samples_per_symbol).collect();
        let css_rx: Vec<Bit> = rx_css_signal(css_tx.iter().cloned(), samples_per_symbol).collect();
        assert_eq!(data_bits, css_rx);
    }
}
