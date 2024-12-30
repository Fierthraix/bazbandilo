use std::ffi::CString;

use lazy_static::lazy_static;
use ndarray::s;
use num_complex::Complex;
use numpy::ndarray::{Array2, Axis};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use serde::{Deserialize, Serialize};

use bazbandilo::{fam::fam, linspace, undb};

pub const NUM_SAMPLES: usize = 65536;

// pub const NUM_ATTEMPTS: usize = 100;
pub const NUM_ATTEMPTS: usize = 1000;

lazy_static! {
    // pub static ref snrs_db: Vec<f64> = linspace(-45f64, 12f64, 15).collect();
    pub static ref snrs_db: Vec<f64> = linspace(-45f64, 12f64, 150).collect();
    pub static ref snrs_lin: Vec<f64> = snrs_db.iter().cloned().map(undb).collect();
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Detector {
    Energy,
    MaxCut,
    Dcs,
    NormalTest,
}

impl Detector {
    pub fn get(&self) -> &str {
        match self {
            Detector::Energy => "Energy Detector",
            Detector::MaxCut => "Max Cut Detector",
            Detector::Dcs => "DCS Detector",
            Detector::NormalTest => "Normal Test Detector",
        }
    }
    pub fn iter() -> impl Iterator<Item = Detector> {
        [
            Detector::Energy,
            Detector::MaxCut,
            Detector::Dcs,
            Detector::NormalTest,
        ]
        .into_iter()
    }
}

pub fn run_detectors<I: Iterator<Item = Complex<f64>>>(signal: I) -> Vec<DetectorOutput> {
    let np = 64;
    let n = 4096;
    let chan_signal: Vec<Complex<f64>> = signal.take(n + np).collect();

    let sxf_fam = fam(&chan_signal, n + np, np);

    vec![
        DetectorOutput {
            kind: Detector::Energy,
            λ: energy_detect(&chan_signal),
        },
        DetectorOutput {
            kind: Detector::MaxCut,
            λ: max_cut_detect(&sxf_fam),
        },
        DetectorOutput {
            kind: Detector::Dcs,
            λ: dcs_detect_fam(&sxf_fam),
        },
        DetectorOutput {
            kind: Detector::NormalTest,
            λ: normal_detect(&chan_signal),
        },
    ]
}

pub fn dcs_detect(sxf: &Array2<Complex<f64>>) -> f64 {
    let middle: usize = sxf.shape()[1] / 2;
    let left = sxf.slice(s![.., ..middle]);
    let right = sxf.slice(s![.., middle + 1..]);

    let left = left.map(Complex::<f64>::norm_sqr).sum_axis(Axis(1));
    let right = right.map(Complex::<f64>::norm_sqr).sum_axis(Axis(1));

    let lambda = left
        .into_iter()
        .chain(right)
        .map(|x| if x.is_normal() { x } else { 0f64 })
        .max_by(|a, b| a.partial_cmp(b).unwrap_or_else(|| panic!("{}, {}", a, b)))
        .unwrap();

    10f64 * lambda.log10()
}

fn dcs_detect_fam(sxf: &Array2<Complex<f64>>) -> f64 {
    // let sxf = sxf.slice(s![1.., ..]);
    let top = sxf
        .slice(s![1.., ..])
        .map(Complex::<f64>::norm_sqr)
        .sum_axis(Axis(0));

    let lambda = top
        .into_iter()
        .map(|x| if x.is_normal() { x } else { 0f64 })
        .max_by(|a, b| a.partial_cmp(b).unwrap_or_else(|| panic!("{}, {}", a, b)))
        .unwrap();

    10f64 * lambda.log10()
}

fn max_cut_detect(sxf: &Array2<Complex<f64>>) -> f64 {
    let lambda: f64 = sxf
        .iter()
        .map(|&s_i| s_i.norm_sqr())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    10f64 * lambda.log10()
}

fn energy_detect(signal: &[Complex<f64>]) -> f64 {
    // Σ |s_i|^2
    // 1/len(s) * Σ |s_i|^2
    // 10 log10( Σ |s_i|^2 )
    10f64 * (signal.iter().map(|&s_i| s_i.norm_sqr()).sum::<f64>()).log10()
}

fn normal_detect(signal: &[Complex<f64>]) -> f64 {
    Python::with_gil(|py| {
        let normtest: Py<PyAny> = PyModule::from_code(
            py,
            c!("import scipy
import numpy as np
def p_vals(signal_im, signal_re):
    t1 = scipy.stats.normaltest(signal_re).statistic
    t2 = scipy.stats.normaltest(signal_im).statistic
    return np.mean([t1, t2])"),
            c!(""),
            c!(""),
        )
        .unwrap()
        .getattr("p_vals")
        .unwrap()
        .into();

        let signal_re: Vec<f64> = signal.iter().map(|&s_i| s_i.re).collect();
        let signal_im: Vec<f64> = signal.iter().map(|&s_i| s_i.im).collect();

        let locals = [("normtest", normtest)].into_py_dict(py).unwrap();
        locals.set_item("signal_re", signal_re).unwrap();
        locals.set_item("signal_im", signal_im).unwrap();

        let λ: f64 = py
            .eval(c!("normtest(signal_re, signal_im)"), None, Some(&locals))
            .unwrap()
            .extract()
            .unwrap();
        λ
    })
}

/// The output of a detector on a lone signal.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectorOutput {
    pub kind: Detector,
    pub λ: f64,
}

/// The output of a detector on many snrs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectorResults {
    pub kind: Detector,
    pub snrs: Vec<f64>,
    pub h0_λs: Vec<Vec<f64>>,
    pub h1_λs: Vec<Vec<f64>>,
}

/// The result of running a modulation at many SNRS many times.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModulationDetectorResults {
    pub name: String,
    pub results: Vec<DetectorResults>,
    pub snrs: Vec<f64>,
}

/// Detector Test harness.
pub struct DetectorTest<'a> {
    pub snrs: Vec<f64>,
    pub run_fn: &'a dyn Fn(&[f64]) -> ModulationDetectorResults,
}

impl DetectorTest<'_> {
    pub fn run(self) -> ModulationDetectorResults {
        (self.run_fn)(&self.snrs)
    }
}

pub fn remap_results(λs: &[Vec<Vec<DetectorOutput>>], kind: &str) -> Vec<Vec<f64>> {
    let λs: Vec<Vec<f64>> = λs
        .iter()
        .map(|input| {
            input
                .iter()
                .map(|attempt| {
                    attempt
                        .iter()
                        .filter_map(|dx_i| {
                            if dx_i.kind.get() == kind {
                                Some(dx_i.λ)
                            } else {
                                None
                            }
                        })
                        .next()
                        .unwrap()
                })
                .collect()
        })
        .collect();
    λs
}

macro_rules! DetectorTest {
    ($name:expr, $tx_fn:expr, $snrs:expr) => {{
        DetectorTest {
            snrs: $snrs.clone(),
            run_fn: &|snrs: &[f64]| {
                // Make a Progress Bar.
                let pb = {
                    let mut pb = par_tqdm!(total = $snrs.len());
                    pb.refresh().unwrap();
                    pb.set_description($name);
                    Mutex::new(pb)
                };

                // Calculate the signal energy.
                let (energy_signal, num_samples, num_bits) =
                    energy_samples_bits!($tx_fn, NUM_SAMPLES);

                // Calculate the detector outputs for the modulation.
                let (unmapped_h0_λs, unmapped_h1_λs): (
                    Vec<Vec<Vec<DetectorOutput>>>,
                    Vec<Vec<Vec<DetectorOutput>>>,
                ) = snrs
                    .par_iter()
                    // Calculate the noise variance.
                    .map(|&snr| (energy_signal / (2f64 * num_samples as f64 * snr)).sqrt())
                    .map(|n0| {
                        // Generate signals.
                        let h0_λs: Vec<Vec<DetectorOutput>> = (0..NUM_ATTEMPTS)
                            .into_par_iter()
                            .map(|_| {
                                let noisy_signal =
                                    awgn((0..num_samples).map(|_| Complex::zero()), n0);
                                run_detectors(noisy_signal)
                            })
                            .collect();

                        let h1_λs: Vec<Vec<DetectorOutput>> = (0..NUM_ATTEMPTS)
                            .into_par_iter()
                            .map(|_| {
                                let mut rng = rand::thread_rng();
                                let data = (0..num_bits).map(|_| rng.gen::<Bit>());

                                run_detectors(awgn($tx_fn(data), n0))
                            })
                            .collect();

                        let mut pb = pb.lock().unwrap();
                        pb.update(1).unwrap();

                        (h0_λs, h1_λs)
                    })
                    .unzip();

                println!("Finished with {}.", $name);
                ModulationDetectorResults {
                    name: String::from($name),
                    results: Detector::iter()
                        .map(|dx| {
                            let h0_λs = remap_results(&unmapped_h0_λs, dx.get());
                            let h1_λs = remap_results(&unmapped_h1_λs, dx.get());
                            DetectorResults {
                                kind: dx,
                                snrs: $snrs.to_vec(),
                                h0_λs,
                                h1_λs,
                            }
                        })
                        .collect(),
                    snrs: $snrs.to_vec(),
                }
            },
        }
    }};
}
