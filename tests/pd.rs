use std::ffi::CString;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;

use kdam::{par_tqdm, BarExt};
use ndarray::s;
use num::Zero;
use num_complex::Complex;
use numpy::ndarray::{Array2, Axis};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[macro_use]
mod util;

use bazbandilo::{
    awgn,
    cdma::{tx_cdma_bpsk_signal, tx_cdma_qpsk_signal},
    csk::tx_csk_signal,
    css::tx_css_signal,
    dcsk::{tx_dcsk_signal, tx_qcsk_signal},
    fam::fam,
    fh_ofdm_dcsk::tx_fh_ofdm_dcsk_signal,
    fsk::tx_bfsk_signal,
    hadamard::HadamardMatrix,
    linspace,
    ofdm::tx_ofdm_signal,
    psk::{tx_bpsk_signal, tx_qpsk_signal},
    qam::tx_qam_signal,
    ssca::{ssca_base, ssca_mapper},
    undb, Bit,
};

fn dcs_detect(sxf: &Array2<Complex<f64>>) -> f64 {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Detector {
    Energy,
    MaxCut,
    Dcs,
    DcsFam,
    MaxCutFam,
}

impl Detector {
    fn get(&self) -> &str {
        match self {
            Detector::Energy => "Energy Detector",
            Detector::MaxCut => "Max Cut Detector",
            Detector::Dcs => "DCS Detector",
            Detector::DcsFam => "DcsFam Detector",
            Detector::MaxCutFam => "MaxCutFam Detector",
        }
    }
    fn iter() -> impl Iterator<Item = Detector> {
        [
            Detector::Energy,
            Detector::MaxCut,
            Detector::Dcs,
            Detector::DcsFam,
            Detector::MaxCutFam,
        ]
        .into_iter()
    }
}

fn run_detectors<I: Iterator<Item = Complex<f64>>>(signal: I) -> Vec<DetectorOutput> {
    let np = 64;
    // let np = 128;
    // let np = 256;
    let n = 4096;
    // let n = 8192;
    // let n = 32768;
    let chan_signal: Vec<Complex<f64>> = signal.take(n + np).collect();
    // let chan_signal: Vec<Complex<f64>> = signal.collect();
    let sxf = ssca_base(&chan_signal, n, np);
    let sxf_mapped = ssca_mapper(&sxf);

    let sxf_fam = fam(&chan_signal, n + np, np);

    vec![
        DetectorOutput {
            kind: Detector::Energy,
            λ: energy_detect(&chan_signal),
        },
        DetectorOutput {
            kind: Detector::MaxCut,
            λ: max_cut_detect(&sxf),
        },
        DetectorOutput {
            kind: Detector::Dcs,
            λ: dcs_detect(&sxf_mapped),
        },
        // DetectorOutput {
        //     kind: Detector::NormalTest,
        //     λ: normal_detect(&chan_signal),
        // },
        DetectorOutput {
            kind: Detector::DcsFam,
            λ: dcs_detect_fam(&sxf_fam),
        },
        DetectorOutput {
            kind: Detector::MaxCutFam,
            λ: max_cut_detect(&sxf_fam),
        },
    ]
}

/// The output of a detector on a lone signal.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct DetectorOutput {
    kind: Detector,
    λ: f64,
}

/// The output of a detector on many snrs.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct DetectorResults {
    kind: Detector,
    snrs: Vec<f64>,
    h0_λs: Vec<Vec<f64>>,
    h1_λs: Vec<Vec<f64>>,
}

/// The result of running a modulation at many SNRS many times.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ModulationDetectorResults {
    name: String,
    results: Vec<DetectorResults>,
    snrs: Vec<f64>,
}

/// Detector Test harness.
struct DetectorTest<'a> {
    snrs: Vec<f64>,
    run_fn: &'a dyn Fn(&[f64]) -> ModulationDetectorResults,
}

impl DetectorTest<'_> {
    fn run(self) -> ModulationDetectorResults {
        (self.run_fn)(&self.snrs)
    }
}

fn remap_results(λs: &[Vec<Vec<DetectorOutput>>], kind: &str) -> Vec<Vec<f64>> {
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

const NUM_SAMPLES: usize = 65536;

// const NUM_ATTEMPTS: usize = 1000;
// const NUM_ATTEMPTS: usize = 20;
const NUM_ATTEMPTS: usize = 150;

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

#[test]
fn main() {
    // let snrs_db: Vec<f64> = linspace(-45f64, 12f64, 15).collect();
    // let snrs_db: Vec<f64> = linspace(-45f64, 12f64, 25).collect();
    let snrs_db: Vec<f64> = linspace(-45f64, 12f64, 150).collect();

    let snrs: Vec<f64> = snrs_db.iter().cloned().map(undb).collect();

    let h_16 = HadamardMatrix::new(16);
    let h_32 = HadamardMatrix::new(32);
    let h_64 = HadamardMatrix::new(64);
    let key_16 = h_16.key(2);
    let key_32 = h_32.key(2);
    let key_64 = h_64.key(2);

    let harness = [
        // PSK
        DetectorTest!("BPSK", tx_bpsk_signal, snrs),
        DetectorTest!("QPSK", tx_qpsk_signal, snrs),
        // CDMA
        DetectorTest!("CDMA-BPSK-16", |m| tx_cdma_bpsk_signal(m, key_16), snrs),
        DetectorTest!("CDMA-QPSK-16", |m| tx_cdma_qpsk_signal(m, key_16), snrs),
        // DetectorTest!("CDMA-BPSK-32", |m| tx_cdma_bpsk_signal(m, key_32), snrs),
        DetectorTest!("CDMA-QPSK-32", |m| tx_cdma_qpsk_signal(m, key_32), snrs),
        // DetectorTest!("CDMA-BPSK-64", |m| tx_cdma_bpsk_signal(m, key_64), snrs),
        DetectorTest!("CDMA-QPSK-64", |m| tx_cdma_qpsk_signal(m, key_64), snrs),
        // QAM
        // DetectorTest!("4QAM", |m| tx_qam_signal(m, 4), snrs),
        DetectorTest!("16QAM", |m| tx_qam_signal(m, 16), snrs),
        DetectorTest!("64QAM", |m| tx_qam_signal(m, 64), snrs),
        // BFSK
        DetectorTest!("BFSK-16", |m| tx_bfsk_signal(m, 16), snrs),
        DetectorTest!("BFSK-32", |m| tx_bfsk_signal(m, 32), snrs),
        DetectorTest!("BFSK-64", |m| tx_bfsk_signal(m, 64), snrs),
        // OFDM
        DetectorTest!(
            "OFDM-BPSK-16",
            |m| tx_ofdm_signal(tx_bpsk_signal(m), 16, 0),
            snrs
        ),
        DetectorTest!(
            "OFDM-QPSK-16",
            |m| tx_ofdm_signal(tx_qpsk_signal(m), 16, 0),
            snrs
        ),
        DetectorTest!(
            "OFDM-BPSK-64",
            |m| tx_ofdm_signal(tx_bpsk_signal(m), 64, 0),
            snrs
        ),
        DetectorTest!(
            "OFDM-QPSK-64",
            |m| tx_ofdm_signal(tx_qpsk_signal(m), 64, 0),
            snrs
        ),
        // Chirp Spread Spectrum
        DetectorTest!("CSS-16", |m| tx_css_signal(m, 16), snrs),
        DetectorTest!("CSS-64", |m| tx_css_signal(m, 64), snrs),
        // DetectorTest!("CSS-128", |m| tx_css_signal(m, 128), snrs),
        // CSK
        DetectorTest!("CSK", tx_csk_signal, snrs),
        DetectorTest!("DCSK", tx_dcsk_signal, snrs),
        DetectorTest!("QCSK", tx_qcsk_signal, snrs),
        // FH-OFDM-DCSK
        DetectorTest!("FH-OFDM-DCSK", tx_fh_ofdm_dcsk_signal, snrs),
    ];

    let results: Vec<ModulationDetectorResults> = {
        let mut results = Vec::with_capacity(harness.len());
        for modulation in harness {
            let result = modulation.run();
            results.push(result);

            // Save results to file.
            {
                let name = "/tmp/results.json";
                let file = File::create(name).unwrap();
                let mut writer = BufWriter::new(file);
                serde_json::to_writer(&mut writer, &results).unwrap();

                writer.flush().unwrap();
                println!("Saved {}", name);
            }
        }
        results
    };

    drop(results);
}
