use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;

use kdam::{par_tqdm, BarExt};
use num::Zero;
use num_complex::Complex;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[macro_use]
mod util;

use bazbandilo::{awgn, linspace, psk::tx_bpsk_signal, undb, Bit};

/// The output of a detector on many snrs.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct DetectorResults {
    h0_λs: Vec<Vec<f64>>,
    h1_λs: Vec<Vec<f64>>,
}

/// The result of running a modulation at many SNRS many times.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TestResults {
    num_samples: usize,
    results: DetectorResults,
    snrs: Vec<f64>,
}

/// Detector Test harness.
struct DetectorTest<'a> {
    snrs: Vec<f64>,
    run_fn: &'a dyn Fn(&[f64]) -> TestResults,
}

impl DetectorTest<'_> {
    fn run(self) -> TestResults {
        (self.run_fn)(&self.snrs)
    }
}

fn energy_detect<I: Iterator<Item = Complex<f64>>>(signal: I) -> f64 {
    // Σ |s_i|^2
    // 1/len(s) * Σ |s_i|^2
    // 10 log10( Σ |s_i|^2 )
    10f64 * (signal.map(|s_i| s_i.norm_sqr()).sum::<f64>()).log10()
}

// const NUM_ATTEMPTS: usize = 10_000;
// const NUM_ATTEMPTS: usize = 1000;
const NUM_ATTEMPTS: usize = 75;

macro_rules! DetectorTest {
    ($tx_fn:expr, $snrs:expr, $pow2:expr) => {{
        DetectorTest {
            snrs: $snrs.clone(),
            run_fn: &|snrs: &[f64]| {
                let name: String = format!("2^{}", $pow2);
                let num_samples: usize = 2_usize.pow($pow2 as u32);

                // Make a Progress Bar.
                let pb = {
                    let mut pb = par_tqdm!(total = snrs.len());
                    pb.refresh().unwrap();
                    pb.set_description(name.clone());
                    Mutex::new(pb)
                };

                // Calculate the signal energy.
                let (energy_signal, ns, num_bits) = energy_samples_bits!($tx_fn, num_samples);

                // Calculate the detector outputs for the modulation.
                let (h0_λs, h1_λs): (Vec<Vec<f64>>, Vec<Vec<f64>>) = snrs
                    .par_iter()
                    // Calculate the noise variance.
                    .map(|&snr| (energy_signal / (2f64 * ns as f64 * snr)).sqrt())
                    .map(|n0| {
                        // Generate signals.
                        let h0_λs: Vec<f64> = (0..NUM_ATTEMPTS)
                            .into_par_iter()
                            .map(|_| {
                                let noisy_signal = awgn((0..ns).map(|_| Complex::zero()), n0);
                                energy_detect(noisy_signal)
                            })
                            .collect();

                        let h1_λs: Vec<f64> = (0..NUM_ATTEMPTS)
                            .into_par_iter()
                            .map(|_| {
                                let mut rng = rand::thread_rng();
                                let data = (0..num_bits).map(|_| rng.gen::<Bit>());

                                energy_detect(awgn($tx_fn(data), n0))
                            })
                            .collect();

                        let mut pb = pb.lock().unwrap();
                        pb.update(1).unwrap();

                        (h0_λs, h1_λs)
                    })
                    .unzip();

                println!("Finished with {}.", name);
                let snrs: Vec<f64> = $snrs.to_vec();
                TestResults {
                    num_samples: ns,
                    results: DetectorResults { h0_λs, h1_λs },
                    snrs: snrs.clone(),
                }
            },
        }
    }};
}

#[test]
fn main() {
    let snrs_db: Vec<f64> = linspace(-45f64, 12f64, 150).collect();

    let snrs: Vec<f64> = snrs_db.iter().cloned().map(undb).collect();

    let harness = [
        DetectorTest!(tx_bpsk_signal, snrs, 1),
        DetectorTest!(tx_bpsk_signal, snrs, 2),
        DetectorTest!(tx_bpsk_signal, snrs, 3),
        DetectorTest!(tx_bpsk_signal, snrs, 4),
        DetectorTest!(tx_bpsk_signal, snrs, 5),
        DetectorTest!(tx_bpsk_signal, snrs, 6),
        DetectorTest!(tx_bpsk_signal, snrs, 7),
        DetectorTest!(tx_bpsk_signal, snrs, 8),
        DetectorTest!(tx_bpsk_signal, snrs, 9),
        DetectorTest!(tx_bpsk_signal, snrs, 10),
        DetectorTest!(tx_bpsk_signal, snrs, 11),
        DetectorTest!(tx_bpsk_signal, snrs, 12),
        DetectorTest!(tx_bpsk_signal, snrs, 13),
        DetectorTest!(tx_bpsk_signal, snrs, 14),
        DetectorTest!(tx_bpsk_signal, snrs, 15),
        DetectorTest!(tx_bpsk_signal, snrs, 16),
        DetectorTest!(tx_bpsk_signal, snrs, 17),
        DetectorTest!(tx_bpsk_signal, snrs, 18),
        DetectorTest!(tx_bpsk_signal, snrs, 19),
        DetectorTest!(tx_bpsk_signal, snrs, 20),
        DetectorTest!(tx_bpsk_signal, snrs, 21),
        DetectorTest!(tx_bpsk_signal, snrs, 22),
        DetectorTest!(tx_bpsk_signal, snrs, 23),
        DetectorTest!(tx_bpsk_signal, snrs, 24),
        DetectorTest!(tx_bpsk_signal, snrs, 25),
        DetectorTest!(tx_bpsk_signal, snrs, 26),
        DetectorTest!(tx_bpsk_signal, snrs, 27),
        DetectorTest!(tx_bpsk_signal, snrs, 28),
        DetectorTest!(tx_bpsk_signal, snrs, 29),
        DetectorTest!(tx_bpsk_signal, snrs, 30),
        DetectorTest!(tx_bpsk_signal, snrs, 31),
        DetectorTest!(tx_bpsk_signal, snrs, 32),
    ];

    let mut results: Vec<TestResults> = Vec::with_capacity(harness.len());
    for modulation in harness {
        let result = modulation.run();
        results.push(result);

        // Save results to file.
        {
            let name = "/tmp/tw.json";
            let file = File::create(name).unwrap();
            let mut writer = BufWriter::new(file);
            serde_json::to_writer(&mut writer, &results).unwrap();

            writer.flush().unwrap();
            println!("Saved {}", name);
        }
    }
}
