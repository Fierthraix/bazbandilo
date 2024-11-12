use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;

use convert_case::{Case, Casing};
use kdam::{par_tqdm, BarExt};
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
    db,
    dcsk::{tx_dcsk_signal, tx_qcsk_signal},
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
    let top = sxf.map(Complex::<f64>::norm_sqr).sum_axis(Axis(1));
    let middle: usize = sxf.shape()[1] / 2;
    let bot = sxf.column(middle).map(Complex::<f64>::norm_sqr);

    let lambda = (top / bot)
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
        let normtest: Py<PyAny> = PyModule::from_code_bound(
            py,
            "import scipy
import numpy as np
def p_vals(signal_im, signal_re):
    t1 = scipy.stats.normaltest(signal_re).statistic
    t2 = scipy.stats.normaltest(signal_im).statistic
    return np.mean([t1, t2])",
            "",
            "",
        )
        .unwrap()
        .getattr("p_vals")
        .unwrap()
        .into();

        let signal_re: Vec<f64> = signal.iter().map(|&s_i| s_i.re).collect();
        let signal_im: Vec<f64> = signal.iter().map(|&s_i| s_i.im).collect();

        let locals = [("normtest", normtest)].into_py_dict_bound(py);
        locals.set_item("signal_re", signal_re).unwrap();
        locals.set_item("signal_im", signal_im).unwrap();

        let λ: f64 = py
            .eval_bound("normtest(signal_re, signal_im)", None, Some(&locals))
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
    NormalTest,
}

impl Detector {
    fn get(&self) -> &str {
        match self {
            Detector::Energy => "Energy Detector",
            Detector::MaxCut => "Max Cut Detector",
            Detector::Dcs => "DCS Detector",
            Detector::NormalTest => "Normal Test Detector",
        }
    }
    fn iter() -> impl Iterator<Item = Detector> {
        [
            Detector::Energy,
            Detector::MaxCut,
            Detector::Dcs,
            Detector::NormalTest,
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
            // λ: dcs_detect(&sxf),
        },
        DetectorOutput {
            kind: Detector::NormalTest,
            λ: normal_detect(&chan_signal),
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

impl<'a> DetectorTest<'a> {
    fn run(self) -> ModulationDetectorResults {
        (self.run_fn)(&self.snrs)
    }
}

/// Youden-J calculation of detector output.
#[derive(Clone)]
struct LogRegressResults {
    name: String,
    detector: DetectorResults,
    youden_js: Vec<f64>,
}

struct ModulationLogRegressResults(Vec<LogRegressResults>);

fn log_regress(results: &DetectorResults, name: &str) -> LogRegressResults {
    let youden_js: Vec<f64> = Python::with_gil(|py| {
        let pandas = py.import_bound("pandas").unwrap();
        let numpy = py.import_bound("numpy").unwrap();
        let sklearn = py.import_bound("sklearn").unwrap();
        let linear_model = py.import_bound("sklearn.linear_model").unwrap();
        let model_selection = py.import_bound("sklearn.model_selection").unwrap();
        let locals = [
            ("linear_model", linear_model),
            ("model_selection", model_selection),
            ("np", numpy),
            ("sklearn", sklearn),
            ("pd", pandas),
        ]
        .into_py_dict_bound(py);

        let youden_js: Vec<f64> = results
            .h0_λs
            .iter()
            .zip(results.h1_λs.iter())
            .map(|(h0, h1)| {
                locals.set_item("h0_λs", h0).unwrap();
                locals.set_item("h1_λs", h1).unwrap();

                // py.run_bound(statement, None, Some(&locals));
                py.run_bound(
                    r#"
x_var = np.concatenate((h0_λs, h1_λs)).reshape(-1, 1)
y_var = np.concatenate((np.zeros(len(h0_λs)), np.ones(len(h1_λs))))
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x_var, y_var, test_size=0.5, random_state=0
)

log_regression = linear_model.LogisticRegression()
log_regression.fit(x_train, y_train)

y_pred_proba = log_regression.predict_proba(x_test)[::, 1]
fpr, tpr, thresholds = sklearn.metrics.roc_curve(
    y_test, y_pred_proba, drop_intermediate=False
)

df_test = pd.DataFrame(
    {
        "x": x_test.flatten(),
        "y": y_test,
        "proba": y_pred_proba,
    }
)

# sort it by predicted probabilities
# because thresholds[1:] = y_proba[::-1]
df_test.sort_values(by="proba", inplace=True)
if len(tpr) == len(h0_λs):
    df_test["tpr"] = tpr[::-1]
else:
    df_test["tpr"] = tpr[1:][::-1]
if len(fpr) == len(h0_λs):
    df_test["fpr"] = fpr[::-1]
else:
    df_test["fpr"] = fpr[1:][::-1]
df_test["youden_j"] = df_test.tpr - df_test.fpr
        "#,
                    None,
                    Some(&locals),
                )
                .unwrap();

                let youden_j: f64 = py
                    .eval_bound(r#"max(abs(df_test["youden_j"]))"#, None, Some(&locals))
                    .unwrap()
                    .extract()
                    .unwrap();
                youden_j
            })
            .collect();
        youden_js
    });
    LogRegressResults {
        name: name.into(),
        detector: results.clone(),
        youden_js,
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

const NUM_ATTEMPTS: usize = 1000;
// const NUM_ATTEMPTS: usize = 500;
// const NUM_ATTEMPTS: usize = 250;
// const NUM_ATTEMPTS: usize = 75;
// const NUM_ATTEMPTS: usize = 20;

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

fn plot_thing(regressions: Vec<LogRegressResults>, snrs: &[f64]) {
    Python::with_gil(|py| {
        let cycler = py.import_bound("cycler").unwrap();
        let matplotlib = py.import_bound("matplotlib").unwrap();
        let plt = py.import_bound("matplotlib.pyplot").unwrap();
        let locals =
            [("cycler", cycler), ("matplotlib", matplotlib), ("plt", plt)].into_py_dict_bound(py);

        let snrs_db: Vec<f64> = snrs.iter().cloned().map(db).collect();
        locals.set_item("snrs", snrs).unwrap();
        locals.set_item("snrs_db", &snrs_db).unwrap();

        let (fig, axes): (&PyAny, &PyAny) = py
            .eval_bound("plt.subplots(1)", None, Some(&locals))
            .unwrap()
            .extract()
            .unwrap();
        // let (fig, axes): (&PyObject, &PyObject) = py
        // let x: &PyAny = py
        //     .eval_bound("plt.subplots(1)", None, Some(&locals))
        //     .unwrap()
        //     .extract()
        //     .unwrap();
        locals.set_item("fig", fig).unwrap();
        locals.set_item("axes", axes).unwrap();

        py.run_bound("cycles = (cycler.cycler(color=['r', 'g', 'b', 'c', 'm', 'y']) * cycler.cycler(linestyle=['-', ':', '-.']))", None, Some(&locals)).unwrap();
        py.eval_bound("axes.set_prop_cycle(cycles)", None, Some(&locals))
            .unwrap();

        // Plot the BER.
        for modulation in regressions.iter() {
            let py_name = format!("youden_js_{}", modulation.name).to_case(Case::Snake);
            locals.set_item(&py_name, &modulation.youden_js).unwrap();
            py.eval_bound(
                &format!(
                    "axes.plot(snrs_db[:len({})], {}, label='{}')",
                    py_name, py_name, modulation.name,
                ),
                None,
                Some(&locals),
            )
            .unwrap();
        }
        for line in [
            &format!("axes.set_title('{}')", regressions[0].name),
            "axes.legend(loc='best')",
            "axes.set_xlabel('SNR (dB)')",
            r"axes.set_ylabel('$\mathbb{P}_d$')",
            "plt.show()",
        ] {
            println!("{}", line);
            py.eval_bound(line, None, Some(&locals)).unwrap();
        }
    });
}

#[test]
fn main() {
    // let snrs_db: Vec<f64> = linspace(-10f64, 12f64, 50).collect();
    // let snrs_db: Vec<f64> = linspace(-25f64, 6f64, 25).collect();
    // let snrs_db: Vec<f64> = linspace(-25f64, 6f64, 15).collect();
    // let snrs_db: Vec<f64> = linspace(-45f64, 12f64, 50).collect();
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
        DetectorTest!("1024QAM", |m| tx_qam_signal(m, 1024), snrs),
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
        DetectorTest!("CSS-128", |m| tx_css_signal(m, 128), snrs),
        // CSK
        DetectorTest!("CSK", tx_csk_signal, snrs),
        DetectorTest!("DCSK", tx_dcsk_signal, snrs),
        DetectorTest!("QCSK", tx_qcsk_signal, snrs),
        // FH-OFDM-DCSK
        DetectorTest!("FH-OFDM-DCSK", tx_fh_ofdm_dcsk_signal, snrs),
    ];

    let results: Vec<ModulationDetectorResults> = harness
        .into_iter()
        .map(|modulation| modulation.run())
        .collect();

    // Save results to file.
    {
        let name = "/tmp/results.json";
        let file = File::create(name).unwrap();
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &results).unwrap();

        writer.flush().unwrap();
        println!("Saved {}", name);
    }

    /*
    let regressions: Vec<ModulationLogRegressResults> = results
        .iter()
        .map(|test_results| {
            ModulationLogRegressResults(
                test_results
                    .results
                    .iter()
                    .map(|dx_result| log_regress(dx_result, &test_results.name))
                    .collect(),
            )
        })
        .collect();

    for dx in Detector::iter() {
        let log_regresses: Vec<LogRegressResults> = {
            regressions
                .iter()
                .map(|mod_log_result| {
                    mod_log_result
                        .0
                        .iter()
                        .filter_map(|log_result| {
                            if log_result.detector.kind.get() == dx.get() {
                                Some(log_result.clone())
                            } else {
                                None
                            }
                        })
                        .next()
                        .unwrap()
                })
                .collect()
        };
        // plot_thing(log_regresses, &snrs);
    }
        */
}
