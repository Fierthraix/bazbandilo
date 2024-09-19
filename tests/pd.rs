use std::sync::Mutex;

use convert_case::{Case, Casing};
use kdam::{par_tqdm, BarExt};
use num::Zero;
use num_complex::Complex;
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
    fh_ofdm_dcsk::tx_fh_ofdm_dcsk_signal,
    fsk::tx_bfsk_signal,
    hadamard::HadamardMatrix,
    linspace,
    ofdm::tx_ofdm_signal,
    psk::{tx_bpsk_signal, tx_qpsk_signal},
    qam::tx_qam_signal,
    ssca::ssca_base,
    undb, Bit,
};

trait Detector {
    fn detect<I: Iterator<Item = Complex<f64>>>(signal: I) -> f64;
}

struct MaxCutDetector;

impl Detector for MaxCutDetector {
    fn detect<I: Iterator<Item = Complex<f64>>>(signal: I) -> f64 {
        let np = 64;
        let n = 4096;
        let lambda: f64 = ssca_base(&signal.take(n + np).collect::<Vec<Complex<f64>>>(), n, np)
            .iter()
            .map(|&s_i| s_i.norm_sqr())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        10f64 * lambda.log10()
    }
}

struct EnergyDetector;

impl Detector for EnergyDetector {
    fn detect<I: Iterator<Item = Complex<f64>>>(signal: I) -> f64 {
        10f64 * (signal.map(|s_i| s_i.norm_sqr()).sum::<f64>()).log10()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DetectorResults {
    name: String,
    h0_λs: Vec<Vec<f64>>,
    h1_λs: Vec<Vec<f64>>,
    snrs: Vec<f64>,
}

struct DetectorTest<'a> {
    name: String,
    snrs: Vec<f64>,
    run_fn: &'a dyn Fn(&[f64]) -> DetectorResults,
}

impl<'a> DetectorTest<'a> {
    fn run(self) -> DetectorResults {
        (self.run_fn)(&self.snrs)
    }
}

struct LogRegressResults {
    detector: DetectorResults,
    youden_js: Vec<f64>,
}

fn log_regress(results: &DetectorResults) -> LogRegressResults {
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
        detector: results.clone(),
        youden_js,
    }
}

const NUM_BITS: usize = 65536;
// const NUM_ATTEMPTS: usize = 200;
const NUM_ATTEMPTS: usize = 75;

macro_rules! DetectorTest {
    ($name:expr, $tx_fn:expr, $snrs:expr) => {{
        DetectorTest {
            name: String::from($name),
            snrs: $snrs.clone(),
            run_fn: &|snrs: &[f64]| {
                let (eb, num_samps) = {
                    let mut rng = rand::thread_rng();
                    let data = (0..NUM_BITS).map(|_| rng.gen::<Bit>());
                    let tx_signal: Vec<Complex<f64>> = $tx_fn(data).collect();

                    let energy: f64 = tx_signal.iter().map(|&s_i| s_i.norm_sqr()).sum();

                    (energy / NUM_BITS as f64, tx_signal.len())
                };

                let pb = {
                    let mut pb = par_tqdm!(total = $snrs.len());
                    pb.refresh().unwrap();
                    pb.set_description($name);
                    Mutex::new(pb)
                };

                let n0s = snrs.par_iter().map(|snr| (eb / (2f64 * snr)).sqrt());
                let (h0_λs, h1_λs) = n0s
                    .map(|n0| {
                        // Generate signals.
                        let h0_λs: Vec<f64> = (0..NUM_ATTEMPTS)
                            .into_par_iter()
                            .map(|_| {
                                let noisy_signal =
                                    awgn((0..num_samps).map(|_| Complex::zero()), n0);
                                MaxCutDetector::detect(noisy_signal)
                            })
                            .collect();

                        let h1_λs: Vec<f64> = (0..NUM_ATTEMPTS)
                            .into_par_iter()
                            .map(|_| {
                                let mut rng = rand::thread_rng();
                                let data = (0..NUM_BITS).map(|_| rng.gen::<Bit>());
                                let signal = $tx_fn(data);
                                let noisy_signal = awgn(signal, n0);

                                MaxCutDetector::detect(noisy_signal)
                            })
                            .collect();

                        let mut pb = pb.lock().unwrap();
                        pb.update(1).unwrap();

                        (h0_λs, h1_λs)
                    })
                    .unzip();

                println!("Finished with {}.", $name);
                DetectorResults {
                    name: String::from($name),
                    h0_λs,
                    h1_λs,
                    snrs: $snrs.to_vec(),
                }
            },
        }
    }};
}

#[test]
fn main() {
    // let snrs_db: Vec<f64> = linspace(-10f64, 12f64, 50).collect();
    // let snrs_db: Vec<f64> = linspace(-25f64, 6f64, 25).collect();
    let snrs_db: Vec<f64> = linspace(-45f64, 12f64, 50).collect();

    let snrs: Vec<f64> = snrs_db.iter().cloned().map(undb).collect();

    let h = HadamardMatrix::new(32);
    let key = h.key(2);

    let harness = [
        // PSK
        DetectorTest!("BPSK", tx_bpsk_signal, snrs),
        DetectorTest!("QPSK", tx_qpsk_signal, snrs),
        // CDMA
        DetectorTest!("CDMA-BPSK", |m| tx_cdma_bpsk_signal(m, key), snrs),
        DetectorTest!("CDMA-QPSK", |m| tx_cdma_qpsk_signal(m, key), snrs),
        // QAM
        DetectorTest!("4QAM", |m| tx_qam_signal(m, 4), snrs),
        DetectorTest!("16QAM", |m| tx_qam_signal(m, 16), snrs),
        DetectorTest!("64QAM", |m| tx_qam_signal(m, 64), snrs),
        DetectorTest!("1024QAM", |m| tx_qam_signal(m, 1024), snrs),
        // BFSK
        DetectorTest!("BFSK", |m| tx_bfsk_signal(m, 16), snrs),
        // OFDM
        DetectorTest!(
            "OFDM-BPSK",
            |m| tx_ofdm_signal(tx_bpsk_signal(m), 16, 14),
            snrs
        ),
        DetectorTest!(
            "OFDM-QPSK",
            |m| tx_ofdm_signal(tx_qpsk_signal(m), 16, 14),
            snrs
        ),
        // Chirp Spread Spectrum
        DetectorTest!("CSS-16", |m| tx_css_signal(m, 16), snrs),
        DetectorTest!("CSS-128", |m| tx_css_signal(m, 128), snrs),
        DetectorTest!("CSS-512", |m| tx_css_signal(m, 512), snrs),
        // CSK
        DetectorTest!("CSK", tx_csk_signal, snrs),
        // FH-OFDM-DCSK
        DetectorTest!("FH-OFDM-DCSK", tx_fh_ofdm_dcsk_signal, snrs),
    ];

    let results: Vec<DetectorResults> = harness
        .into_iter()
        .map(|modulation| modulation.run())
        .collect();

    let regressions: Vec<LogRegressResults> = results
        // .into_par_iter()
        .iter()
        // .map(|test_result| log_regress(&test_result))
        .map(log_regress)
        .collect();

    Python::with_gil(|py| {
        let matplotlib = py.import_bound("matplotlib").unwrap();
        let plt = py.import_bound("matplotlib.pyplot").unwrap();
        let locals = [("matplotlib", matplotlib), ("plt", plt)].into_py_dict_bound(py);

        locals.set_item("snrs", &snrs).unwrap();
        locals.set_item("snrs_db", &snrs_db).unwrap();

        let (fig, axes): (&PyAny, &PyAny) = py
            .eval_bound("plt.subplots(1)", None, Some(&locals))
            .unwrap()
            .extract()
            .unwrap();
        locals.set_item("fig", fig).unwrap();
        locals.set_item("axes", axes).unwrap();

        // Plot the BER.
        for modulation in regressions.iter() {
            let py_name = format!("youden_js_{}", modulation.detector.name).to_case(Case::Snake);
            locals.set_item(&py_name, &modulation.youden_js).unwrap();
            py.eval_bound(
                &format!(
                    "axes.plot(snrs_db[:len({})], {}, label='{}')",
                    py_name, py_name, modulation.detector.name
                ),
                None,
                Some(&locals),
            )
            .unwrap();
        }
        for line in [
            "axes.legend(loc='best')",
            // "axes.set_yscale('log')",
            "axes.set_xlabel('SNR (dB)')",
            // "axes.set_ylabel('$P_d$')",
            "plt.show()",
        ] {
            println!("{}", line);
            py.eval_bound(line, None, Some(&locals)).unwrap();
        }
    });
}
