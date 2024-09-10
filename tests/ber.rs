use std::sync::Mutex;

use util::ber::{ber_bfsk, ber_bpsk, ber_qam, ber_qpsk};

use convert_case::{Case, Casing};
use kdam::{par_tqdm, BarExt};
use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;
use rayon::prelude::*;

#[macro_use]
mod util;

use bazbandilo::{
    awgn,
    cdma::{rx_cdma_bpsk_signal, rx_cdma_qpsk_signal, tx_cdma_bpsk_signal, tx_cdma_qpsk_signal},
    csk::{rx_baseband_csk_signal, tx_baseband_csk_signal},
    fh_ofdm_dcsk::{rx_fh_ofdm_dcsk_signal, tx_fh_ofdm_dcsk_signal},
    fsk::{rx_bfsk_signal, tx_bfsk_signal},
    hadamard::HadamardMatrix,
    linspace,
    ofdm::{rx_ofdm_signal, tx_ofdm_signal},
    psk::{rx_bpsk_signal, rx_qpsk_signal, tx_bpsk_signal, tx_qpsk_signal},
    qam::{rx_qam_signal, tx_qam_signal},
    undb, Bit,
};

fn random_data(num_bits: usize) -> Vec<Bit> {
    let mut rng = rand::thread_rng();
    (0..num_bits).map(|_| rng.gen::<Bit>()).collect()
}

struct BitErrorResults {
    name: String,
    bers: Vec<f64>,
}

const NUM_ERRORS: usize = 10_000;
// const NUM_ERRORS: usize = 100;

macro_rules! BitErrorTest {
    ($name:expr, $tx_fn:expr, $rx_fn:expr, $snrs:expr) => {{
        let eb = {
            let num_bits = 65536;

            let data = random_data(num_bits);
            let tx_signal: Vec<Complex<f64>> = $tx_fn(data.iter().cloned()).collect();

            let energy: f64 = tx_signal.iter().map(|&s_i| s_i.norm_sqr()).sum();
            let num_bits_received: usize =
                $rx_fn(tx_signal.iter().cloned()).fold(0, |acc, _| acc + 1);

            energy / num_bits_received as f64
        };

        let mut pb = par_tqdm!(total = $snrs.len());
        pb.refresh().unwrap();
        pb.set_description($name);
        let pb = Mutex::new(pb);

        let n0s = $snrs.par_iter().map(|snr| (eb / (2f64 * snr)).sqrt());
        let bers: Vec<f64> = n0s
            .map(|n0| {
                let mut errors = 0;
                let mut num_total_bits = 0;

                while errors < NUM_ERRORS {
                    let num_bits = 9088;
                    let parallel = num_cpus::get();
                    errors += (0..parallel)
                        .into_par_iter()
                        .map(|_| {
                            let data = random_data(num_bits);
                            let tx_signal = $tx_fn(data.iter().cloned());
                            let rx_signal = $rx_fn(awgn(tx_signal, n0));

                            rx_signal
                                .zip(data.iter())
                                .map(|(r_i, &d_i)| if d_i == r_i { 0 } else { 1 })
                                .sum::<usize>()
                        })
                        .sum::<usize>();

                    num_total_bits += parallel * num_bits;
                }

                let mut pb = pb.lock().unwrap();
                pb.update(1).unwrap();
                errors as f64 / num_total_bits as f64
            })
            .collect();

        println!("Finished with {}.", $name);
        BitErrorResults {
            name: String::from($name),
            bers,
        }
    }};
}

#[test]
fn main() {
    // let snrs_db: Vec<f64> = linspace(0f64, 10f64, 25).collect();
    let snrs_db: Vec<f64> = linspace(-25f64, 6f64, 25).collect();
    // let snrs_db: Vec<f64> = linspace(-25f64, 12f64, 50).collect();

    let snrs: Vec<f64> = snrs_db.iter().cloned().map(undb).collect();

    let h = HadamardMatrix::new(32);
    let key = h.key(2);

    let bers = [
        // PSK
        BitErrorTest!("BPSK", tx_bpsk_signal, rx_bpsk_signal, snrs),
        BitErrorTest!("QPSK", tx_qpsk_signal, rx_qpsk_signal, snrs),
        // CDMA
        BitErrorTest!(
            "CDMA-BPSK",
            |m| tx_cdma_bpsk_signal(m, key),
            |s| rx_cdma_bpsk_signal(s, key),
            snrs
        ),
        BitErrorTest!(
            "CDMA-QPSK",
            |m| tx_cdma_qpsk_signal(m, key),
            |s| rx_cdma_qpsk_signal(s, key),
            snrs
        ),
        // QAM
        BitErrorTest!(
            "4QAM",
            |m| tx_qam_signal(m, 4),
            |s| rx_qam_signal(s, 4),
            snrs
        ),
        BitErrorTest!(
            "16QAM",
            |m| tx_qam_signal(m, 16),
            |s| rx_qam_signal(s, 16),
            snrs
        ),
        BitErrorTest!(
            "64QAM",
            |m| tx_qam_signal(m, 64),
            |s| rx_qam_signal(s, 64),
            snrs
        ),
        BitErrorTest!(
            "1024QAM",
            |m| tx_qam_signal(m, 1024),
            |s| rx_qam_signal(s, 1024),
            snrs
        ),
        // BFSK
        BitErrorTest!(
            "BFSK",
            |m| tx_bfsk_signal(m, 16),
            |s| rx_bfsk_signal(s, 16),
            snrs
        ),
        // OFDM
        BitErrorTest!(
            "OFDM-BPSK",
            |m| tx_ofdm_signal(tx_bpsk_signal(m), 16, 14),
            |s| rx_bpsk_signal(rx_ofdm_signal(s, 16, 14)),
            snrs
        ),
        BitErrorTest!(
            "OFDM-QPSK",
            |m| tx_ofdm_signal(tx_qpsk_signal(m), 16, 14),
            |s| rx_qpsk_signal(rx_ofdm_signal(s, 16, 14)),
            snrs
        ),
        // CSK
        BitErrorTest!("CSK", tx_baseband_csk_signal, rx_baseband_csk_signal, snrs),
        // FH-OFDM-DCSK
        BitErrorTest!(
            "FH-OFDM-DCSK",
            tx_fh_ofdm_dcsk_signal,
            rx_fh_ofdm_dcsk_signal,
            snrs
        ),
    ];

    let theory_bers = [
        ("BPSK", bers!(ber_bpsk, snrs)),
        ("QPSK", bers!(ber_qpsk, snrs)),
        ("FSK", bers!(ber_bfsk, snrs)),
        ("16QAM", bers!(|snr| ber_qam(snr, 16), snrs)),
    ];

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
        for ber_result in bers {
            let py_name = format!("bers_{}", ber_result.name).to_case(Case::Snake);
            locals.set_item(&py_name, &ber_result.bers).unwrap();
            py.eval_bound(
                &format!(
                    "axes.plot(snrs_db, {}, label='{}')",
                    py_name, ber_result.name
                ),
                None,
                Some(&locals),
            )
            .unwrap();
        }
        locals
            .set_item("bpsk_theory", theory_bers[0].1.clone())
            .unwrap();

        for line in [
            "axes.plot(snrs_db, bpsk_theory, label='BPSK Theoretical')",
            "axes.legend(loc='best')",
            "axes.set_yscale('log')",
            "plt.show()",
        ] {
            py.eval_bound(line, None, Some(&locals)).unwrap();
        }
    })
}
