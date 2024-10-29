use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;

use util::ber::{ber_bfsk, ber_bpsk, ber_qam, ber_qpsk};

use convert_case::{Case, Casing};
use kdam::{par_tqdm, BarExt};
use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[macro_use]
mod util;

use bazbandilo::{
    awgn, bit_to_nrz,
    cdma::{rx_cdma_bpsk_signal, rx_cdma_qpsk_signal, tx_cdma_bpsk_signal, tx_cdma_qpsk_signal},
    csk::{rx_csk_signal, tx_csk_signal},
    css::{rx_css_signal, tx_css_signal},
    dcsk::{rx_dcsk_signal, rx_qcsk_signal, tx_dcsk_signal, tx_qcsk_signal},
    fh_ofdm_dcsk::{rx_fh_ofdm_dcsk_signal, tx_fh_ofdm_dcsk_signal},
    fsk::{rx_bfsk_signal, tx_bfsk_signal},
    hadamard::HadamardMatrix,
    iter::Iter,
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

#[derive(Serialize, Deserialize, Debug)]
struct BitErrorResults {
    name: String,
    bers: Vec<f64>,
    snrs: Vec<f64>,
}

struct BitErrorTest<'a> {
    name: String,
    snrs: Vec<f64>,
    calc_ber_fn: &'a dyn Fn(&[f64]) -> Vec<f64>,
}

impl<'a> BitErrorTest<'a> {
    fn calc_bers(self) -> BitErrorResults {
        let bers: Vec<f64> = (self.calc_ber_fn)(&self.snrs);
        let snrs = self.snrs[..bers.len()].to_vec();
        BitErrorResults {
            name: self.name,
            bers,
            snrs,
        }
    }
}

const NUM_ERRORS: usize = 100_000;
// const NUM_ERRORS: usize = 100;
const BER_CUTOFF: f64 = 10e-4;
// const BER_CUTOFF: f64 = 10e-5;

macro_rules! BitErrorTest {
    ($name:expr, $tx_fn:expr, $rx_fn:expr, $snrs:expr) => {{
        BitErrorTest {
            name: String::from($name),
            snrs: $snrs.clone(),
            calc_ber_fn: &|snrs: &[f64]| {
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

                let mut bers: Vec<f64> = Vec::with_capacity(snrs.len());
                let n0s = snrs.iter().map(|snr| (eb / (2f64 * snr)).sqrt());

                for n0 in n0s {
                    let mut errors = 0;
                    let mut num_total_bits = 0;
                    let mut curr_ber = 0.5;

                    while errors < NUM_ERRORS && curr_ber >= BER_CUTOFF {
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
                        curr_ber = errors as f64 / num_total_bits as f64;
                    }

                    let mut pb = pb.lock().unwrap();
                    pb.update(1).unwrap();
                    bers.push(curr_ber);
                    if curr_ber <= BER_CUTOFF {
                        break;
                    }
                }

                println!("Finished with {}.", $name);

                bers
            },
        }
    }};
}

macro_rules! rx_inflated {
    ($rx:expr, $signal:expr, $chunks:expr) => {{
        $rx($signal)
            .chunks($chunks)
            .map(|r_i| r_i.into_iter().map(bit_to_nrz).sum::<f64>() > 0f64)
    }};
}

fn rx_inflated_bpsk_signal<I: Iterator<Item = Complex<f64>>>(
    signal: I,
    chunks: usize,
) -> impl Iterator<Item = Bit> {
    rx_bpsk_signal(
        signal
            .chunks(chunks)
            .map(|chunk| chunk.iter().sum::<Complex<f64>>() / chunk.len() as f64),
    )
}

#[test]
fn main() {
    // let snrs_db: Vec<f64> = linspace(-25f64, 10f64, 35).collect();
    // let snrs_db: Vec<f64> = linspace(-25f64, 6f64, 25).collect();
    // let snrs_db: Vec<f64> = linspace(-25f64, 12f64, 50).collect();
    // let snrs_db: Vec<f64> = linspace(-25f64, 10f64, 50).collect();
    let snrs_db: Vec<f64> = linspace(-45f64, 12f64, 150).collect();
    // let snrs_db: Vec<f64> = linspace(0.0, 13f64, 25).collect();

    let snrs: Vec<f64> = snrs_db.iter().cloned().map(undb).collect();

    let h_16 = HadamardMatrix::new(16);
    let h_32 = HadamardMatrix::new(32);
    let h_64 = HadamardMatrix::new(64);
    let key_16 = h_16.key(2);
    let key_32 = h_32.key(2);
    let key_64 = h_64.key(2);

    let bers = [
        // PSK
        BitErrorTest!("BPSK", tx_bpsk_signal, rx_bpsk_signal, snrs),
        BitErrorTest!(
            "BPSK-16",
            |m| tx_bpsk_signal(m).inflate(16),
            |s| rx_inflated_bpsk_signal(s, 16),
            snrs
        ),
        BitErrorTest!(
            "BPSK-32",
            |m| tx_bpsk_signal(m).inflate(32),
            |s| rx_inflated_bpsk_signal(s, 32),
            snrs
        ),
        BitErrorTest!(
            "BPSK-64",
            |m| tx_bpsk_signal(m).inflate(64),
            |s| rx_inflated_bpsk_signal(s, 64),
            snrs
        ),
        BitErrorTest!(
            "BPSK-16-AVG",
            |m| tx_bpsk_signal(m).inflate(16),
            |s| rx_inflated!(rx_bpsk_signal, s, 16),
            snrs
        ),
        BitErrorTest!(
            "BPSK-32-AVG",
            |m| tx_bpsk_signal(m).inflate(32),
            |s| rx_inflated!(rx_bpsk_signal, s, 32),
            snrs
        ),
        BitErrorTest!(
            "BPSK-64-AVG",
            |m| tx_bpsk_signal(m).inflate(64),
            |s| rx_inflated!(rx_bpsk_signal, s, 64),
            snrs
        ),
        // BitErrorTest!("QPSK", tx_qpsk_signal, rx_qpsk_signal, snrs),
        // CDMA
        BitErrorTest!(
            "CDMA-BPSK-16",
            |m| tx_cdma_bpsk_signal(m, key_16),
            |s| rx_cdma_bpsk_signal(s, key_16),
            snrs
        ),
        BitErrorTest!(
            "CDMA-BPSK-32",
            |m| tx_cdma_bpsk_signal(m, key_32),
            |s| rx_cdma_bpsk_signal(s, key_32),
            snrs
        ),
        BitErrorTest!(
            "CDMA-BPSK-64",
            |m| tx_cdma_bpsk_signal(m, key_64),
            |s| rx_cdma_bpsk_signal(s, key_64),
            snrs
        ),
        /*
        BitErrorTest!(
            "CDMA-QPSK-16",
            |m| tx_cdma_qpsk_signal(m, key_16),
            |s| rx_cdma_qpsk_signal(s, key_16),
            snrs
        ),
        BitErrorTest!(
            "CDMA-QPSK-32",
            |m| tx_cdma_qpsk_signal(m, key_32),
            |s| rx_cdma_qpsk_signal(s, key_32),
            snrs
        ),
        BitErrorTest!(
            "CDMA-QPSK-64",
            |m| tx_cdma_qpsk_signal(m, key_64),
            |s| rx_cdma_qpsk_signal(s, key_64),
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
            "BFSK-16",
            |m| tx_bfsk_signal(m, 16),
            |s| rx_bfsk_signal(s, 16),
            snrs
        ),
        BitErrorTest!(
            "BFSK-32",
            |m| tx_bfsk_signal(m, 32),
            |s| rx_bfsk_signal(s, 32),
            snrs
        ),
        BitErrorTest!(
            "BFSK-64",
            |m| tx_bfsk_signal(m, 64),
            |s| rx_bfsk_signal(s, 64),
            snrs
        ),
        // OFDM
        BitErrorTest!(
            "OFDM-BPSK-2-16",
            |m| tx_ofdm_signal(tx_bpsk_signal(m), 16, 14),
            |s| rx_bpsk_signal(rx_ofdm_signal(s, 16, 14)),
            snrs
        ),
        BitErrorTest!(
            "OFDM-QPSK-2-16",
            |m| tx_ofdm_signal(tx_qpsk_signal(m), 16, 14),
            |s| rx_qpsk_signal(rx_ofdm_signal(s, 16, 14)),
            snrs
        ),
        BitErrorTest!(
            "OFDM-BPSK-8-16",
            |m| tx_ofdm_signal(tx_bpsk_signal(m), 16, 8),
            |s| rx_bpsk_signal(rx_ofdm_signal(s, 16, 8)),
            snrs
        ),
        BitErrorTest!(
            "OFDM-QPSK-8-16",
            |m| tx_ofdm_signal(tx_qpsk_signal(m), 16, 8),
            |s| rx_qpsk_signal(rx_ofdm_signal(s, 16, 8)),
            snrs
        ),
        BitErrorTest!(
            "OFDM-BPSK-2-64",
            |m| tx_ofdm_signal(tx_bpsk_signal(m), 64, 62),
            |s| rx_bpsk_signal(rx_ofdm_signal(s, 64, 62)),
            snrs
        ),
        BitErrorTest!(
            "OFDM-QPSK-2-64",
            |m| tx_ofdm_signal(tx_qpsk_signal(m), 64, 62),
            |s| rx_qpsk_signal(rx_ofdm_signal(s, 64, 62)),
            snrs
        ),
        BitErrorTest!(
            "OFDM-BPSK-32-64",
            |m| tx_ofdm_signal(tx_bpsk_signal(m), 64, 32),
            |s| rx_bpsk_signal(rx_ofdm_signal(s, 64, 32)),
            snrs
        ),
        BitErrorTest!(
            "OFDM-QPSK-32-64",
            |m| tx_ofdm_signal(tx_qpsk_signal(m), 64, 32),
            |s| rx_qpsk_signal(rx_ofdm_signal(s, 64, 32)),
            snrs
        ),
        // Chirp Spread Spectrum
        BitErrorTest!(
            "CSS-16",
            |m| tx_css_signal(m, 16),
            |s| rx_css_signal(s, 16),
            snrs
        ),
        BitErrorTest!(
            "CSS-128",
            |m| tx_css_signal(m, 128),
            |s| rx_css_signal(s, 128),
            snrs
        ),
        BitErrorTest!(
            "CSS-512",
            |m| tx_css_signal(m, 512),
            |s| rx_css_signal(s, 512),
            snrs
        ),
        // CSK
        BitErrorTest!("CSK", tx_csk_signal, rx_csk_signal, snrs),
        BitErrorTest!("DCSK", tx_dcsk_signal, rx_dcsk_signal, snrs),
        BitErrorTest!("QCSK", tx_qcsk_signal, rx_qcsk_signal, snrs),
        // FH-OFDM-DCSK
        BitErrorTest!(
            "FH-OFDM-DCSK",
            tx_fh_ofdm_dcsk_signal,
            rx_fh_ofdm_dcsk_signal,
            snrs
        ),
        */
    ];

    let bers: Vec<BitErrorResults> = bers.into_iter().map(|test| test.calc_bers()).collect();

    let theory_bers = [
        ("BPSK", bers!(ber_bpsk, snrs)),
        ("QPSK", bers!(ber_qpsk, snrs)),
        ("FSK", bers!(ber_bfsk, snrs)),
        ("16QAM", bers!(|snr| ber_qam(snr, 16), snrs)),
    ];

    {
        // Save the results to a JSON file.
        let file = File::create("/tmp/bers.json").unwrap();
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &bers).unwrap();
        writer.flush().unwrap();
    }

    {
        // Save the results to a MessagePack file.
        let file = File::create("/tmp/bers.msgpack").unwrap();
        let mut writer = BufWriter::new(file);
        writer
            // .write_all(&rmp_serde::to_vec_named(&bers).unwrap())
            .write_all(&rmp_serde::to_vec(&bers).unwrap())
            .unwrap();
        writer.flush().unwrap();
    }

    /*
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
        for ber_result in bers.iter() {
            let py_name = format!("bers_{}", ber_result.name).to_case(Case::Snake);
            locals.set_item(&py_name, &ber_result.bers).unwrap();
            py.eval_bound(
                &format!(
                    "axes.plot(snrs_db[:len({})], {}, label='{}')",
                    py_name, py_name, ber_result.name
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
            // "axes.plot(snrs_db, bpsk_theory, label='BPSK Theoretical')",
            "axes.legend(loc='best')",
            "axes.set_yscale('log')",
            "axes.set_xlabel('SNR (dB)')",
            "axes.set_ylabel('BER')",
            "plt.show()",
        ] {
            py.eval_bound(line, None, Some(&locals)).unwrap();
        }
    });
    */
}
