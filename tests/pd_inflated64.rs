use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;

use bazbandilo::{
    awgn,
    cdma::{tx_cdma_bpsk_signal, tx_cdma_qpsk_signal},
    csk::tx_csk_signal,
    css::tx_css_signal,
    dcsk::{tx_dcsk_signal, tx_qcsk_signal},
    fh_ofdm_dcsk::tx_fh_ofdm_dcsk_signal,
    fsk::tx_bfsk_signal,
    hadamard::HadamardMatrix,
    iter::Iter,
    ofdm::tx_ofdm_signal,
    psk::{tx_bpsk_signal, tx_qpsk_signal},
    qam::tx_qam_signal,
    Bit,
};

use kdam::{par_tqdm, BarExt};
use num::Zero;
use num_complex::Complex;
use rand::Rng;
use rayon::prelude::*;

#[macro_use]
mod util;

#[macro_use]
mod pd;

use pd::{
    remap_results, run_detectors, snrs_lin, Detector, DetectorOutput, DetectorResults,
    DetectorTest, ModulationDetectorResults, NUM_ATTEMPTS, NUM_SAMPLES,
};

#[test]
fn inflated64() {
    let h_16 = HadamardMatrix::new(16);
    let h_32 = HadamardMatrix::new(32);
    let h_64 = HadamardMatrix::new(64);
    let key_16 = h_16.key(2);
    let key_32 = h_32.key(2);
    let key_64 = h_64.key(2);

    let harness = [
        // PSK
        DetectorTest!("BPSK", |m| tx_bpsk_signal(m).inflate(64), snrs_lin),
        DetectorTest!("QPSK", |m| tx_qpsk_signal(m).inflate(64), snrs_lin),
        // CDMA
        DetectorTest!(
            "CDMA-BPSK-16",
            |m| tx_cdma_bpsk_signal(m, key_16).inflate(4),
            snrs_lin
        ),
        DetectorTest!(
            "CDMA-QPSK-16",
            |m| tx_cdma_qpsk_signal(m, key_16).inflate(4),
            snrs_lin
        ),
        // DetectorTest!("CDMA-BPSK-32", |m| tx_cdma_bpsk_signal(m, key_32), snrs_lin),
        DetectorTest!(
            "CDMA-QPSK-32",
            |m| tx_cdma_qpsk_signal(m, key_32).inflate(2),
            snrs_lin
        ),
        // DetectorTest!("CDMA-BPSK-64", |m| tx_cdma_bpsk_signal(m, key_64), snrs_lin),
        DetectorTest!("CDMA-QPSK-64", |m| tx_cdma_qpsk_signal(m, key_64), snrs_lin),
        // QAM
        // DetectorTest!("4QAM", |m| tx_qam_signal(m, 4), snrs_lin),
        DetectorTest!("16QAM", |m| tx_qam_signal(m, 16).inflate(4), snrs_lin),
        DetectorTest!("64QAM", |m| tx_qam_signal(m, 64), snrs_lin),
        // BFSK
        DetectorTest!("BFSK-16", |m| tx_bfsk_signal(m, 16).inflate(4), snrs_lin),
        DetectorTest!("BFSK-32", |m| tx_bfsk_signal(m, 32).inflate(2), snrs_lin),
        DetectorTest!("BFSK-64", |m| tx_bfsk_signal(m, 64), snrs_lin),
        // OFDM
        DetectorTest!(
            "OFDM-BPSK-16",
            |m| tx_ofdm_signal(tx_bpsk_signal(m), 16, 0).inflate(4),
            snrs_lin
        ),
        DetectorTest!(
            "OFDM-QPSK-16",
            |m| tx_ofdm_signal(tx_qpsk_signal(m), 16, 0).inflate(4),
            snrs_lin
        ),
        DetectorTest!(
            "OFDM-BPSK-64",
            |m| tx_ofdm_signal(tx_bpsk_signal(m), 64, 0),
            snrs_lin
        ),
        DetectorTest!(
            "OFDM-QPSK-64",
            |m| tx_ofdm_signal(tx_qpsk_signal(m), 64, 0),
            snrs_lin
        ),
        // Chirp Spread Spectrum
        DetectorTest!("CSS-16", |m| tx_css_signal(m, 16).inflate(4), snrs_lin),
        DetectorTest!("CSS-64", |m| tx_css_signal(m, 64), snrs_lin),
        // DetectorTest!("CSS-128", |m| tx_css_signal(m, 128), snrs_lin),
        // CSK
        DetectorTest!("CSK", |m| tx_csk_signal(m).inflate(64), snrs_lin),
        DetectorTest!("DCSK", |m| tx_dcsk_signal(m).inflate(32), snrs_lin),
        DetectorTest!("QCSK", |m| tx_qcsk_signal(m).inflate(32), snrs_lin),
        // FH-OFDM-DCSK
        DetectorTest!("FH-OFDM-DCSK", tx_fh_ofdm_dcsk_signal, snrs_lin),
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
