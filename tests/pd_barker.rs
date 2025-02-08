use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;

use bazbandilo::{awgn, barker::get_barker_code, cdma::tx_cdma_bpsk_signal, random_bits};

use kdam::{par_tqdm, BarExt};
use num::Zero;
use num_complex::Complex;
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
fn barker() {
    let key_5 = get_barker_code(5).unwrap();
    let key_7 = get_barker_code(7).unwrap();
    let key_11 = get_barker_code(11).unwrap();
    let key_13 = get_barker_code(13).unwrap();

    let harness = [
        DetectorTest!("CDMA-BPSK-5", |m| tx_cdma_bpsk_signal(m, &key_5), snrs_lin),
        DetectorTest!("CDMA-BPSK-7", |m| tx_cdma_bpsk_signal(m, &key_7), snrs_lin),
        DetectorTest!(
            "CDMA-BPSK-11",
            |m| tx_cdma_bpsk_signal(m, &key_11),
            snrs_lin
        ),
        DetectorTest!(
            "CDMA-BPSK-13",
            |m| tx_cdma_bpsk_signal(m, &key_13),
            snrs_lin
        ),
    ];

    let results: Vec<ModulationDetectorResults> = {
        let mut results = Vec::with_capacity(harness.len());
        for modulation in harness {
            let result = modulation.run();
            results.push(result);

            // Save results to file.
            {
                let name = "/tmp/results_barker.json";
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
