use std::ffi::CString;

use bazbandilo::ofdm::{rx_ofdm_signal, tx_ofdm_signal};
use bazbandilo::psk::{rx_qpsk_signal, tx_qpsk_signal};
use bazbandilo::Bit;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;
use rustfft::num_complex::{Complex, ComplexFloat};

#[macro_use]
mod util;

#[test]
#[ignore]
fn py_version() -> PyResult<()> {
    let num_bits = 2080;

    let mut rng = rand::thread_rng();
    let data: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

    let scs = 64;
    let pilots = 12;

    let tx_sig: Vec<Complex<f64>> =
        tx_ofdm_signal(tx_qpsk_signal(data.iter().cloned()), scs, pilots).collect();

    let rx_dat: Vec<Bit> =
        rx_qpsk_signal(rx_ofdm_signal(tx_sig.iter().cloned(), scs, pilots)).collect();

    Python::with_gil(|py| {
        let pathlib = py.import("pathlib")?;
        let importlib = py.import("importlib")?;
        let plt = py.import("matplotlib.pyplot")?;
        let np = py.import("numpy")?;

        let locals = [
            ("importlib", importlib),
            ("pathlib", pathlib),
            ("np", np),
            ("plt", plt),
        ]
        .into_py_dict(py)?;

        locals.set_item("tx_sig_rs", tx_sig.clone())?;

        let mod_path = py.eval(
            c!("pathlib.Path.home() / 'projects' / 'comms_py' / 'ofdm_trx5.py'"),
            None,
            Some(&locals),
        )?;
        locals.set_item("mod_path", mod_path)?;

        let comms_py = py.eval(
            c!("importlib.machinery.SourceFileLoader(mod_path.name, str(mod_path)).load_module()"),
            None,
            Some(&locals),
        )?;
        locals.set_item("comms_py", comms_py)?;

        locals.set_item("data", data.clone())?;
        let py_tx_sig_re: Vec<f64> = py
            .eval(
                c!("list(np.array(comms_py.tx_ofdm(data)).real)"),
                None,
                Some(&locals),
            )?
            .extract()?;
        let py_tx_sig_im: Vec<f64> /*&pyo3::PyAny*/ = py.eval(
            c!("list(np.array(comms_py.tx_ofdm(data)).imag)"),
            None,
            Some(&locals),
        )?.extract()?;

        let py_rx_dat: Vec<bool> = py
            .eval(
                c!("list(map(bool, comms_py.rx_ofdm(comms_py.tx_ofdm(data))))"),
                None,
                Some(&locals),
            )?
            .extract()?;

        let py_rx_dat_rs_py: Vec<bool> = py
            .eval(
                c!("list(map(bool, comms_py.rx_ofdm(tx_sig_rs)))"),
                None,
                Some(&locals),
            )?
            .extract()?;

        let py_tx_sig: Vec<Complex<f64>> = py_tx_sig_re
            .iter()
            .zip(py_tx_sig_im.iter())
            .map(|(&re, &im)| Complex::new(re, im))
            .collect();

        let t: Vec<f64> = (0..tx_sig.len())
            .map(|i| i as f64 /* sample_rate as f64*/)
            .collect();
        plot!(t, tx_sig, py_tx_sig, "/tmp/ofdm_py_vs_rs.png");

        let diff: Vec<f64> = py_tx_sig
            .iter()
            .zip(tx_sig.iter())
            .map(|(&ai, &bi)| (ai - bi).abs())
            .collect();
        plot!(t, diff, "/tmp/ofdm_py_vs_rs_diff.png");
        plot!(t, tx_sig, "/tmp/ofdm_tx_rs.png");
        plot!(t, py_tx_sig, "/tmp/ofdm_tx_py.png");

        assert_eq!(tx_sig.len(), py_tx_sig.len());
        assert_eq!(rx_dat.len(), py_rx_dat.len());

        assert_eq!(Complex::new(1f64, 1f64).abs(), std::f64::consts::SQRT_2);

        let tx_err: usize = tx_sig
            .iter()
            .zip(py_tx_sig.iter())
            .map(|(&i, &j)| if i == j { 0 } else { 1 })
            .sum();
        let rx_err: usize = rx_dat
            .iter()
            .zip(py_rx_dat.iter())
            .map(|(&i, &j)| if i == j { 0 } else { 1 })
            .sum();
        println!(
            "TX ERRs : {} ({}%)",
            tx_err,
            tx_err as f64 / tx_sig.len() as f64 * 100f64
        );
        println!(
            "RX ERRs : {} ({}%)",
            rx_err,
            rx_err as f64 / rx_dat.len() as f64 * 100f64
        );

        // assert_eq!(tx_sig, py_tx_sig);
        assert_eq!(rx_dat, py_rx_dat);
        assert_eq!(data, rx_dat);
        assert_eq!(data, py_rx_dat);
        assert_eq!(data, py_rx_dat_rs_py);

        PyResult::Ok(())
    })?;
    Ok(())
}
