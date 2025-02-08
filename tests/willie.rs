use std::ffi::CString;

use bazbandilo::{
    avg_energy, awgn, cdma::tx_cdma_bpsk_signal, hadamard::HadamardMatrix, linspace,
    psk::tx_bpsk_signal, random_bits, Bit,
};

use num::complex::Complex;
use num::Zero;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

#[macro_use]
mod util;

#[test]
#[ignore]
fn p_values() -> PyResult<()> {
    let num_bits = 10_000;
    let data: Vec<Bit> = random_bits(num_bits);

    let matrix_size = 16;
    let walsh_codes = HadamardMatrix::new(matrix_size);
    let key = walsh_codes.key(0);

    let tx_cdma: Vec<Complex<f64>> = tx_cdma_bpsk_signal(data.iter().cloned(), key).collect();

    let tx_bpsk: Vec<Complex<f64>> = tx_bpsk_signal(data.iter().cloned()).collect();

    Python::with_gil(|py| {
        let scipy = py.import("scipy")?;
        let matplotlib = py.import("matplotlib")?;
        let plt = py.import("matplotlib.pyplot")?;
        let locals =
            [("scipy", scipy), ("matplotlib", matplotlib), ("plt", plt)].into_py_dict(py)?;
        py.eval(c!("matplotlib.use('agg')"), None, Some(&locals))?;

        locals.set_item("tx_bpsk", tx_bpsk)?;
        locals.set_item("tx_cdma", tx_cdma)?;

        // Check their p-values !!
        let p_vals: Py<PyAny> = PyModule::from_code(
            py,
            c!("def p_vals(signal):
                from random import gauss
                import numpy
                import scipy
                def awgn(signal, n0):
                    return [s_i + gauss(sigma=n0) for s_i in signal]

                N0s = numpy.arange(1e-10, 5, 0.1)
                p_vals = []
                for N0 in N0s:
                    chan_sig = awgn(signal, N0)
                    res = scipy.stats.normaltest(chan_sig)
                    p_vals.append(res.pvalue)

                return p_vals
            "),
            c!(""),
            c!(""),
        )?
        .getattr("p_vals")?
        .into();

        locals.set_item("p_vals", p_vals)?;

        let cdma_p_vals: Vec<f64> = py
            .eval(c!("p_vals(tx_cdma)"), None, Some(&locals))?
            .extract()?;
        let bpsk_p_vals: Vec<f64> = py
            .eval(c!("p_vals(tx_bpsk)"), None, Some(&locals))?
            .extract()?;
        let x: Vec<f64> = linspace(1e-10, 5f64, cdma_p_vals.len()).collect();
        plot!(x, bpsk_p_vals, cdma_p_vals, "/tmp/willie_001.png");
        println!("{}", cdma_p_vals.len());

        PyResult::Ok(())
    })?;

    Ok(())
}

#[test]
fn energy_detector() {
    let num_samples = 10_000;

    let n0s: Vec<f64> = linspace(1e-3, 10f64, 100).collect();

    let n0_estimates: Vec<f64> = n0s
        .iter()
        .map(|&n0| {
            avg_energy(
                &awgn((0..num_samples).map(|_| Complex::zero()), n0).collect::<Vec<Complex<f64>>>(),
            )
            .sqrt()
        })
        .collect();

    plot!(n0s, n0_estimates, "/tmp/energy_detector_estimates");
}
