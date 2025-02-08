use std::ffi::CString;

use bazbandilo::{
    awgn, bit_to_nrz,
    cdma::{rx_cdma_bpsk_signal, tx_cdma_bpsk_signal},
    hadamard::HadamardMatrix,
    iter::Iter,
    psk::tx_bpsk_signal,
    random_bits, Bit,
};

use num::complex::Complex;
use num::Zero;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rayon::prelude::*;

#[macro_use]
mod util;

#[test]
fn cdma_graphs() {
    let data: Vec<Bit> = vec![true, true, false, false, true];

    let h = HadamardMatrix::new(32);
    let key = h.key(2);

    let tx: Vec<Complex<f64>> = tx_cdma_bpsk_signal(data.iter().cloned(), key).collect();

    let rx_clean: Vec<Bit> = rx_cdma_bpsk_signal(tx.iter().cloned(), key).collect();

    let sigma = 2f64;
    let noisy_signal: Vec<Complex<f64>> = awgn(tx.iter().cloned(), sigma).collect();

    let rx_dirty: Vec<Bit> = rx_cdma_bpsk_signal(noisy_signal.iter().cloned(), key).collect();

    let t: Vec<f64> = (0..tx.len())
        .map(|idx| {
            // let time_step = symbol_rate as f64 / samp_rate as f64;
            idx as f64
        })
        .collect();

    let bpsk_tx: Vec<Complex<f64>> = tx_bpsk_signal(data.iter().cloned()).collect();
    let t2: Vec<f64> = (0..bpsk_tx.len())
        .map(|idx| {
            // let time_step = symbol_rate as f64 / samp_rate as f64;
            idx as f64
        })
        .collect();
    // assert_eq!(bpsk_tx.len(), tx.len());
    plot!(t2, bpsk_tx, "/tmp/cdma_tx_bpsk.png");
    plot!(t, tx, "/tmp/cdma_tx.png");
    plot!(t, tx, noisy_signal, "/tmp/cdma_tx_awgn.png");

    // assert!(save_vector2(&tx, &t, "/tmp/cdma_bpsk.csv").is_ok());
    // plot!(t, rx_clean, rx_dirty, "/tmp/cdma_rx_awgn.png");
    println!("ERROR: {}", error!(rx_clean, rx_dirty));
    assert!(error!(rx_clean, rx_dirty) <= 0.2);
    assert_eq!(rx_clean, rx_dirty);
}

#[test]
fn python_plotz() -> PyResult<()> {
    let data: Vec<Bit> = vec![true, true, false, false, true];
    let samp_rate = 80_000; // Clock rate for both RX and TX.
    let symbol_rate = 1000; // Rate symbols come out the things.
    let carrier_freq = 2500_f64;

    let h = HadamardMatrix::new(8);
    let key = h.key(2);

    let data_tx: Vec<f64> = data
        .iter()
        .cloned()
        .map(bit_to_nrz)
        .inflate(samp_rate / symbol_rate)
        .collect();

    let bpsk_tx: Vec<Complex<f64>> = tx_bpsk_signal(data.iter().cloned()).collect();

    let cdma_tx: Vec<Complex<f64>> = tx_cdma_bpsk_signal(data.iter().cloned(), key).collect();

    let t_step: f64 = 1f64 / (samp_rate as f64);

    Python::with_gil(|py| {
        let plt = py.import("matplotlib.pyplot")?;
        let np = py.import("numpy")?;
        let locals = [("np", np), ("plt", plt)].into_py_dict(py)?;

        locals.set_item("data_tx", data_tx)?;
        locals.set_item("bpsk_tx", bpsk_tx)?;
        locals.set_item("cdma_tx", cdma_tx)?;
        locals.set_item("dt", t_step)?;

        let x = py.eval(
            c!("lambda s, dt: [dt * i for i in range(len(s))]"),
            None,
            None,
        )?;
        locals.set_item("x", x)?;

        let (fig, axes): (PyObject, PyObject) = py
            .eval(c!("plt.subplots(4)"), None, Some(&locals))?
            .extract()?;
        locals.set_item("fig", fig)?;
        locals.set_item("axes", axes)?;

        for line in [
            c!("axes[0].plot(x(data_tx, dt), data_tx, label='DATA: [1 1 0 0 1]')"),
            c!("axes[1].plot(x(bpsk_tx, dt), bpsk_tx, label='BPSK: (2.5kHz, 1kHz Data Rate)')"),
            c!("axes[2].plot(x(cdma_tx, dt), cdma_tx, label='CDMA: 8kHz Chip Rate')"),
        ] {
            py.eval(line, None, Some(&locals))?;
        }

        locals.set_item("samp_rate", samp_rate)?;
        locals.set_item("freq", carrier_freq)?;

        for line in [
            c!("plt.psd(bpsk_tx, Fs=samp_rate, Fc=freq)"),
            c!("plt.psd(cdma_tx, Fs=samp_rate, Fc=freq)"),
            c!("[x.legend() for x in axes[:-1]]"),
            c!("fig.set_size_inches(16, 9)"),
            // c!("plt.show()"),
            c!("fig.savefig('/tmp/cdma_works.png', dpi=300)"),
        ] {
            py.eval(line, None, Some(&locals))?;
        }

        Ok(())
    })
}

#[test]
#[ignore]
fn mai_plot() {
    let num_users = 63;
    let keysize = num_users + 1;

    // Simulation parameters.
    let num_bits = 1000; // How many bits to transmit overall.
    let num_samples = num_bits * keysize;

    // The data each user will transmit.
    let datas: Vec<Vec<Bit>> = (0..num_users).map(|_| random_bits(num_bits)).collect();

    // The keys each user will use.
    let walsh_codes = HadamardMatrix::new(keysize);
    let keys: Vec<Vec<Bit>> = (0..num_users)
        .map(|idx| walsh_codes.key(idx).clone())
        .collect();

    // Calculate BER as a function of users.
    let bers: Vec<f64> = (0..num_users)
        .into_par_iter()
        .map(|user_count| {
            // The channel comprises of `user_count` users' CDMA-BPSK signals added.
            let channel: Vec<Complex<f64>> = datas
                .iter()
                .take(user_count + 1)
                .zip(keys.iter())
                .map(|(data, key)| tx_cdma_bpsk_signal(data.iter().cloned(), key))
                .fold(vec![Complex::zero(); num_samples], |mut acc, tx| {
                    acc.iter_mut().zip(tx).for_each(|(s_i, tx_i)| *s_i += tx_i);
                    acc
                });

            // Find the BER for each user, then average.
            let bers: Vec<f64> = (0..user_count)
                .map(|idx| {
                    let rx: Vec<Bit> =
                        rx_cdma_bpsk_signal(channel.iter().cloned(), &keys[idx]).collect();

                    let errors: usize = rx
                        .iter()
                        .zip(&datas[idx])
                        .map(|(rxi, txi)| if rxi == txi { 0 } else { 1 })
                        .sum();
                    errors as f64 / rx.len() as f64
                })
                .collect();
            bers.iter().sum::<f64>() / bers.len() as f64
        })
        .collect();

    let x: Vec<f64> = (0..bers.len()).map(|i| (i + 1) as f64).collect();
    plot!(x, bers, "/tmp/cdma_mai.png");
}
