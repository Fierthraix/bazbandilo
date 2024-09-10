use bazbandilo::erfc;

#[macro_export]
macro_rules! bers {
    ($ber_fn:expr, $snrs:expr) => {
        $snrs.iter().cloned().map($ber_fn).collect::<Vec<f64>>()
    };
}

pub fn is_int(num: f64) -> bool {
    num == (num as u64) as f64
}

pub fn ber_bpsk(eb_n0: f64) -> f64 {
    0.5 * erfc((eb_n0).sqrt())
}

pub fn ber_qpsk(eb_n0: f64) -> f64 {
    0.5 * erfc((eb_n0).sqrt()) - 0.25 * erfc((eb_n0).sqrt()).powi(2)
}

pub fn ber_bfsk(eb_n0: f64) -> f64 {
    0.5 * erfc((eb_n0 / 2f64).sqrt())
}

pub fn ber_qam(eb_n0: f64, m: usize) -> f64 {
    let m = m as f64;
    assert!(is_int(m.log(4f64)));
    2f64 / m.log2()
        * (1f64 - 1f64 / (m).sqrt())
        * erfc((3f64 * eb_n0 * m.log2()) / (2f64 * (m - 1f64)))
}
