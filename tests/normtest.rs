#![allow(unused_variables, dead_code, non_snake_case)]
use average::{concatenate, Estimate, Kurtosis, Skewness};

concatenate!(NormTest, [Kurtosis, kurtosis], [Skewness, skewness]);

fn normtest(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;

    let kurtskew: NormTest = data.iter().cloned().collect();
    let b2 = kurtskew.kurtosis.kurtosis();

    let E = 3f64 * (n - 1f64) / (n + 1f64);
    let varb2 =
        24f64 * n * (n - 2f64) * (n - 3f64) / ((n + 1f64) * (n + 1f64) * (n + 3f64) * (n + 5f64)); // [1]_ Eq. 1
    let x = (b2 - E) / varb2.sqrt(); // [1]_ Eq. 4
                                     // [1]_ Eq. 2:
    let sqrtbeta1 = 6f64 * (n * n - 5f64 * n + 2f64) / ((n + 7f64) * (n + 9f64))
        * ((6.0 * (n + 3f64) * (n + 5f64)) / (n * (n - 2f64) * (n - 3f64))).sqrt();
    // [1]_ Eq. 3:
    let A =
        6f64 + 8f64 / sqrtbeta1 * (2f64 / sqrtbeta1 + (1f64 + 4f64 / (sqrtbeta1.powi(2))).sqrt());
    let term1 = 1f64 - 2f64 / (9f64 * A);
    let denom = 1f64 + x * (2f64 / (A - 4f64)).sqrt();
    let term2 = denom.signum() * (1f64 - 2f64 / A) / denom.abs().powf(1f64 / 3f64);
    // let term2 = np.sign(denom) * np.where(denom == 0.0, np.nan,np.power((1-2.0/A)/np.abs(denom), 1/3.0));
    /*
    if np.any(denom == 0):
        msg = ("Test statistic not defined in some cases due to division by "
               "zero. Return nan in that case...")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    */
    let Z = (term1 - term2) / (2f64 / (9f64 * A)).sqrt(); // [1]_ Eq. 5

    (0.0, 0f64)
}
