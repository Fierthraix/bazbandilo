use crate::linspace;

use rand::prelude::*;

const SEED: u64 = 64;

/// Hop Table
/// As an iterator, it is constantly shuffling the hop-pattern.
pub(crate) struct HopTable {
    rng: StdRng,
    hopping_table: Vec<f64>,
    // filters: Vec<SosFormatFilter<f64>>,
    idx: usize,
}

impl HopTable {
    pub fn new(
        low: f64,
        high: f64,
        num_freqs: usize,
        // base_freq: f64,
        // sample_rate: usize,
        seed: u64,
    ) -> Self {
        let hopping_table: Vec<f64> = linspace(low, high, num_freqs).collect();

        Self {
            rng: StdRng::seed_from_u64(seed),
            hopping_table,
            // filters,
            idx: 0,
        }
    }
}

impl Iterator for HopTable {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        if self.idx == 0 {
            self.hopping_table.shuffle(&mut self.rng);
            self.idx += 1;
            Some(self.hopping_table[self.idx - 1])
        } else if self.idx == self.hopping_table.len() - 1 {
            self.idx = 0;
            Some(self.hopping_table[self.hopping_table.len() - 1])
        } else {
            self.idx += 1;
            Some(self.hopping_table[self.idx - 1])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iter::Iter;
    #[test]
    fn hop_table() {
        let low = 10e3;
        let high = 20e3;
        let num_freqs = 8;
        let hops: Vec<Vec<f64>> = HopTable::new(low, high, num_freqs, SEED)
            .take(8 * 5)
            .chunks(8)
            .collect();

        let mut freqs: Vec<f64> = linspace(low, high, num_freqs).collect();
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut expected: Vec<Vec<f64>> = vec![];
        for _ in 0..5 {
            freqs.shuffle(&mut rng);
            expected.push(freqs.clone());
        }

        assert_eq!(hops, expected);
    }
}
