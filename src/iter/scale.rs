pub struct Scale<T: std::ops::Mul<f64, Output = T>, I: Iterator<Item = T>> {
    source: I,
    scalar: f64,
}

impl<T: std::ops::Mul<f64, Output = T>, I: Iterator<Item = T>> Scale<T, I> {
    pub fn new(source: I, scalar: f64) -> Scale<T, I> {
        Self { source, scalar }
    }
}

impl<T: std::ops::Mul<f64, Output = T>, I: Iterator<Item = T>> Iterator for Scale<T, I> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        Some(self.source.next()? * self.scalar)
    }
}

#[cfg(test)]
mod tests {
    use crate::iter::Iter;

    #[test]
    fn scale() {
        let num = 1000;
        let scaled: Vec<f64> = (0..num).map(|i| i as f64).scale(2f64).collect();

        let expected: Vec<f64> = (0..num).map(|i| (2 * i) as f64).collect();

        assert_eq!(scaled, expected);
    }
}
