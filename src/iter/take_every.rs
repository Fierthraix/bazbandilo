pub struct Take<T, I: Iterator<Item = T>> {
    source: I,
    frequency: usize,
    curr: usize,
}

impl<T, I: Iterator<Item = T>> Take<T, I> {
    pub fn new(source: I, frequency: usize) -> Take<T, I> {
        Self {
            source,
            frequency,
            curr: 0,
        }
    }
}

impl<T, I: Iterator<Item = T>> Iterator for Take<T, I> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {
            if self.curr % self.frequency == 0 {
                self.curr += 1;
                return self.source.next();
            } else {
                self.source.next();
                self.curr += 1;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::iter::Iter;

    #[test]
    fn test_take_every() {
        let list: Vec<usize> = (0..10).take_every(2).collect();

        assert_eq!(list, [0, 2, 4, 6, 8]);
    }
}
