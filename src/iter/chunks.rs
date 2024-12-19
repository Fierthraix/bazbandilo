pub struct Chunks<T: Copy, I: Iterator<Item = T>> {
    source: I,
    num_chunks: usize,
}

impl<T: Copy, I: Iterator<Item = T>> Chunks<T, I> {
    pub fn new(source: I, num_chunks: usize) -> Chunks<T, I> {
        Self { source, num_chunks }
    }
}

impl<T: Copy, I: Iterator<Item = T>> Iterator for Chunks<T, I> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Vec<T>> {
        let mut buf = Vec::with_capacity(self.num_chunks);
        // Get up to num_chunks items
        while buf.len() < self.num_chunks {
            match self.source.next() {
                Some(t) => {
                    buf.push(t);
                }
                None => {
                    break;
                }
            }
        }
        if buf.is_empty() {
            None
        } else {
            Some(buf)
        }
    }
}

pub struct WholeChunks<T: Copy + Default, I: Iterator<Item = T>> {
    source: I,
    num_chunks: usize,
    done: bool,
}

impl<T: Copy + Default, I: Iterator<Item = T>> WholeChunks<T, I> {
    pub fn new(source: I, num_chunks: usize) -> WholeChunks<T, I> {
        Self {
            source,
            num_chunks,
            done: false,
        }
    }
}

impl<T: Copy + Default, I: Iterator<Item = T>> Iterator for WholeChunks<T, I> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Vec<T>> {
        let mut buf = Vec::with_capacity(self.num_chunks);
        if !self.done {
            // Get up to num_chunks items
            for _ in 0..self.num_chunks {
                match self.source.next() {
                    Some(t) => {
                        buf.push(t);
                    }
                    None => {
                        self.done = true;
                        buf.push(T::default());
                    }
                }
            }
        }
        if buf.is_empty() {
            None
        } else {
            Some(buf)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::iter::Iter;

    #[test]
    fn chunks() {
        let num = 1000;
        let num_chunks = 5;
        let ones: Vec<usize> = std::iter::repeat_n(1, num)
            .chunks(num_chunks)
            .map(|chunk| chunk.into_iter().product::<usize>())
            .collect();

        let expected = vec![1; num / num_chunks];

        assert_eq!(ones, expected);
    }
}
