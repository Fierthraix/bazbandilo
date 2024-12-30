use crate::Bit;

pub fn supported_barker_lengths() -> [usize; 7] {
    [2, 3, 4, 5, 7, 11, 13]
}

pub fn get_barker_code(len: usize) -> Option<Vec<Bit>> {
    match len {
        2 => Some(vec![true, false]), // [true, true]),
        3 => Some(vec![true, true, false]),
        4 => Some(vec![
            true, true, false, true, true, true, true, false, false,
        ]),
        5 => Some(vec![true, true, true, false, true, false]),
        7 => Some(vec![true, true, true, false, false, true, false, false]),
        11 => Some(vec![
            true, true, true, false, false, false, true, false, false, true, false,
        ]),
        13 => Some(vec![
            true, true, true, true, true, false, false, true, true, false, true, false, true,
        ]),
        _ => None,
    }
}
