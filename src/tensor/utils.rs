use std::cmp::PartialEq;

use std::ops::{Add, Index, Mul};

// use std::ops::{RangeBounds, RangeFrom, RangeFull};
// TODO: figure out how to make this hold references with two lifetimes, and get the iterator to return a reference
#[derive(Default)]
pub struct IndexIterator {
    index: Vec<usize>,
    dimensions: Vec<usize>,
    carry: usize,
}

impl IndexIterator {
    pub fn new(dimensions: Vec<usize>) -> IndexIterator {
        IndexIterator {
            index: vec![0; dimensions.len()],
            dimensions,
            carry: Default::default(),
        }
    }
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        for i in (0..self.index.len()).rev() {
            let v = self.index[i];
            let dim = self.dimensions[i];
            while v < dim - 1 && self.carry > 0 {
                self.index[i] = v + 1;
                self.carry -= 1;
            }
            if self.carry == 0 {
                reset_trailing_indices(&mut self.index, i);
                self.carry = 1; // for next iteration
                return Some(self.index.clone());
            }
        }
        None
    }
}

fn reset_trailing_indices(index: &mut [usize], position: usize) {
    for idx in index.iter_mut().skip(position + 1) {
        *idx = 0;
    }
}
