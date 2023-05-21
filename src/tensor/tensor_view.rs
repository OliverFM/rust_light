use crate::tensor::utils::{IndexIterator, Numeric};
use crate::tensor::FrozenTensorView;
use crate::tensor::SliceRange;
use crate::tensor::Tensor;
use crate::tensor::TensorLike;
use crate::tensor::TensorView;
use itertools::{EitherOrBoth::*, Itertools};
use num::{One, Zero};
use std::cmp::{max, PartialEq};
use std::convert::From;
use std::ops::{Add, Index, Mul};

impl<'a, T> Index<&Vec<usize>> for TensorView<'a, T>
where
    T: Numeric,
{
    type Output = T;

    fn index(&self, index: &Vec<usize>) -> &Self::Output {
        self.tensor.get_with_offset(index, &self.offset).unwrap()
    }
}

// a borrowed Tensor, but with a new shape.
impl<'a, T> TensorView<'a, T>
where
    T: Numeric,
{
    pub fn to_tensor(&self) -> Tensor<T> {
        (*self.tensor).clone()
    }

    pub fn freeze(&self) -> FrozenTensorView<'_, T> {
        FrozenTensorView {
            tensor: self.tensor,
            shape: &self.shape,
        }
    }
}

impl<'a, T> TensorLike<'a> for TensorView<'a, T>
where
    T: Numeric,
{
    type Elem = T;
    fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    fn tensor(&self) -> &Tensor<T> {
        self.tensor
    }

    fn to_tensor(&self) -> Tensor<T> {
        // self.tensor.clone()
        todo!();
    }

    fn get(&self, index: &Vec<usize>) -> Result<&T, String> {
        let idx = self
            .tensor
            .get_global_index(index, Some(&self.offset))
            .unwrap();
        Ok(&self.tensor.array[idx])
    }
}

impl<'a, T, U> PartialEq<U> for TensorView<'a, T>
where
    T: Numeric,
    U: TensorLike<'a, Elem = T>,
{
    fn eq(&self, other: &U) -> bool {
        // println!(
        // "self.shape={:?}, other.shape={:?}",
        // self.shape.clone(),
        // other.shape().clone()
        // );
        if other.shape() != &self.shape {
            return false;
        }
        let index_iter = IndexIterator {
            index: vec![0; self.shape.len()],
            dimensions: self.shape.clone(),
            carry: Default::default(),
        };

        for idx in index_iter {
            // println!(
            // "self[{:?}]={:?}, other[{:?}]={:?}",
            // idx.clone(),
            // self.get(&idx),
            // idx.clone(),
            // other.get(&idx)
            // );
            if self.get(&idx) != other.get(&idx) {
                return false;
            }
        }
        true
    }
}
