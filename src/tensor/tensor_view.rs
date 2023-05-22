use crate::tensor::utils::{IndexIterator, Numeric};
use crate::tensor::FrozenTensorView;
use crate::tensor::SliceRange;
use crate::tensor::Tensor;
use crate::tensor::TensorLike;
// use crate::tensor::TensorView;
use super::tensor_like::*;
use super::utils::*;
use itertools::{EitherOrBoth::*, Itertools};
use num::{One, Zero};
use std::cmp::{max, PartialEq};
use std::convert::From;
use std::ops::{Add, Index, Mul};

#[derive(Debug, Clone)]
pub struct TensorView<'a, T>
where
    T: Numeric,
{
    // TODO: convert this to look at slices of tensors. e.g. tensor[..1]
    tensor: &'a Tensor<T>,
    shape: Vec<usize>,
    offset: Vec<SliceRange>,
}
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
    pub fn new(tensor: &Tensor<T>, offset: Vec<SliceRange>, shape: Vec<usize>) -> TensorView<T> {
        TensorView {
            tensor,
            offset,
            shape, // TODO: fix this
        }
    }
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
        if other.shape() != &self.shape {
            return false;
        }

        for idx in self.iter_elements() {
            if self.get(&idx) != other.get(&idx) {
                return false;
            }
        }
        true
    }
}
