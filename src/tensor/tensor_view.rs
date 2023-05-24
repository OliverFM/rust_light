use super::numeric::*;
use crate::tensor::{ElementIterator, FrozenTensorView, SliceRange, Tensor, TensorLike};
use std::cmp::PartialEq;
use std::ops::Index;

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
    pub fn new(tensor: &Tensor<T>, offset: Vec<SliceRange>) -> TensorView<T> {
        assert_eq!(offset.len(), tensor.shape.len());
        let mut shape = Vec::with_capacity(offset.len());
        for (slice_range, &tensor_dim) in offset.iter().zip(tensor.shape.iter()) {
            // NOTE: assuming that all intervals are half open, for now.
            // TODO: add a better parser once I have generalised this
            assert!(slice_range.end <= tensor_dim);
            shape.push(slice_range.end - slice_range.start);
        }
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

impl<T> TensorLike<'_> for TensorView<'_, T>
where
    T: Numeric,
{
    // type Iter = std::slice::Iter<'a, T>;
    type Elem = T;
    fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    fn tensor(&self) -> &Tensor<T> {
        self.tensor
    }

    fn to_tensor(&self) -> Tensor<T> {
        todo!();
        // let mut tensor = Tensor::new_empty(self.shape);
    }

    // fn iter_elements(&self) -> Self::Iter {
    // todo!();
    // }
    fn sum(&self) -> Self::Elem {
        let iter = ElementIterator::new(self);
        let v = iter.fold(Self::Elem::zero(), |acc, x| acc + *x);
        v
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

        for idx in self.iter_indices() {
            if self.get(&idx) != other.get(&idx) {
                return false;
            }
        }
        true
    }
}

#[test]
fn test_sum_tensor_view() {
    let tensor = Tensor::from([
        [[0, 1, 2, 3], [2, 3, 4, 5], [3, 4, 5, 6]],
        [[0, 1, 2, 3], [2, 3, 4, 5], [3, 4, 5, 6]],
    ]);
    let view = tensor.slice(vec![
        SliceRange::new(0, 2),
        SliceRange::new(1, 2),
        SliceRange::new(2, 4),
    ]);

    assert_eq!(view.sum(), 2 * (4 + 5));
}
