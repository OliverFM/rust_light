use super::numeric::*;
use crate::tensor::{ElementIterator, RawTensor, SliceRange, Tensor, TensorLike};
use std::cell::Ref;
use std::cmp::PartialEq;
use std::ops::{Deref, Index};

#[derive(Debug, Clone)]
pub struct TensorView<T>
where
    T: Numeric,
{
    // TODO: convert this to look at slices of tensors. e.g. tensor[..1]
    // tensor: &'a Tensor<T>,
    tensor: Tensor<T>,
    shape: Vec<usize>,
    offset: Vec<SliceRange>,
}
impl<T> Index<&Vec<usize>> for TensorView<T>
where
    T: Numeric,
{
    type Output = T;

    fn index(&self, index: &Vec<usize>) -> &Self::Output {
        self.tensor
            .0
            .borrow()
            .get_with_offset(index, &self.offset)
            .unwrap()
    }
}

// a borrowed Tensor, but with a new shape.
impl<T> TensorView<T>
where
    T: Numeric,
{
    pub fn new(tensor: Tensor<T>, offset: Vec<SliceRange>) -> TensorView<T> {
        assert_eq!(offset.len(), tensor.shape().len());
        let mut shape = Vec::with_capacity(offset.len());
        for (slice_range, &tensor_dim) in offset.iter().zip(tensor.shape().iter()) {
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
    pub fn to_tensor(&self) -> RawTensor<T> {
        self.tensor.0.borrow().clone()
    }
}

impl<T> TensorLike for TensorView<T>
where
    T: Numeric,
{
    type Elem = T;
    type ShapeReturn<'a> = &'a Vec<usize> where Self: 'a ;
    type TensorRef<'a>= Tensor<T> where Self: 'a; // &'tensor Tensor<Self::Elem> where Self : 'tensor;
    fn shape(&self) -> Self::ShapeReturn<'_> {
        &self.shape
    }

    fn tensor(&self) -> Self::TensorRef<'_> {
        // Ref::clone(&self.tensor)
        self.tensor.clone()
    }

    fn to_tensor(&self) -> RawTensor<T> {
        todo!();
        // let mut tensor = Tensor::new_empty(self.shape);
    }

    fn sum(&self) -> Self::Elem {
        let iter = ElementIterator::new(self);

        iter.fold(Self::Elem::zero(), |acc, x| acc + *x)
    }

    fn get(&self, index: &Vec<usize>) -> Result<&T, String> {
        let idx = self
            .tensor
            .0
            .borrow()
            .get_global_index(index, Some(&self.offset))
            .unwrap();
        Ok(&self.tensor.0.borrow().array[idx])
    }
}

impl<'a, T, V> PartialEq<V> for TensorView<T>
where
    T: Numeric,
    V: TensorLike<Elem = T>,
{
    fn eq(&self, other: &V) -> bool {
        if *other.shape() != self.shape {
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
    let tensor = RawTensor::from([
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
