use super::numeric::*;
use crate::tensor::functional;
use crate::tensor::utils::ElementIterator;
use crate::tensor::{IndexType, RcTensor, Scalar, SliceRange, TensorLike};

use std::cmp::PartialEq;
use std::ops::{Deref, Index};
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct View {
    shape: Vec<usize>,
    offset: Vec<SliceRange>,
}

#[derive(Debug, Clone)]
pub struct TensorView<T>
where
    T: Numeric,
{
    // TODO: convert this to look at slices of tensors. e.g. tensor[..1]
    // tensor: &'a Tensor<T>,
    tensor: RcTensor<T>,
    view: Rc<View>,
}

impl<T> Index<&Vec<usize>> for TensorView<T>
where
    T: Numeric,
{
    type Output = T;

    fn index(&self, index: IndexType) -> &Self::Output {
        self.tensor
            .get_with_offset(index, &self.view.offset)
            .unwrap()
    }
}

// a borrowed Tensor, but with a new shape.
impl<T> TensorView<T>
where
    T: Numeric,
{
    pub fn new(tensor: RcTensor<T>, offset: Vec<SliceRange>) -> TensorView<T> {
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
            view: View {
                offset,
                shape, // TODO: fix this
            }
            .into(),
        }
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        assert_eq!(
            self.view.shape.iter().product::<usize>(),
            shape.iter().product::<usize>()
        );
        TensorView {
            tensor: self.tensor.clone(),
            view: View {
                shape,
                offset: self.view.offset.clone(),
            }
            .into(),
        }
    }
}

impl<T> crate::tensor::tensor_like::TensorLikePrivate for TensorView<T> where T: Numeric {}

impl<T> TensorLike for TensorView<T>
where
    T: Numeric,
{
    type Elem = T;
    type ShapeReturn<'a> = &'a Vec<usize> where Self: 'a ;
    type TensorRef<'a>= RcTensor<T> where Self: 'a; // &'tensor Tensor<Self::Elem> where Self : 'tensor;
    type ResultTensorType<'a>= RcTensor<T> where Self: 'a; // &'tensor Tensor<Self::Elem> where Self : 'tensor;
    type SumType = Scalar<Self::Elem>;
    type GradType = RcTensor<T>;

    fn dot<U, V>(&self, _other: U) -> RcTensor<Self::Elem>
    where
        U: Deref<Target = V> + std::fmt::Debug + Clone,
        V: TensorLike<Elem = Self::Elem>,
    {
        todo!()
    }

    fn shape(&self) -> Self::ShapeReturn<'_> {
        &self.view.shape
    }

    fn tensor(&self) -> Self::TensorRef<'_> {
        // Ref::clone(&self.tensor)
        self.tensor.clone()
    }

    fn to_tensor(&self) -> RcTensor<T> {
        let mut array = Vec::with_capacity(self.view.shape.iter().product());
        for elem in ElementIterator::new(self) {
            array.push(elem);
        }
        RcTensor::new(array, self.view.shape.clone())
    }

    fn sum(&self) -> Scalar<Self::Elem> {
        let iter = ElementIterator::new(self);

        let v = iter.fold(Self::Elem::zero(), |acc, x| acc + x);
        Scalar::from(v)
    }

    fn get(&self, index: IndexType) -> Result<&T, String> {
        let idx = self
            .tensor
            .global_index(index, Some(&self.view.offset))
            .unwrap();
        Ok(&self.tensor.array[idx])
    }

    fn count(&self) -> usize {
        self.view.shape.iter().product()
    }

    fn slice(&self, offset: Vec<SliceRange>) -> TensorView<T> {
        TensorView::new(self.tensor(), offset)
    }

    fn bmm<U>(&self, right: &U) -> Self::ResultTensorType<'_>
    where
        U: TensorLike<Elem = Self::Elem>,
    {
        self.bmm_rc(right)
    }

    #[inline]
    fn bmm_rc<U>(&self, right: &U) -> RcTensor<Self::Elem>
    where
        U: TensorLike<Elem = Self::Elem>,
    {
        RcTensor::from_raw(functional::bmm_raw(self, right))
    }
}

impl<T, V> PartialEq<V> for TensorView<T>
where
    T: Numeric,
    V: TensorLike<Elem = T>,
{
    fn eq(&self, other: &V) -> bool {
        if *other.shape() != self.view.shape {
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
    use crate::tensor::RawTensor;
    let tensor = RawTensor::from([
        [[0, 1, 2, 3], [2, 3, 4, 5], [3, 4, 5, 6]],
        [[0, 1, 2, 3], [2, 3, 4, 5], [3, 4, 5, 6]],
    ]);
    let view = tensor.slice(vec![
        SliceRange::new(0, 2),
        SliceRange::new(1, 2),
        SliceRange::new(2, 4),
    ]);

    assert_eq!(view.sum().elem(), 2 * (4 + 5));
}
