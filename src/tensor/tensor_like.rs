use super::functional;
use super::numeric::*;
use super::utils::IndexIterator;
use super::{RawTensor, RcTensor, SliceRange, TensorView};
use std::ops::Deref;

pub trait TensorLikePublic: TensorLike {}

pub(in crate::tensor) mod private {
    pub trait TensorLikePrivate {}
}
pub use crate::tensor::private::TensorLikePrivate;

pub trait TensorLike: TensorLikePrivate + std::fmt::Debug {
    type Elem: Numeric;
    type ShapeReturn<'a>: Deref<Target = Vec<usize>>
    where
        Self: 'a;
    type TensorRef<'a>: Deref<Target = RawTensor<Self::Elem>>
    where
        Self: 'a;
    type ResultTensorType<'a>: TensorLike
    where
        Self: 'a;
    type SumType: TensorLike<Elem = Self::Elem>;
    type GradType: TensorLike;

    fn update_grad(&self, _grad: Self::GradType) {
        todo!();
    }
    fn zero_grad(&self) {
        todo!();
    }

    fn get(&self, index: &Vec<usize>) -> Result<&Self::Elem, String>;

    #[inline]
    fn get_first_elem(&self) -> &Self::Elem {
        let shape = vec![0; self.shape().deref().len()];
        self.get(&shape).unwrap()
    }

    fn elem(&self) -> Self::Elem {
        assert!(self.shape().is_empty());
        *self.get_first_elem()
    }

    fn shape(&self) -> Self::ShapeReturn<'_>;

    fn sum(&self) -> Self::SumType;

    /// Return a reference to the underlying tensor
    fn tensor(&self) -> Self::TensorRef<'_>;

    /// Convert this self into a new Tensor -- is self is already a Tensor this is a clone.
    /// for a `TensorView`, for example, the new Tensor is the same shape as the view.
    fn to_tensor(&self) -> RcTensor<Self::Elem>;

    fn deep_clone(&self) -> RcTensor<Self::Elem> {
        let mut raw_tensor = self.to_tensor().0.deref().clone();
        raw_tensor.derivative = None;
        raw_tensor.zero_grad();
        RcTensor::from_raw(raw_tensor)
    }

    fn slice(&self, offset: Vec<SliceRange>) -> TensorView<Self::Elem>;

    fn left_scalar_multiplication(&self, &scalar: &Self::Elem) -> RawTensor<Self::Elem> {
        let mut result = RawTensor::new_empty((*self.shape()).clone());
        for &elem in self.tensor().array.iter() {
            result.array.push(scalar * elem);
        }
        result
    }

    fn right_scalar_multiplication(&self, &scalar: &Self::Elem) -> RawTensor<Self::Elem> {
        let mut result = RawTensor::new_empty((*self.shape()).clone());
        for &elem in self.tensor().array.iter() {
            result.array.push(elem * scalar);
        }
        result
    }

    // TODO: split this by return type in the same way as bmm
    fn dot<U, V>(&self, other: U) -> RcTensor<Self::Elem>
    where
        U: Deref<Target = V> + std::fmt::Debug + Clone,
        V: TensorLike<Elem = Self::Elem>;

    /// A Naive batch matrix multiply operation
    ///
    /// ```
    /// # use rust_light::tensor::*;
    /// let v = vec![0, 1, 2, 3];
    /// let matrix = RcTensor::new(v, vec![2, 2]);
    /// let shape = vec![2, 1];
    /// let e1 = RcTensor::new(vec![0, 1], vec![2, 1]);
    /// let e2 = RcTensor::new(vec![1, 0], vec![2, 1]);
    /// let diag = RcTensor::new(vec![1, 1], vec![2, 1]);
    /// let r = matrix.bmm(&diag);
    ///
    /// assert_eq!(r.shape(), &shape);
    /// assert_eq!(r, RcTensor::new(vec![1, 5], shape.clone()));
    /// assert_eq!(matrix.bmm(&e1), RcTensor::new(vec![1, 3], shape.clone()));
    /// ```
    fn bmm<U>(&self, right: &U) -> Self::ResultTensorType<'_>
    where
        U: TensorLike<Elem = Self::Elem>;

    // TODO: consider making this private with type magic
    // https://jack.wrenn.fyi/blog/private-trait-methods/
    fn bmm_rc<U>(&self, right: &U) -> RcTensor<Self::Elem>
    where
        U: TensorLike<Elem = Self::Elem>;

    fn same_shape<U, V>(&self, other: &U) -> bool
    where
        U: Deref<Target = V> + std::fmt::Debug + Clone,
        V: TensorLike<Elem = Self::Elem>,
    {
        *self.shape() == **other.shape()
    }

    fn broadcastable<V: Deref<Target = Vec<usize>>>(&self, new_shape: V) -> bool {
        // TODO: test this!
        for (&d1, &d2) in self
            .shape()
            .iter()
            .rev()
            .zip(new_shape.to_vec().iter().rev())
        {
            if d1 != d2 {
                if d1 == 1 || d2 == 1 {
                    continue;
                }
                return false;
            }
        }
        true
    }

    fn iter_indices(&self) -> IndexIterator {
        IndexIterator::new(self.shape().clone())
    }
}
