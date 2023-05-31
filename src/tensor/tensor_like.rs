use super::numeric::*;
use super::utils::IndexIterator;
use crate::tensor::{RawTensor, RcTensor, SliceRange, TensorView};
use std::ops::Deref;

pub trait TensorLikePublic: TensorLike {}

impl<T, U> TensorLikePublic for U
where
    T: Numeric,
    U: TensorLike<Elem = T>,
{
}

pub trait TensorLike {
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

    fn get(&self, index: &Vec<usize>) -> Result<&Self::Elem, String>;

    fn get_first_elem(&self) -> &Self::Elem {
        let shape = vec![0; self.shape().deref().len()];
        self.get(&shape).unwrap()
    }

    fn shape(&self) -> Self::ShapeReturn<'_>;

    fn sum(&self) -> Self::Elem;

    /// Return a reference to the underlying tensor
    fn tensor(&self) -> Self::TensorRef<'_>;

    /// Convert this self into a new Tensor -- is self is already a Tensor this is a clone.
    /// for a `TensorView`, for example, the new Tensor is the same shape as the view.
    fn to_tensor(&self) -> RcTensor<Self::Elem>;

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
    fn dot<U>(&self, other: &U) -> RawTensor<Self::Elem>
    where
        U: TensorLike<Elem = Self::Elem>,
    {
        //! generalised dot product: returns to acculumulated sum of the elementwise product.
        assert!(self.same_shape(other));
        let mut result = Self::Elem::zero();
        for i in 0..self.tensor().array.len() {
            result = result + self.tensor().array[i] * other.tensor().array[i];
        }
        RawTensor {
            array: vec![result],
            shape: vec![1],
            ..Default::default()
        }
    }

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
    #[inline]
    fn bmm_rc<U>(&self, right: &U) -> RcTensor<Self::Elem>
    where
        U: TensorLike<Elem = Self::Elem>,
    {
        RcTensor::from_raw(self.bmm_raw(right))
    }

    #[inline]
    fn bmm_raw<U>(&self, right: &U) -> RawTensor<Self::Elem>
    where
        U: TensorLike<Elem = Self::Elem>,
    {
        assert!(2 <= self.shape().len() && self.shape().len() <= 3); // For now we can only do Batch matrix
        assert!(right.shape().len() == 2); // rhs must be a matrix
        assert!(self.shape()[self.shape().len() - 1] == right.shape()[right.shape().len() - 2]);
        let new_shape = if self.shape().len() == 2 {
            vec![1, self.shape()[0], right.shape()[1]]
        } else {
            vec![self.shape()[0], self.shape()[1], right.shape()[1]]
        };

        let mut result = RawTensor::new_empty(new_shape);

        let mut self_index = self.shape().clone();
        let self_index_len = self_index.len();
        let mut right_index = right.shape().clone();
        for batch_idx in 0..result.shape[0] {
            if self.shape().len() == 3 {
                self_index[0] = batch_idx;
            }
            for i in 0..result.shape[1] {
                self_index[self_index_len - 2] = i;
                for j in 0..result.shape[2] {
                    right_index[1] = j;
                    let mut val = Self::Elem::zero();
                    for k in 0..right.shape()[0] {
                        self_index[self_index_len - 1] = k;
                        right_index[0] = k;
                        val = val
                            + *self.get(&self_index).unwrap().deref()
                                * (*right.get(&right_index).unwrap().deref());
                    }
                    result.array.push(val);
                }
            }
        }
        if self.shape().len() == 2 {
            return RawTensor {
                array: result.array,
                shape: result.shape[1..].to_vec(),
                ..Default::default()
            };
        }
        result
    }

    fn same_shape<U>(&self, other: &U) -> bool
    where
        U: TensorLike<Elem = Self::Elem>,
    {
        *self.shape() == *other.shape()
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
