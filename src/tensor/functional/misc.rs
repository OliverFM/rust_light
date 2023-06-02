use super::super::numeric::*;
use crate::tensor::autograd::{Derivative};

use crate::tensor::{RawTensor, RcTensor, TensorLike};
use std::ops::Deref;

pub(crate) fn todo_deriv<T: Numeric>(_inputs: Vec<RcTensor<T>>) -> RcTensor<T> {
    todo!()
}

pub(crate) fn dot_raw<T, U1, U2, V1, V2>(left: U1, right: U2) -> RawTensor<T>
where
    T: Numeric,
    U1: Deref<Target = V1> + std::fmt::Debug + Clone,
    V1: TensorLike<Elem = T>,
    U2: Deref<Target = V2> + Clone + std::fmt::Debug,
    V2: TensorLike<Elem = T>,
{
    //! generalised dot product: returns to acculumulated sum of the elementwise product.
    assert!(left.same_shape(&right));
    let mut result = T::zero();
    for i in 0..left.tensor().array.len() {
        result = result + left.tensor().array[i] * right.tensor().array[i];
    }
    RawTensor {
        array: vec![result],
        shape: vec![1],
        derivative: Some(Derivative::new(
            vec![left.to_tensor(), right.to_tensor()],
            todo_deriv,
        )),
        ..Default::default()
    }
}

pub fn dot<T, U1, U2, V1, V2>(left: U1, right: U2) -> RcTensor<T>
where
    T: Numeric,
    U1: Deref<Target = V1> + std::fmt::Debug + Clone,
    V1: TensorLike<Elem = T>,
    U2: Deref<Target = V2> + Clone + std::fmt::Debug,
    V2: TensorLike<Elem = T>,
{
    RcTensor::from_raw(dot_raw(left, right))
}
