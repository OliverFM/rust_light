use super::super::numeric::*;
use crate::tensor::autograd::Derivative;
use crate::tensor::utils::ElementIterator;

use crate::tensor::{RawTensor, RcTensor, TensorLike};
use std::ops::Deref;

pub(crate) fn todo_backward<T: Numeric>(
    _inputs: Vec<RcTensor<T>>,
    _grads: Vec<RcTensor<T>>,
) -> Vec<RcTensor<T>> {
    todo!()
}
pub(crate) fn todo_deriv<T: Numeric>(_inputs: Vec<RcTensor<T>>) -> Vec<RcTensor<T>> {
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
        ..Default::default()
    }
}

pub fn element_wise_multiplication<T, U1, V1, U2, V2>(left: U1, right: U2) -> RawTensor<T>
where
    T: Numeric,
    U1: Deref<Target = V1> + std::fmt::Debug + Clone,
    V1: TensorLike<Elem = T>,
    U2: Deref<Target = V2> + Clone + std::fmt::Debug,
    V2: TensorLike<Elem = T>,
{
    let left_shape_vec = left.shape().to_vec();
    assert!(left_shape_vec == right.shape().to_vec());
    let length = left.shape().iter().product();
    let mut array = Vec::with_capacity(length);
    for (x, y) in ElementIterator::new(left).zip(ElementIterator::new(right)) {
        array.push(x * y);
    }

    RawTensor::new(array, left_shape_vec)
}

pub fn dot_no_derivative<T, U1, U2, V1, V2>(left: U1, right: U2) -> RcTensor<T>
where
    T: Numeric,
    U1: Deref<Target = V1> + std::fmt::Debug + Clone,
    V1: TensorLike<Elem = T>,
    U2: Deref<Target = V2> + Clone + std::fmt::Debug,
    V2: TensorLike<Elem = T>,
{
    RcTensor::from_raw(dot_raw(left, right))
}

// TODO: generalise this to views
pub fn dot<T>(left: &RcTensor<T>, right: &RcTensor<T>) -> RcTensor<T>
where
    T: Numeric,
{
    let mut raw_tensor = dot_raw(left, right);
    raw_tensor.derivative = Some(Derivative::new(
        vec![left.clone(), right.clone()],
        todo_deriv,
    ));

    RcTensor::from_raw(raw_tensor)
}

#[test]
fn test_dot() {
    let v = vec![0, 1, 2];
    let vec = RcTensor::new(v, vec![3]);
    assert_eq!(dot(&vec, &vec), RcTensor::new(vec![5], vec![1]));
}
