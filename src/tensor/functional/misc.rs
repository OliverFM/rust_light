use super::super::numeric::*;
use crate::tensor::autograd::Derivative;
use crate::tensor::utils::ElementIterator;

use crate::tensor::{autograd, global_index, RawTensor, RcTensor, TensorLike};
use std::ops::Deref;

pub(crate) fn todo_backward<T: Numeric>(
    _inputs: Vec<RcTensor<T>>,
    _grads: Vec<RcTensor<T>>,
) -> Vec<RcTensor<T>> {
    todo!()
}
pub(crate) fn todo_deriv<T: Numeric>(
    _inputs: Vec<RcTensor<T>>,
    _: Vec<RcTensor<T>>,
) -> Vec<RcTensor<T>> {
    todo!()
}

#[inline]
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

#[inline]
pub(crate) fn bmm_raw<T, U1, U2, V1, V2>(left: U1, right: U2) -> RawTensor<T>
where
    T: Numeric,
    U1: Deref<Target = V1> + std::fmt::Debug + Clone,
    V1: TensorLike<Elem = T>,
    U2: Deref<Target = V2> + Clone + std::fmt::Debug,
    V2: TensorLike<Elem = T>,
{
    // assert!(2 <= self.shape().len() && self.shape().len() <= 3); // For now we can only do Batch matrix
    dbg!(left.shape().to_vec());
    assert!(2 <= left.shape().len()); // For now we can only do Batch matrix
    assert!(right.shape().len() == 2); // rhs must be a matrix
    assert!(left.shape()[left.shape().len() - 1] == right.shape()[right.shape().len() - 2]);
    let new_shape = if left.shape().len() == 2 {
        vec![1, left.shape()[0], right.shape()[1]]
    } else {
        vec![left.shape()[0], left.shape()[1], right.shape()[1]]
    };

    let mut result = RawTensor::new_empty(new_shape);

    let mut left_index = left.shape().clone();
    let left_index_len = left_index.len();
    let mut right_index = right.shape().clone();
    for batch_idx in 0..result.shape[0] {
        if left.shape().len() == 3 {
            left_index[0] = batch_idx;
        }
        for i in 0..result.shape[1] {
            left_index[left_index_len - 2] = i;
            for j in 0..result.shape[2] {
                right_index[1] = j;
                let mut val = T::zero();
                for k in 0..right.shape()[0] {
                    left_index[left_index_len - 1] = k;
                    right_index[0] = k;
                    val = val
                        + *left.get(&left_index).unwrap().deref()
                            * (*right.get(&right_index).unwrap().deref());
                }
                result.array.push(val);
            }
        }
    }
    if left.shape().len() == 2 {
        return RawTensor {
            array: result.array,
            shape: result.shape[1..].to_vec(),
            ..Default::default()
        };
    }
    result
}

pub(crate) fn bmm_jvp<T: Numeric>(inputs: Vec<RcTensor<T>>, jacobians: Vec<RcTensor<T>>) {
    assert!(
        inputs.len() == 2 && jacobians.len() == 1,
        "inputs.len()={}, jacobians.len()={}",
        inputs.len(),
        jacobians.len()
    );
    // c[i,j] = dot(A[i, ..], B[..,j])
    // so A[i,k] * B[k,j] appears for all j -> so J_A[ø[i,k], ø[i,j]] = B[k,j]
    // : Key thing to note here is that J
    // is over a flat input, and we are thinking about matrices, which makes this a bit weird.
    // So we need to have some map ø(i,j)->k where i,j are matrix coords, and k is the param vector
    // coords
    // need to compute jacobians[0] @ J_A, jacobians[0] @ J_B with J_A being a matrix

    let shape = vec![jacobians[0].shape()[0], inputs[1].shape()[1]];
    let length = jacobians[0].shape()[0] * inputs[1].shape()[1];
    let mut array = Vec::<usize>::with_capacity(length);

    // TODO: instead loop through all the non-zero values:
    // J_A[ø[i,k], ø[i,j]] = B[k,j]
    // Consider seeing if there is a way to get this to work such that we also build the array as
    // we go
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            // sum_h jacobians[i, h] * J_A[h, j]
            for k in 0..inputs[0].shape()[1] {
                // loop through inner values of
            }
        }
    }
    todo!()
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
    assert!(left.same_shape(&right));
    let array = ElementIterator::new(left)
        .zip(ElementIterator::new(right))
        .map(|(x, y)| x * y)
        .collect();

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

fn dot_jvp<T: Numeric>(inputs: Vec<RcTensor<T>>, jacobians: Vec<RcTensor<T>>) -> Vec<RcTensor<T>> {
    assert!(
        inputs.len() == 2 && jacobians.len() == 1,
        "inputs.len()={}, jacobians.len()={}",
        inputs.len(),
        jacobians.len()
    );
    let (left, right) = (inputs[0].clone(), inputs[1].clone());
    let jacobian = jacobians[0].clone();
    let left_jvp = autograd::jvp_from_full_jacobians(jacobian.clone(), right);
    let right_jvp = autograd::jvp_from_full_jacobians(jacobian, left);
    vec![left_jvp, right_jvp]
}

#[test]
fn test_dot_autograd() {
    let left = RcTensor::from([1.0, 2.0, 3.0]);
    let right = RcTensor::from([4.0, 5.0, 6.0]);
    dot(&left, &right).backward();
    assert_eq!(&right, left.get_grad().borrow().as_ref().unwrap());
    assert_eq!(&left, right.get_grad().borrow().as_ref().unwrap());
}

// TODO: generalise this to views
pub fn dot<T>(left: &RcTensor<T>, right: &RcTensor<T>) -> RcTensor<T>
where
    T: Numeric,
{
    let mut raw_tensor = dot_raw(left, right);
    raw_tensor.derivative = Some(Derivative::new(vec![left.clone(), right.clone()], dot_jvp));

    RcTensor::from_raw(raw_tensor)
}

#[test]
fn test_dot() {
    let v = vec![0, 1, 2];
    let vec = RcTensor::new(v, vec![3]);
    assert_eq!(dot(&vec, &vec), RcTensor::new(vec![5], vec![1]));
}
