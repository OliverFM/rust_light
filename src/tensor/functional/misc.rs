use super::super::numeric::*;
use crate::tensor::autograd::Derivative;

use crate::tensor::utils::{global_index, ElementIterator};

use crate::tensor::{autograd, RawTensor, RcTensor, TensorLike};

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
    U2: Deref<Target = V2> + std::fmt::Debug + Clone,
    V2: TensorLike<Elem = T>,
{
    //! generalised dot product: returns to acculumulated sum of the elementwise product.
    assert!(left.same_shape(&right));
    let mut result = T::zero();
    for i in 0..left.tensor().array.len() {
        result += left.tensor().array[i] * right.tensor().array[i];
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
    //    dbg!(left.shape(), right.shape());
    assert!(2 <= left.shape().len()); // For now we can only do Batch matrix
    assert!(right.shape().len() == 2); // rhs must be a matrix
                                       //
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
                    val += *left.get(&left_index).unwrap().deref()
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

/// c[i,j] = dot(A[i, ..], B[..,j])
/// so A[i,k] * B[k,j] appears for all j -> so J_A[ø[i,j], ø[i,k], ] = B[k,j]
/// so A[i,k] * B[k,j] appears for all j -> so J_A[ø_J[i,j], ø_B[k,j], ] = A[i,k]
/// : Key thing to note here is that J
/// is over a flat input, and we are thinking about matrices, which makes this a bit weird.
/// So we need to have some map ø(i,j)->k where i,j are matrix coords, and k is the param vector
/// coords
/// need to compute jacobians[0] @ J_A, jacobians[0] @ J_B with J_A being a matrix
pub(crate) fn bmm_jvp<T: Numeric>(
    inputs: Vec<RcTensor<T>>,
    jacobians: Vec<RcTensor<T>>,
) -> Vec<RcTensor<T>> {
    assert!(
        inputs.len() == 2 && jacobians.len() == 1,
        "inputs.len()={}, jacobians.len()={}",
        inputs.len(),
        jacobians.len()
    );

    assert_eq!(
        inputs[0].shape()[1],
        inputs[1].shape()[0],
        "inputs[0].shape()={:?}, inputs[1].shape()={:?}",
        inputs[0].shape(),
        inputs[1].shape()
    );
    let bmm_output_shape = vec![inputs[0].shape()[0], inputs[1].shape()[1]];
    let left_jacobian_shape = vec![
        bmm_output_shape[0] * bmm_output_shape[1],
        inputs[0].shape()[0] * inputs[0].shape()[1], // we are only doing this for the left input
    ];
    let left_output_shape = vec![jacobians[0].shape()[0], left_jacobian_shape[1]];
    let left_length = left_output_shape[0] * left_output_shape[1]; // jacobians[0].shape()[0] * inputs[1].shape()[1];
    let mut left_array = Vec::with_capacity(left_length);
    for _ in 0..left_length {
        left_array.push(T::zero());
    }

    let right_jacobian_shape = vec![
        bmm_output_shape[0] * bmm_output_shape[1],
        inputs[1].shape()[0] * inputs[1].shape()[1], // we are only doing this for the left input
    ];
    let right_output_shape = vec![jacobians[0].shape()[0], right_jacobian_shape[1]];
    let right_length = right_output_shape[0] * right_output_shape[1]; // jacobians[0].shape()[0] * inputs[1].shape()[1];
    let mut right_array = Vec::with_capacity(right_length);
    for _ in 0..right_length {
        right_array.push(T::zero());
    }

    // println!(
    //     "inputs[0].shape()={:?}, inputs[1].shape()={:?},
    //     left_jacobian_shape={:?},
    //     right_jacobian_shape={:?},
    //     left_output_shape={:?}
    //     right_output_shape={:?}
    //     jacobians[0].shape()={:?}
    //     bmm_output_shape={:?}",
    //     inputs[0].shape(),
    //     inputs[1].shape(),
    //     &left_jacobian_shape,
    //     &right_jacobian_shape,
    //     &left_output_shape,
    //     &right_output_shape,
    //     jacobians[0].shape(),
    //     &bmm_output_shape,
    // );

    // currently: loop through all the non-zero values:
    // J_A[ø[i,j], ø[i,k]] = B[k,j]
    // Consider seeing if there is a way to get this to work such that we also build the array as
    for i in 0..inputs[0].shape()[0] {
        for k in 0..inputs[0].shape()[1] {
            for j in 0..inputs[1].shape()[1] {
                let self_jac_idx0 = global_index(&vec![i, j], &bmm_output_shape, None).unwrap();
                let left_jac_idx1 = global_index(&vec![i, k], inputs[0].shape(), None).unwrap(); // for J_A
                let right_jac_idx1 = global_index(&vec![k, j], inputs[1].shape(), None).unwrap(); // for J_A
                                                                                                  // println!(
                                                                                                  //     "i={i}, k={k}, j={j}, self_jac_idx0={:?}, self_jac_idx1={:?}",
                                                                                                  //     &self_jac_idx0, &left_jac_idx1,
                                                                                                  // );
                assert!(self_jac_idx0 < left_jacobian_shape[0]);
                assert!(self_jac_idx0 < jacobians[0].shape()[1]);
                assert!(left_jac_idx1 < left_jacobian_shape[1]);
                for input_jac_idx in 0..jacobians[0].shape()[0] {
                    let tmp_left = match global_index(
                        &vec![input_jac_idx, left_jac_idx1],
                        &left_output_shape,
                        None,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            panic!("{e}")
                        }
                    };
                    left_array[tmp_left] +=
                        jacobians[0][&vec![input_jac_idx, self_jac_idx0]] * inputs[1][&vec![k, j]];

                    let tmp_right = match global_index(
                        &vec![input_jac_idx, right_jac_idx1],
                        &right_output_shape,
                        None,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            panic!("{e}")
                        }
                    };
                    // println!("tmp_right={tmp_right:?}");
                    right_array[tmp_right] +=
                        jacobians[0][&vec![input_jac_idx, self_jac_idx0]] * inputs[0][&vec![i, k]];
                }
            }
        }
    }
    vec![
        RcTensor::new(left_array, left_output_shape),
        RcTensor::new(right_array, right_output_shape),
    ]
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
    raw_tensor.grad_fn = Some(Derivative::new(
        vec![left.clone(), right.clone()],
        dot_jvp,
        format!("dot, file: {}, line: {}", file!(), line!(),),
    ));

    RcTensor::from_raw(raw_tensor)
}

#[test]
fn test_dot() {
    let v = vec![0, 1, 2];
    let vec = RcTensor::new(v, vec![3]);
    assert_eq!(dot(&vec, &vec), RcTensor::new(vec![5], vec![1]));
}

#[test]
fn test_bmm_jvp() {
    // for (matrix_a, matrix_b)
    let matrix_a = RcTensor::from([[1.0, 2.0], [3.0, 4.0]]);
    let matrix_b = RcTensor::from([[10.0, 3.0], [42.0, -7.0]]);

    // calculated by hand and checked against pytorch
    let expected_jvp_a = RcTensor::from([[13.0, 42.0 + -7.0], [10.0 + 3.0, 42.0 + -7.0]]);
    let expected_jvp_b = RcTensor::from([[4., 4.], [6., 6.]]);
    matrix_a.bmm(&matrix_b).sum().backward();
    assert_eq!(
        *matrix_a.get_grad().borrow().as_ref().unwrap(),
        expected_jvp_a
    );
    assert_eq!(
        *matrix_b.get_grad().borrow().as_ref().unwrap(),
        expected_jvp_b
    );
}

#[test]
fn test_bmm_jvp_differing_shapes() {
    let matrix_a = RcTensor::from([[1.0, 2.0], [3.0, 4.0], [123.4, 1e-3]]);
    let matrix_b = RcTensor::from([[10.0, 1e-2, -12.0], [42.0, 3.142, -7.0]]);

    // calculated by hand and checked against pytorch
    let expected_jvp_a =
        RcTensor::from([[-1.9900, 38.1420], [-1.9900, 38.1420], [-1.9900, 38.1420]]);
    let expected_jvp_b = RcTensor::from([[127.4500, 127.4500, 127.4500], [6.0010, 6.0010, 6.0010]]);
    matrix_a.bmm(&matrix_b).sum().backward();
    assert!(
        (matrix_a.get_grad().borrow().as_ref().unwrap() - &expected_jvp_a)
            .sum()
            .elem()
            <= 1e-3
    );

    assert!(
        (matrix_b.get_grad().borrow().as_ref().unwrap() - &expected_jvp_b)
            .sum()
            .elem()
            <= 1e-3
    );
}

#[test]
fn test_add_grad() {
    for (left, right, left_grad, _right_grad) in vec![(
        RcTensor::from([1.0, 2.0, 3.0]),
        RcTensor::from([10.0, 42.0, -5.0]),
        RcTensor::from([1.0, 1.0, 1.0]),
        RcTensor::from([1.0, 1.0, 1.0]),
    )] {
        (&left + &right).sum().backward();

        let computed_grad = left.get_grad().borrow().as_ref().unwrap().deref().clone();
        let diff = &computed_grad - &left_grad;
        //        dbg!(&computed_grad, &left_grad, &diff);
        assert!(diff.sum().elem() <= 1e-3);

        let computed_grad = right.get_grad().borrow().as_ref().unwrap().deref().clone();
        let diff = &computed_grad - &left_grad;
        //        dbg!(&computed_grad, &right_grad, &diff);
        assert!(diff.sum().elem() <= 1e-3);
    }
}

#[test]
fn test_bmm_2x2() {
    let v = vec![0, 1, 2, 3];
    let matrix = RcTensor::new(v, vec![2, 2]); // [[0,1],[2,3]]
    let shape = vec![2, 1];
    let e1 = RcTensor::new(vec![0, 1], vec![2, 1]);
    let e2 = RcTensor::new(vec![1, 0], vec![2, 1]);
    let diag = RcTensor::new(vec![1, 1], vec![2, 1]);

    let r = matrix.bmm(&diag);
    assert_eq!(r.shape(), &shape);
    assert_eq!(r, RcTensor::new(vec![1, 5], shape.clone()));
    matrix.zero_grad();
    let r = matrix.bmm(&e1);
    r.sum().backward();
    assert_eq!(r, RcTensor::new(vec![1, 3], shape.clone()));
    matrix.grad();
    matrix.zero_grad();
    r.sum().backward();
    let r = matrix.bmm(&e2);
    matrix.grad();
    assert_eq!(r, RcTensor::new(vec![0, 2], shape.clone()));
}

#[test]
fn test_bmm_runs() {
    use rand::random;
    for (left_shape, right_shape) in vec![
        (vec![2, 2], vec![2, 2]),
        (vec![2, 2], vec![2, 1]),
        (vec![8, 2], vec![2, 1]),
        // (vec![2, 8, 2], vec![2, 1]), // to get this to work we need to allow tensor
        // views/reshaping
    ] {
        let length = left_shape.iter().product();
        let mut left_array: Vec<f32> = Vec::with_capacity(length);
        for _ in 0..length {
            left_array.push(random());
        }
        let left_array = left_array;
        //        dbg!(&left_array, &left_shape, &length);
        let length = right_shape.iter().product();
        let mut right_array: Vec<f32> = Vec::with_capacity(length);
        for _ in 0..length {
            right_array.push(random());
        }

        let left = RcTensor::new(left_array, left_shape);
        let right = RcTensor::new(right_array, right_shape);
        left.bmm(&right).sum().backward();
    }
}
