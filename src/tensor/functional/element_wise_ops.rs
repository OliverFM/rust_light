use crate::tensor::numeric::*;

use crate::tensor::{RawTensor, RcTensor, TensorLike, TensorList};
use num::traits::real::Real;

use crate::tensor::autograd::*;

use crate::tensor::autograd::Derivative;
use crate::tensor::utils::IndexIterator;
use crate::tensor::utils::{global_index, increment_index, ElementIterator};
use std::ops::Deref;

use std::cmp::max;

use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;

pub fn tanh<T, U, V>(tensor_like: U) -> RcTensor<T>
where
    T: Numeric + Real,
    U: Deref<Target = V> + std::fmt::Debug + Clone,
    V: TensorLike<Elem = T>,
{
    let tensor = tensor_like.to_tensor();
    let length = tensor.shape().iter().fold(1, |acc, x| acc * *x);
    let mut array = Vec::with_capacity(length);
    for elem in ElementIterator::new(&tensor) {
        array.push(elem.tanh());
    }
    let mut raw_tensor = RawTensor::new(array, tensor.shape().clone());
    raw_tensor.grad_fn = Some(Derivative::new(
        vec![tensor.clone()],
        tanh_derivative_outer,
        format!("tanh, file: {}, line: {}", file!(), line!(),),
    ));
    RcTensor::from_raw(raw_tensor)
}

fn tanh_derivative_outer<T: Numeric + Real>(
    tensors: Vec<RcTensor<T>>,
    grads: Vec<RcTensor<T>>,
) -> Vec<RcTensor<T>> {
    assert!(tensors.len() == 1);
    assert!(grads.len() == 1);
    let jacobian = tanh_derivative(&tensors[0], true);
    let new_grad = jvp_from_full_jacobians(grads[0].clone(), jacobian);
    vec![new_grad]
}

fn tanh_derivative<T: Numeric + Real>(
    tensor: &RcTensor<T>,
    full_jacobian_output: bool,
) -> RcTensor<T> {
    let length = tensor.shape().iter().product::<usize>();
    if !full_jacobian_output {
        let mut array = Vec::with_capacity(length);
        for elem in ElementIterator::new(tensor) {
            let v = T::one() - elem.tanh().powi(2);
            array.push(v);
        }
        return RcTensor::new(array, tensor.shape().clone());
    }
    let mut array = Vec::with_capacity(length * length);
    for i in 0..length {
        for j in 0..length {
            let v = if i == j {
                T::one() - tensor.0.array[i].tanh().powi(2)
            } else {
                T::zero()
            };
            array.push(v);
        }
    }
    let result = RcTensor::new(array, vec![length, length]);
    for i in 0..length {
        for j in 0..length {
            if i == j {
                assert_eq!(
                    result[&vec![i, j]],
                    T::one() - tensor.0.array[i].tanh().powi(2)
                );
            } else {
                assert_eq!(result[&vec![i, j]], T::zero());
            };
        }
    }
    result
}

pub(crate) fn add<T, U1, U2, V1, V2>(left: U1, right: U2) -> RcTensor<T>
where
    T: Numeric,
    U1: Deref<Target = V1> + std::fmt::Debug + Clone,
    V1: TensorLike<Elem = T>,
    U2: Deref<Target = V2> + Clone + std::fmt::Debug,
    V2: TensorLike<Elem = T>,
{
    let left_res = left.to_tensor();
    let right_res = right.to_tensor();
    let mut raw_tensor = add_raw(left, right);
    raw_tensor.grad_fn = Some(Derivative::new(
        vec![left_res, right_res],
        add_jvp,
        format!("add, file: {}, line: {}", file!(), line!(),),
    ));
    RcTensor::from_raw(raw_tensor)
}

fn add_jvp<T: Numeric>(tensors: TensorList<T>, grads: TensorList<T>) -> TensorList<T> {
    assert!(tensors.len() == 2);
    assert!(grads.len() == 1);
    assert!(grads[0].shape().len() == 2);
    let (left, right) = (tensors[0].clone(), tensors[1].clone());
    let grad = grads[0].clone();
    let broadcast_shape = max_shape(left.shape(), right.shape());
    let diag_length = broadcast_shape.iter().product::<usize>();
    assert_eq!(
        grad.shape()[1],
        diag_length,
        "grad is of shape {:?} sum has {:?} parameters,
jacobians are not matrix multipliable",
        grad.shape(),
        diag_length
    );
    // let length_grad = grads[0].shape().iter().product::<usize>();

    let mut left_array = vec![T::zero(); left.count() * grad.shape()[0]];
    let mut right_array = vec![T::zero(); right.count() * grad.shape()[0]];
    let left_jvp_shape = vec![grad.shape()[0], left.count()];
    let right_jvp_shape = vec![grad.shape()[0], right.count()];

    let mut idx = vec![0; broadcast_shape.len()];

    // TODO: multiply by grad!
    dbg!(&broadcast_shape);
    loop {
        // through all elements of the broadcast inputs
        // noting that this is going along the diagonal of J_left, and J_right
        // in their expanded broadcasted form
        // We then compute grad @ J_left and grad @ J_right
        let jac_idx1 = global_index(&idx, &broadcast_shape, None).unwrap();
        for jac_idx0 in 0..grad.shape()[0] {
            let left_input_idx = global_index(&idx, left.shape(), None).unwrap();

            let left_jac_idx =
                global_index(&vec![jac_idx0, left_input_idx], &left_jvp_shape, None).unwrap();
            let right_input_idx = global_index(&idx, right.shape(), None).unwrap();
            let right_jac_idx =
                global_index(&vec![jac_idx0, right_input_idx], &right_jvp_shape, None).unwrap();

            let grad_val = grad[&vec![jac_idx0, jac_idx1]];
            dbg!(&left_jac_idx, &right_jac_idx, &idx, &grad_val);

            left_array[left_jac_idx] += grad_val;
            right_array[right_jac_idx] += grad_val;
        }
        if !increment_index(&mut idx, &broadcast_shape[..]) {
            break;
        }
    }
    let raw_left_grad = RawTensor::new(left_array, left_jvp_shape);
    let raw_right_grad = RawTensor::new(right_array, right_jvp_shape);
    dbg!(&raw_left_grad, &raw_right_grad);
    vec![
        RcTensor::from_raw(raw_left_grad),
        RcTensor::from_raw(raw_right_grad),
    ]
}

fn max_shape(left_shape: &[usize], right_shape: &[usize]) -> Vec<usize> {
    let mut max_shape = Vec::with_capacity(max(left_shape.len(), right_shape.len()));

    // TODO: consider getting rid of itertools
    for pair in left_shape
        .iter()
        .rev()
        .zip_longest(right_shape.iter().rev())
        .rev()
    {
        let dim = match pair {
            Both(&l, &r) => max(l, r),
            Left(&l) => l,
            Right(&r) => r,
        };
        max_shape.push(dim);
    }
    max_shape
}
pub(crate) fn add_raw<T, U1, U2, V1, V2>(left: U1, right: U2) -> RawTensor<T>
where
    T: Numeric,
    U1: Deref<Target = V1> + std::fmt::Debug + Clone,
    V1: TensorLike<Elem = T>,
    U2: Deref<Target = V2> + Clone + std::fmt::Debug,
    V2: TensorLike<Elem = T>,
{
    assert!(left.broadcastable(right.shape())); // TODO: figure out broadcasting
    let _length = max(right.shape().len(), left.shape().len());
    let max_shape = max_shape(&left.shape()[..], &right.shape()[..]);
    let index_iter = IndexIterator::new(max_shape.clone());
    let mut result = RawTensor::new_with_filler(max_shape, T::zero());
    for idx in index_iter {
        let v = *left.deref().get(&idx).unwrap() + *right.deref().get(&idx).unwrap();
        if let Err(e) = result.set(&idx, v) {
            panic!("{}", e)
        }
    }
    result
}

#[test]
fn test_add_jvp() {
    for (case_number, (left, right, grad, expected)) in vec![
        (
            RcTensor::from([1.0, 2.0, 3.0]),
            RcTensor::from([1.0, 2.0, 3.0]),
            RcTensor::from([[1.0, 1.0, 1.0]]),
            RcTensor::from([[1.0, 1.0, 1.0]]),
        ),
        (
            RcTensor::from([10.0, -2.0, 3.0, 4.0]),
            RcTensor::from([1.0, 2.0, 3.0, 0.1]),
            RcTensor::from([[1.0, 1.0, 1.0, 1.0]]),
            RcTensor::from([[1.0, 1.0, 1.0, 1.0]]),
        ),
        (
            RcTensor::from([1.0, 2.0, 3.0]),
            RcTensor::from([1.0, 2.0, 3.0]),
            RcTensor::from([[1.0, 2.0, 3.0]]),
            RcTensor::from([[1.0, 2.0, 3.0]]),
        ),
        (
            RcTensor::from([1.0, 2.0, 3.0]),
            RcTensor::from([1.0, 2.0, 3.0]),
            RcTensor::from([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            RcTensor::from([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let count = left.count();
        let grad_shape = grad.shape().to_vec();
        let jvp = add_jvp(vec![left, right], vec![grad]);
        assert_eq!(
            expected.shape(),
            &vec![grad_shape[0], count],
            "case_number: {case_number}"
        );
        assert_eq!(jvp[0], expected, "case_number: {case_number}");
    }
}

#[test]
fn test_sum_backward() {
    let input = RcTensor::from([1.0, 2.0, 3.0]);
    input.sum().backward();
    dbg!(&input.grad());
    assert_eq!(
        input.grad(),
        RcTensor::from([1.0, 1.0, 1.0]) // RcTensor::from([[1.0], [1.0], [1.0]])
    );
}

#[test]
fn test_tanh_sets_grad() {
    let input = RcTensor::from([0.666]);
    let epsilon = 1e-12 as f64;
    let output = tanh(&input).sum();
    let output_perturbed = tanh(&(&input + &RcTensor::scalar(epsilon))).sum();
    let numerical_derivative = &RcTensor::scalar(1.0 / epsilon) * &(&output_perturbed - &output);
    output
        .grad_fn
        .clone()
        .unwrap()
        .compute_jvp(vec![RcTensor::scalar(1.0 as f64)]);
    let input = dbg!(input);
    let grad = input.get_grad().take().unwrap();
    dbg!("input.get_grad()={:?}", input.get_grad().clone());
    let abs_diff = (&numerical_derivative - &grad).abs();
    assert!(abs_diff.sum().elem() <= 2e-4);
}

#[test]
fn test_multi_dimensional_tanh() {
    // TODO: get this to work with non-scalar inputs
    let input = RcTensor::from([[0.666, 12.0], [-3.2, -0.1]]);
    let output = tanh(&tanh(&input)).sum();
    let expected = RcTensor::from([[0.4792, 0.0000], [0.0028, 0.9803]]);
    output
        .grad_fn
        .clone()
        .unwrap()
        .compute_jvp(vec![RcTensor::scalar(1.0 as f64)]);
    // let grad = output.derivative.clone().unwrap().compute();
    let grad = input.get_grad().take().unwrap();
    let abs_diff = (&expected - &grad).abs();

    assert!(abs_diff.sum().elem() <= 2e-4);
}

#[test]
fn test_tanh_twice_sets_grad() {
    // TODO: get this to work with non-scalar inputs
    let _input = RcTensor::from([[0.666, 12.0], [-3.2, -0.1]]);
    let input = RcTensor::from([0.666]);
    let epsilon = 1e-12 as f64;
    let output = tanh(&tanh(&input)).sum();
    let output_perturbed = tanh(&tanh(&(&input + &RcTensor::scalar(epsilon)))).sum();
    println!(
        "output_perturbed=
    {output_perturbed:?}"
    );
    println!(
        "output=
    {output:?}"
    );
    println!("____={:?}\n\n", RcTensor::scalar(epsilon));
    let numerical_derivative = &RcTensor::scalar(1.0 / epsilon) * &(&output_perturbed - &output);
    output.backward();
    let grad = input.get_grad().take().unwrap();
    // dbg!("input.get_grad()={:?}", input.get_grad().clone());
    let abs_diff = (&numerical_derivative - &grad).abs();
    println!(
        "numerical_derivative=
    {numerical_derivative:?}\n\n"
    );

    println!(
        "grad=
        {grad:?}\n\n"
    );
    println!("abs_diff.sum()={:?}", abs_diff.sum());
    assert!(abs_diff.sum().elem() <= 2e-4);
}

#[test]
fn test_tanh_derivative() {
    let input = RcTensor::new((1..65).map(|x| x as f64).collect(), vec![4, 4, 4]);
    let epsilon = 1e-12 as f64;
    let epsilon_tensor = RcTensor::new_with_filler(vec![4, 4, 4], epsilon);
    println!(
        "epsilon_tensor.abs().sum()={:?}",
        epsilon_tensor.abs().sum()
    );
    let perturbed_input = &input + &epsilon_tensor;
    println!("input.abs().sum()={:?}", input.abs().sum());
    println!(
        "perturbed_input.abs().sum()={:?}",
        perturbed_input.abs().sum()
    );
    let output = tanh(&input);
    let output_perturbed = tanh(&perturbed_input);
    println!("output.abs().sum()={:?}", output.abs().sum());
    println!(
        "output_perturbed.abs().sum()={:?}",
        output_perturbed.abs().sum()
    );
    let numerical_derivative = &RcTensor::scalar(1.0 / epsilon) * &(&output_perturbed - &output);
    let calculated_derivative = tanh_derivative(&input, false);
    println!(
        "numerical_derivative.abs().sum()={:?}",
        numerical_derivative.abs().sum()
    );
    println!(
        "calculated_derivative.abs().sum()={:?}",
        calculated_derivative.abs().sum()
    );
    println!(
        "numerical_derivative=
    {numerical_derivative:?}"
    );

    println!(
        "calculated_derivative=
        {calculated_derivative:?}"
    );
    let abs_diff = (&numerical_derivative - &calculated_derivative).abs();
    println!("abs_diff.sum()={:?}", abs_diff.sum());
    assert!(abs_diff.sum().elem() / 64.0 <= 1e-5);
}

#[test]
fn test_add() {
    let tensor1 = RcTensor::new_with_filler(vec![4, 4], 1);
    let tensor2 = RcTensor::new((0..32).collect(), vec![2, 4, 4]);
    let tensor3 = RcTensor::new((1..33).collect(), vec![2, 4, 4]);
    assert_eq!(&tensor2 + &tensor1, tensor3);
    assert_eq!(&tensor1 + &tensor2, tensor3);
    assert_eq!(tensor1 + tensor2, tensor3);
}
