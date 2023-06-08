use crate::tensor::numeric::*;

use crate::tensor::utils::ElementIterator;
use crate::tensor::{RawTensor, RcTensor, TensorLike, TensorList};
use num::traits::real::Real;

#[derive(Debug, PartialEq, Clone)]
pub(in crate::tensor) struct Derivative<T: Numeric> {
    inputs: TensorList<T>,
    // TODO: remove the need to take ownership here
    /// signature: jvp(inputs, jacobians) -> jvp
    jacobian_vector_product: fn(TensorList<T>, TensorList<T>) -> TensorList<T>,
}

impl<T: Numeric> Derivative<T> {
    pub fn new(
        inputs: TensorList<T>,
        jacobian_vector_product: fn(TensorList<T>, TensorList<T>) -> TensorList<T>,
        // TODO: switch this to working on views.
        // OR just force everythng to be flat... and set to the same shape at the end...
    ) -> Derivative<T> {
        Derivative {
            inputs,
            jacobian_vector_product,
        }
    }

    /// Computes the product outer_grads @ JVP(self.input)
    /// Recursively expanding the JVP of the input with the chain rule
    /// e.g. for f(g(h(x))), jvp(1.0) = jvp(1.0 * f'(g(h(x))))
    pub fn compute_jvp(&self, outer_grads: Vec<RcTensor<T>>) {
        // f(g(h(x))) how do i set x.grad if we are now computing f'
        // grad = f'(g(hx)) g'(h(x)) h'(x)
        // f(g(h(x), z)) how do i set x.grad if we are now computing f'
        let self_grads = (self.jacobian_vector_product)(self.inputs.clone(), outer_grads);
        for (grad, input) in self_grads.iter().zip(self.inputs.iter()) {
            dbg!(&grad, &input);
            debug_assert_eq!(
                grad.count(),
                input.count(),
                "grad and input must have the same number of elements"
            );
            let shaped_grad = RcTensor::new(grad.0.array.clone(), input.shape().to_vec());
            input.update_grad(shaped_grad);
            if let Some(d) = input.grad_fn.as_ref() {
                d.compute_jvp(vec![grad.clone()]);
            }
        }
    }
}

/// When used for making a jacobian_vector_product function
/// left ought to be the gradient passed in and right is the computed jacobian
pub(in crate::tensor) fn jvp_from_full_jacobians<T: Numeric>(
    left: RcTensor<T>,
    right: RcTensor<T>,
) -> RcTensor<T> {
    if right.is_scalar()
        || right.shape().iter().product::<usize>() == 1
        || left.is_scalar()
        || left.shape().iter().product::<usize>() == 1
    {
        assert!(left.is_scalar() || left.shape().iter().product::<usize>() == 1);
        left * right
    } else {
        left.bmm(&right)
    }
}

pub fn ones<T: Numeric>(tensors: Vec<RcTensor<T>>, grads: Vec<RcTensor<T>>) -> Vec<RcTensor<T>> {
    assert!(tensors.len() == 1);
    assert!(grads.len() == 1);
    let length_1 = tensors[0].shape().iter().product::<usize>();
    let length_2 = grads[0].shape().iter().product::<usize>();
    let raw_tensor = RawTensor::new_with_filler(vec![length_2, length_1], T::one());
    dbg!("sum_backward", vec![RcTensor::from_raw(raw_tensor)]).1
}

pub fn tanh<T: Numeric + Real>(tensor: &RcTensor<T>) -> RcTensor<T> {
    let length = tensor.shape().iter().fold(1, |acc, x| acc * *x);
    let mut array = Vec::with_capacity(length);
    for elem in ElementIterator::new(tensor) {
        array.push(elem.tanh());
    }
    let mut raw_tensor = RawTensor::new(array, tensor.shape().clone());
    raw_tensor.grad_fn = Some(Derivative::new(vec![tensor.clone()], tanh_derivative_outer));
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

#[test]
fn test_sum_backward() {
    let input = RcTensor::from([1.0, 2.0, 3.0]);
    input.sum().backward();
    assert_eq!(
        *input.get_grad().borrow().as_ref().unwrap(),
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
