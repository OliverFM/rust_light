use super::numeric::*;


use crate::tensor::{ElementIterator, RawTensor, RcTensor, TensorLike};
use num::traits::real::Real;

// use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
pub(in crate::tensor) struct Derivative<T: Numeric> {
    // TODO: what happens if left and right are refering to the same underlying tensor?
    // In that case is is not so clear how to assign gradients.
    // e.g. d/dx(xy) = y, but d/dx(x^2) = 2x. Need to figure this one out.
    // Even bigger issue: what if we have f(h(x), g(x))
    inputs: Vec<RcTensor<T>>,
    // TODO: remove the need to take ownership here
    jacobian_vector_product: fn(Vec<RcTensor<T>>) -> Vec<RcTensor<T>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct DerivativeConst<T: Numeric, const N: usize> {
    // TODO: switch to using this
    inputs: Box<[RcTensor<T>; N]>,
    derivative: fn(Box<[RcTensor<T>; N]>) -> Box<[RcTensor<T>; N]>,
}

impl<T: Numeric> Derivative<T> {
    pub fn new(
        inputs: Vec<RcTensor<T>>,
        derivative: fn(Vec<RcTensor<T>>) -> Vec<RcTensor<T>>,
        // TODO: switch this to working on views.
        // OR just force everythng to be flat... and set to the same shape at the end...
    ) -> Derivative<T> {
        Derivative {
            inputs,
            jacobian_vector_product: derivative,
        }
    }

    // pub fn compute(&self) -> RcTensor<T> {
    //     assert!(self.inputs.len() <= 1);
    //     // f(g(h(x))) how do i set x.grad if we are now computing f'
    //     // grad = f'(g(hx)) g'(h(x)) h'(x)
    //     // f(g(h(x), z)) how do i set x.grad if we are now computing f'
    //     let grad = if let Some(grads) = self.inputs[0].compute_grad() {
    //         let input_grad: RcTensor<T> = grads;
    //         let self_grad: RcTensor<T> =
    //             (self.jacobian_vector_product)(self.inputs.clone())[0].clone();
    //         // TODO: consider moving this logic into a generalised_mul function.
    //         if self_grad.is_scalar() || self_grad.shape().iter().product::<usize>() == 1 {
    //             assert!(
    //                 input_grad.is_scalar() || input_grad.shape().iter().product::<usize>() == 1
    //             );
    //             self_grad * input_grad
    //         } else {
    //             self_grad.bmm(&input_grad)
    //         }
    //     } else {
    //         (self.jacobian_vector_product)(self.inputs.clone())[0].clone()
    //     };
    //     println!("grad={:?}", &grad);
    //     grad
    // }

    /// Computes the product outer_grads @ JVP(self.input)
    /// Recursively expanding the JVP of the input with the chain rule
    /// e.g. for f(g(h(x))), jvp(1.0) = jvp(1.0 * f'(g(h(x))))
    pub fn compute_jvp(&self, outer_grads: Vec<RcTensor<T>>) {
        assert!(self.inputs.len() == 1);
        assert!(outer_grads.len() == 1);

        // f(g(h(x))) how do i set x.grad if we are now computing f'
        // grad = f'(g(hx)) g'(h(x)) h'(x)
        // f(g(h(x), z)) how do i set x.grad if we are now computing f'
        let self_grads = (self.jacobian_vector_product)(self.inputs.clone());
        let self_grad = self_grads[0].clone();
        let outer_grad = outer_grads[0].clone();
        self.inputs[0].set_grad(self_grad.clone());
        dbg!(outer_grad.shape().clone());
        dbg!(self_grad.shape().clone());
        let new_grad = if self_grad.is_scalar()
            || self_grad.shape().iter().product::<usize>() == 1
            || outer_grad.is_scalar()
            || outer_grad.shape().iter().product::<usize>() == 1
        {
            assert!(outer_grad.is_scalar() || outer_grad.shape().iter().product::<usize>() == 1);
            outer_grad * self_grad
        } else {
            outer_grad.bmm(&self_grad)
        };

        let shaped_grad = RcTensor::new(new_grad.0.array.clone(), self.inputs[0].shape().to_vec());
        self.inputs[0].set_grad(shaped_grad);
        if let Some(d) = self.inputs[0].derivative.as_ref() {
            d.compute_jvp(vec![new_grad])
        }
    }
}

pub fn ones<T: Numeric>(tensors: Vec<RcTensor<T>>) -> Vec<RcTensor<T>> {
    assert!(tensors.len() == 1);
    let length = tensors[0].shape().iter().product::<usize>();
    let raw_tensor = RawTensor::new_with_filler(vec![1, length], T::one());
    dbg!("sum_backward", vec![RcTensor::from_raw(raw_tensor)]).1
}

pub fn tanh<T: Numeric + Real>(tensor: &RcTensor<T>) -> RcTensor<T> {
    let length = tensor.shape().iter().fold(1, |acc, x| acc * *x);
    let mut array = Vec::with_capacity(length);
    for elem in ElementIterator::new(tensor) {
        array.push(elem.tanh());
    }
    let mut raw_tensor = RawTensor::new(array, tensor.shape().clone());
    raw_tensor.derivative = Some(Derivative::new(vec![tensor.clone()], tanh_derivative_outer));
    RcTensor::from_raw(raw_tensor)
}

fn tanh_derivative_outer<T: Numeric + Real>(tensors: Vec<RcTensor<T>>) -> Vec<RcTensor<T>> {
    assert!(tensors.len() == 1);
    vec![tanh_derivative(&tensors[0], true)]
}

fn tanh_derivative<T: Numeric + Real>(
    tensor: &RcTensor<T>,
    full_jacobian_output: bool,
) -> RcTensor<T> {
    // let length = tensor.shape().iter().fold(1, |acc, x| acc * *x);
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
    output
        .derivative
        .clone()
        .unwrap()
        .compute_jvp(vec![RcTensor::scalar(1.0 as f64)]);
    let input = dbg!(input);
    let grad = input.get_grad().take().unwrap();
    dbg!("input.get_grad()={:?}", input.get_grad().clone());
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
fn test_multi_dimensional_tanh() {
    // TODO: get this to work with non-scalar inputs
    let input = RcTensor::from([[0.666, 12.0], [-3.2, -0.1]]);
    let output = tanh(&tanh(&input)).sum();
    let expected = RcTensor::from([[0.4792, 0.0000], [0.0028, 0.9803]]);
    output
        .derivative
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
    // let grad = output.derivative.clone().unwrap().compute();
    // let abs_diff2 = (&calculated_derivative - &grad).abs();
    // println!("abs_diff2.sum()={:?}", abs_diff2.sum());
    // assert!(abs_diff2.sum().elem() / 64.0 <= 1e-15);
}
