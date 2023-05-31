use super::numeric::*;
use crate::tensor::{ElementIterator, RawTensor, RcTensor, SliceRange, TensorLike};
use num::traits::real::Real;

#[derive(Debug, PartialEq, Clone)]
pub struct Derivative<T: Numeric> {
    inputs: Vec<RcTensor<T>>,
    // TODO: remove the need to take ownership here
    derivative: fn(Vec<RcTensor<T>>) -> RcTensor<T>,
}

impl<T: Numeric> Derivative<T> {
    pub fn new(
        inputs: Vec<RcTensor<T>>,
        derivative: fn(Vec<RcTensor<T>>) -> RcTensor<T>,
    ) -> Derivative<T> {
        Derivative { inputs, derivative }
    }

    pub fn compute(&self) -> RcTensor<T> {
        // TODO: add chain rule in
        (self.derivative)(self.inputs.clone())
    }
}

pub fn ones<T: Numeric>(tensors: Vec<RcTensor<T>>) -> RcTensor<T> {
    assert!(tensors.len() == 1);
    let raw_tensor = RawTensor::new_with_filler(tensors[0].shape().to_vec(), T::one());
    RcTensor::from_raw(raw_tensor)
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

fn tanh_derivative_outer<T: Numeric + Real>(tensors: Vec<RcTensor<T>>) -> RcTensor<T> {
    tanh_derivative(&tensors[0])
}

fn tanh_derivative<T: Numeric + Real>(tensor: &RcTensor<T>) -> RcTensor<T> {
    let length = tensor.shape().iter().fold(1, |acc, x| acc * *x);
    let mut array = Vec::with_capacity(length);
    for elem in ElementIterator::new(tensor) {
        let v = T::one() - elem.tanh().powi(2);
        println!("tanh^-1({elem:?})={v:?}");
        array.push(v);
    }
    // RcTensor::new(array, tensor.shape().clone())
    RcTensor::new(array, tensor.shape().clone())
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
    let calculated_derivative = tanh_derivative(&input);
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
    println!("abs_diff.sum()={}", abs_diff.sum());
    assert!(abs_diff.sum() / 64.0 <= 1e-5);
    let grad = output.derivative.clone().unwrap().compute();
    let abs_diff2 = (&calculated_derivative - &grad).abs();
    println!("abs_diff2.sum()={}", abs_diff2.sum());
    assert!(abs_diff2.sum() / 64.0 <= 1e-15);
}
