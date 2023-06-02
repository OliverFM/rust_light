use super::numeric::*;
use crate::tensor::{ElementIterator, RawTensor, RcTensor, TensorLike};
use num::traits::real::Real;

#[derive(Debug, PartialEq, Clone)]
pub struct Derivative<T: Numeric> {
    inputs: Vec<RcTensor<T>>,
    // TODO: remove the need to take ownership here
    derivative: fn(Vec<RcTensor<T>>) -> RcTensor<T>,
}

fn pass_through<T: Numeric>(tensors: Vec<RcTensor<T>>) -> RcTensor<T> {
    tensors[0].clone()
}

impl<T: Numeric> Default for Derivative<T> {
    fn default() -> Self {
        Derivative {
            inputs: Vec::new(),
            derivative: pass_through,
        }
    }
}

impl<T: Numeric> Derivative<T>
// where RcTensor<T>: Mul<Output = RcTensor<T>>
{
    pub fn new(
        inputs: Vec<RcTensor<T>>,
        derivative: fn(Vec<RcTensor<T>>) -> RcTensor<T>,
    ) -> Derivative<T> {
        Derivative { inputs, derivative }
    }
    pub(crate) fn pass_grad_back(&self, grad: RcTensor<T>) {
        self.inputs[0].set_grad(grad.clone());
        self.inputs[0]
            .derivative
            .iter()
            .for_each(|d| d.pass_grad_back(grad.clone()))
    }

    pub fn compute(&self) -> RcTensor<T> {
        // TODO: add chain rule in
        assert!(self.inputs.len() <= 1);

        // f(g(h(x))) how do i set x.grad if we are now computing f'
        // grad = f'(g(hx)) g'(h(x)) h'(x)
        // f(g(h(x), z)) how do i set x.grad if we are now computing f'
        let grad = if let Some(grad) = self.inputs[0].compute_grad() {
            let self_grad = (self.derivative)(self.inputs.clone());
            // TODO: consider moving this logic into a generalised_mul function.
            if self_grad.is_scalar() || self_grad.shape().iter().product::<usize>() == 1 {
                assert!(grad.is_scalar() || grad.shape().iter().product::<usize>() == 1);
                self_grad * grad
            } else {
                self_grad.bmm(&grad)
            }
        } else {
            (self.derivative)(self.inputs.clone())
        };
        println!("grad={:?}", &grad);
        self.pass_grad_back(grad.clone());
        grad
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
        // println!("tanh^-1({elem:?})={v:?}");
        array.push(v);
    }
    // RcTensor::new(array, tensor.shape().clone())
    RcTensor::new(array, tensor.shape().clone())
}

#[test]
fn test_tanh_twice_sets_grad() {
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
    output.derivative.clone().unwrap().compute();
    let grad = input.get_grad().take().unwrap();
    println!("input.get_grad()={:?}", input.get_grad());
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
fn test_tanh_twice() {
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
    println!("epsilon={:?}\n\n", RcTensor::scalar(epsilon));
    let numerical_derivative = &RcTensor::scalar(1.0 / epsilon) * &(&output_perturbed - &output);
    // output.derivative.clone().unwrap().compute();
    // let grad = input.get_grad().take().unwrap();
    let grad = output.derivative.clone().unwrap().compute();
    println!("input.get_grad()={:?}", input.get_grad());
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
    println!("abs_diff.sum()={:?}", abs_diff.sum());
    assert!(abs_diff.sum().elem() / 64.0 <= 1e-5);
    let grad = output.derivative.clone().unwrap().compute();
    let abs_diff2 = (&calculated_derivative - &grad).abs();
    println!("abs_diff2.sum()={:?}", abs_diff2.sum());
    assert!(abs_diff2.sum().elem() / 64.0 <= 1e-15);
}
