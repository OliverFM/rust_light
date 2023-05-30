use super::tensor::{ElementIterator, Numeric, RcTensor, TensorLike, TensorLikePublic};
use num::traits::real::Real;

pub struct LinearLayer<T>
where
    T: Numeric,
{
    weights: RcTensor<T>,
    bias: RcTensor<T>,
}

impl<T> LinearLayer<T>
where
    T: Numeric,
{
    pub fn forward<U>(&self, batch: &U) -> RcTensor<T>
    where
        U: TensorLikePublic<Elem = T>,
    {
        let y = &self.weights * batch;
        println!("y.shape()={:?}", y.shape());
        println!("bias.shape()={:?}", self.bias.shape());

        &y + &self.bias
    }
}

fn tanh<T: Numeric + Real>(tensor: &RcTensor<T>) -> RcTensor<T> {
    let length = tensor.shape().iter().fold(1, |acc, x| acc * *x);
    let mut array = Vec::with_capacity(length);
    for elem in ElementIterator::new(tensor) {
        array.push(elem.tanh());
    }
    RcTensor::new(array, tensor.shape().clone())
}

fn tanh_derivative<T: Numeric + Real>(tensor: &RcTensor<T>) -> RcTensor<T> {
    let length = tensor.shape().iter().fold(1, |acc, x| acc * *x);
    let mut array = Vec::with_capacity(length);
    for elem in ElementIterator::new(tensor) {
        // let v = T::one() - T::one() / elem.tanh().powi(2);
        let v = T::one() - elem.tanh().powi(2);
        println!("tanh^-1({elem:?})={v:?}");
        array.push(v);
    }
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
}

#[test]
fn test_layer() {
    let layer = LinearLayer {
        weights: RcTensor::new_with_filler(vec![1, 2, 2], 1),
        bias: RcTensor::new_with_filler(vec![1, 2, 1], 1),
    };
    let input = RcTensor::new(vec![1, 2], vec![2, 1]);
    let res = layer.forward(&input);
    let expected = RcTensor::new(vec![4, 4], vec![1, 2, 1]);

    assert_eq!(res, expected);
}
