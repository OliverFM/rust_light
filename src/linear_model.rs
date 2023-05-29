use super::tensor::{ElementIterator, Numeric, RcTensor, TensorLike};
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
        U: TensorLike<Elem = T>,
    {
        let y = &self.weights * batch;
        println!("y.shape()={:?}", y.shape());
        println!("bias.shape()={:?}", self.bias.shape());

        &y + &self.bias
    }
}

fn tanh<T: Numeric + Real>(tensor: RcTensor<T>) -> RcTensor<T> {
    let length = tensor.shape().iter().fold(1, |acc, x| acc * *x);
    let mut array = Vec::with_capacity(length);
    for elem in ElementIterator::new(&tensor) {
        array.push(elem.tanh());
    }
    RcTensor::new(array, tensor.shape().clone())
}

fn tanh_derivative<T: Numeric + Real>(tensor: RcTensor<T>) -> RcTensor<T> {
    let length = tensor.shape().iter().fold(1, |acc, x| acc * *x);
    let mut array = Vec::with_capacity(length);
    for elem in ElementIterator::new(&tensor) {
        let _v = T::one() - T::one() / elem.tanh().powi(2);
        array.push(elem.tanh());
    }
    RcTensor::new(array, tensor.shape().clone())
}

#[test]
fn test_tanh_derivative() {
    let _input = RcTensor::new((0..64).collect(), vec![4, 4, 4]);
    // let epsilon = 1e-7 as f64;
    // let epsilon_tensor = Tensor::new_with_filler(epsilon, vec![4, 4, 4]);
    // let perturbed_input = &input + &epsilon_tensor;
    // let output = tanh(input);
    // let output_perturbed = tanh(perturbed_input);
    // let numerical_derivative = (1 / epsilon) * (output + (-1.0) * output_perturbed);
    // let calculated_derivative = tanh_derivative(output);

    // let abs_diff = (numerical_derivative + (-1.0) * calculated_derivative).abs();
    // assert!(abs_diff.sum() / 64.0 <= 1e-5);
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
