use std::ops::Deref;

use num::traits::real::Real;

use crate::tensor::{Numeric, RcTensor, TensorLike};

pub struct Linear<T>
where
    T: Numeric,
{
    pub weights: RcTensor<T>,
    pub bias: RcTensor<T>,
    activation: fn(RcTensor<T>) -> RcTensor<T>,
}

impl<T> Linear<T>
where
    T: Numeric + Real,
{
    pub fn forward<U, V>(&self, batch_ref: U) -> RcTensor<T>
    where
        U: Deref<Target = V> + std::fmt::Debug + Clone,
        V: TensorLike<Elem = T>,
    {
        let batch = batch_ref.to_tensor();
        let y = self.weights.bmm(&batch);

        (self.activation)(&y + &self.bias)
    }

    pub fn update_params(&mut self, weights: RcTensor<T>, bias: RcTensor<T>) {
        self.weights = weights;
        self.bias = bias;
    }

    pub fn params(&mut self) -> impl Iterator<Item = RcTensor<T>> {
        vec![self.weights.clone(), self.bias.clone()].into_iter()
    }

    pub fn new(
        weights: RcTensor<T>,
        bias: RcTensor<T>,
        activation: Option<fn(RcTensor<T>) -> RcTensor<T>>,
    ) -> Self {
        Linear {
            weights,
            bias,
            activation: match activation {
                Some(f) => f,
                None => |t| t,
            },
        }
    }
}

#[test]
fn test_layer_no_grad() {
    let layer = Linear::new(
        RcTensor::new_with_filler(vec![1, 2, 2], 1.0),
        RcTensor::new_with_filler(vec![1, 2, 1], 1.0),
        None,
    );
    let input = RcTensor::new(vec![1.0, 2.0], vec![2, 1]);
    let res = layer.forward(input);
    let expected = RcTensor::new(vec![4.0, 4.0], vec![1, 2, 1]);

    assert_eq!(res, expected);
}

#[test]
fn test_layer() {
    let layer = Linear::new(
        RcTensor::from([[1.0, -2.0], [-1.1, 0.7]]),
        RcTensor::new_with_filler(vec![1, 2, 1], 1.0),
        None,
    );
    let input = RcTensor::new(vec![1.0, 2.0], vec![2, 1]);
    let res = layer.forward(input);
    res.sum().backward();
    layer.weights.grad();
    layer.bias.grad();
}

#[ignore]
#[test]
fn test_layer_batch_tensor() {
    // TODO: figure out how to update bmm to work correctly with autograd
    let layer = Linear::new(
        RcTensor::from([[1.0, -2.0], [-1.1, 0.7]]),
        RcTensor::new_with_filler(vec![2], 1.0),
        None,
    );
    let input = RcTensor::new(vec![1.0, 2.0], vec![1, 2]);
    let res = layer.forward(input);
    res.sum().backward();
    layer.weights.grad();
    layer.bias.grad();
}
