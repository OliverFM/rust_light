use num::traits::real::Real;

use crate::nn::Module;
use crate::tensor::{Numeric, RcTensor, TensorLike, TensorList};

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

impl<T: Numeric> crate::nn::module::private::Private for Linear<T> {}

impl<T: Numeric> Module<T> for Linear<T> {
    type InputType = RcTensor<T>;
    type OutputType = RcTensor<T>;

    fn forward(&self, batch_ref: RcTensor<T>) -> RcTensor<T> {
        let batch = batch_ref.to_tensor();
        // let y = self.weights.bmm(&batch);
        let y = batch.bmm(&self.weights);

        (self.activation)(&y + &self.bias)
    }

    fn params(&self) -> TensorList<T> {
        vec![self.weights.clone(), self.bias.clone()]
    }
    fn update_params(&mut self, mut new_params: TensorList<T>) {
        self.bias = new_params.remove(1);
        self.weights = new_params.remove(0);
    }
}

#[test]
fn test_layer_no_grad() {
    let layer = Linear::new(
        RcTensor::new_with_filler(vec![2, 2], 1.0),
        RcTensor::new_with_filler(vec![1, 2], 1.0),
        None,
    );
    let input = RcTensor::new(vec![1.0, 2.0], vec![1, 2]);
    let res = layer.forward(input);
    let expected = RcTensor::new(vec![4.0, 4.0], vec![1, 2]);

    assert_eq!(res, expected);
}

#[test]
fn test_layer() {
    let layer = Linear::new(
        RcTensor::from([[1.0, -2.0], [-1.1, 0.7]]),
        RcTensor::new_with_filler(vec![1, 2], 1.0),
        None,
    );
    let input = RcTensor::new(vec![1.0, 2.0], vec![1, 2]);
    let res = layer.forward(input.clone());
    res.sum().backward();
    layer.weights.grad();
    layer.bias.grad();
    let res = layer.forward(input.clone());
    res.sum().backward();
    layer.weights.grad();
    layer.bias.grad();
}

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
