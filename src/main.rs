// use rust_light as rlt;
use rust_light::nn::linear::Linear;
use rust_light::tensor::{Numeric, RcTensor, TensorLike};

fn main() {
    let mut layer = Linear::new(
        RcTensor::from([[1.0, -2.0], [-1.1, 0.7]]),
        RcTensor::new_with_filler(vec![1, 2, 1], 1.0),
        None,
    );
    let input = RcTensor::new(vec![1.0, 2.0], vec![2, 1]);
    let res = layer.forward(input);
    res.sum().backward();
    layer.weights.grad();
    layer.bias.grad();

    let step_size = RcTensor::scalar(1e-3);
    layer.weights = layer.weights.clone() - &step_size * &layer.weights.grad();

    println!("res={res:?}");
    println!("\n\nsuccess!!!")
}
