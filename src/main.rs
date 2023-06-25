// use rust_light as rlt;
use rust_light::nn::{Linear, Mlp, Module};
use rust_light::optim::sgd::sgd_step;
use rust_light::tensor::functional;
use rust_light::tensor::{RcTensor, TensorLike};

fn main() {
    let mut mlp = Mlp::new([
        Linear::new(
            RcTensor::from([[1.0, 1e-2, -1e-3, -2.0], [-1.1, 0., 0., 0.7]]),
            RcTensor::new_with_filler(vec![1, 4], 1.0),
            Some(functional::tanh),
        ),
        Linear::new(
            RcTensor::from([[1.0, -2.0], [-1.1, 0.7], [0.1, -0.2], [0.1, 0.0]]),
            RcTensor::new_with_filler(vec![1, 2], 1.0),
            Some(functional::tanh),
        ),
    ]);
    let input = RcTensor::new(vec![1.0, 2.0], vec![1, 2]);
    let expected = RcTensor::new(vec![-1.0, 1.0], vec![1, 2]);

    for i in 0..301 {
        let res = mlp.forward(input.clone());
        // maybe the issue is because expected has no grad?
        let loss = (&res - &expected).abs().sum();
        // let loss = (&res).abs().sum();
        println!("loss={}", loss);
        loss.backward();

        let step_size = RcTensor::scalar(1e-2);
        sgd_step(&mut mlp, step_size);
        if i % 10 == 0 {
            println!("res={}", res);
        }
        expected.zero_grad();
    }

    println!("\n\nsuccess!!!")
}
