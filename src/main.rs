// use rust_light as rlt;
use rust_light::nn::{Linear, Mlp, Module};
use rust_light::optim::sgd::sgd_step;
use rust_light::tensor::functional;
use rust_light::tensor::{RcTensor, TensorLike};

use rand::prelude::*;
use rand_distr::Normal;

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 0.1).unwrap();
    let mut sample_value = || normal.sample(&mut rng);
    let mut sample_vector = |shape: &[usize]| {
        RcTensor::new(
            (0..shape.iter().product::<usize>())
                .map(|_| sample_value())
                .collect(),
            shape.to_vec(),
        )
    };

    /*
    let left = sample_vector(&vec![1024, 8192]);
    let right = sample_vector(&vec![8192, 1024]);
    let res = left.bmm(&right);
    */

    let mut mlp = Mlp::new([
        Linear::new(
            sample_vector(&[8, 8192]),
            sample_vector(&[1, 8192]),
            Some(functional::relu),
        ),
        Linear::new(
            sample_vector(&[8192, 8192]),
            sample_vector(&[1, 8192]),
            Some(functional::tanh),
        ),
        Linear::new(
            sample_vector(&[8192, 8]),
            sample_vector(&[1, 8]),
            Some(functional::tanh),
        ),
    ]);
    // let mut mlp = Mlp::new([
    //     Linear::new(
    //         sample_vector(&vec![8, 1024]),
    //         sample_vector(&vec![1, 1024]),
    //         Some(functional::relu),
    //     ),
    //     Linear::new(
    //         sample_vector(&vec![1024, 1024]),
    //         sample_vector(&vec![1, 1024]),
    //         Some(functional::tanh),
    //     ),
    //     Linear::new(
    //         sample_vector(&vec![1024, 8]),
    //         sample_vector(&vec![1, 8]),
    //         Some(functional::tanh),
    //     ),
    // ]);
    // let mut mlp = Mlp::new([
    //     Linear::new(
    //         sample_vector(&vec![8, 8]),
    //         sample_vector(&vec![1, 8]),
    //         Some(functional::relu),
    //     ),
    //     Linear::new(
    //         sample_vector(&vec![8, 8]),
    //         sample_vector(&vec![1, 8]),
    //         Some(functional::tanh),
    //     ),
    // ]);
    let input = sample_vector(&[1, 8]);
    let expected = RcTensor::new_with_filler(vec![1, 8], 1.0);

    for _i in 0..2 {
        let res = mlp.forward(input.clone());
        // maybe the issue is because expected has no grad?
        let loss = (&res - &expected).abs().sum();
        // let loss = (&res).abs().sum();
        // println!("loss={}", loss);
        loss.backward();

        let step_size = RcTensor::scalar(1e-2);
        sgd_step(&mut mlp, step_size);
        // if i % 10 == 0 {
        // println!("res={}", res);
        // }
        expected.zero_grad();
        input.zero_grad();
    }

    println!("\n\nsuccess!!!")
}
