use crate::nn::{Linear, Module};
#[allow(unused_imports)] // needed for .abs().sum()
use crate::tensor::TensorLike;
use crate::tensor::{Numeric, RcTensor, TensorList};

#[derive(Debug)]
pub struct Mlp<T, const N: usize>
where
    T: Numeric,
{
    layers: [Linear<T>; N],
}

impl<T: Numeric, const N: usize> Mlp<T, N> {
    pub fn new(layers: [Linear<T>; N]) -> Mlp<T, N> {
        Mlp { layers }
    }
}

impl<T: Numeric, const N: usize> crate::nn::module::private::Private for Mlp<T, N> {}

impl<T: Numeric, const N: usize> Module<T> for Mlp<T, N> {
    type InputType = RcTensor<T>;
    type OutputType = RcTensor<T>;

    fn forward(&self, batch_ref: RcTensor<T>) -> RcTensor<T> {
        self.layers
            .iter()
            .fold(batch_ref, |prev, layer| layer.forward(prev))
    }

    fn params(&self) -> TensorList<T> {
        self.layers
            .iter()
            .flat_map(|layer| layer.params())
            .collect()
    }

    fn update_params(&mut self, new_params: TensorList<T>) {
        let mut param_iter = new_params.into_iter().rev();
        for i in (0..self.layers.len()).rev() {
            let bias = param_iter.next().unwrap();
            let weights = param_iter.next().unwrap();
            (self.layers[i]).update_params(vec![weights, bias]);
        }
    }
}

#[test]
fn test_mlp_creation() {
    use crate::tensor::functional;
    Mlp::new([
        Linear::new(
            RcTensor::from([[1.0, -2.0], [-1.1, 0.7]]),
            RcTensor::new_with_filler(vec![1, 2], 1.0),
            Some(functional::tanh),
        ),
        Linear::new(
            RcTensor::from([[1.0, -2.0], [-1.1, 0.7]]),
            RcTensor::new_with_filler(vec![1, 2], 1.0),
            Some(functional::tanh),
        ),
    ]);
}

#[test]
fn test_mlp() {
    use crate::optim::sgd::sgd_step;
    use crate::tensor::functional;

    for mut mlp in vec![Mlp::new([
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
    ])] {
        let input = RcTensor::new(vec![1.0, 2.0], vec![1, 2]);
        let expected = RcTensor::new(vec![-1.0, 1.0], vec![1, 2]);

        for _ in 0..31 {
            let res = mlp.forward(input.clone());
            // maybe the issue is because expected has no grad?
            let loss = (&res - &expected).abs().sum();
            // let loss = (&res).abs().sum();
            // println!("loss={}", loss);
            loss.backward();

            let step_size = RcTensor::scalar(1e-2);
            sgd_step(&mut mlp, step_size);
            // if i % 10 == 0 {
            //     println!("res={}", res);
            // }
            expected.zero_grad();
        }
        let res = mlp.forward(input.clone());
        let loss = (&res - &expected).abs().sum();
        assert!(loss.elem() < 0.2, "loss={loss}");
    }
}

#[ignore] // WAAAAAAAYYYYYY too slow to run every time
#[test]
fn test_mlp_fits_random_function() {
    use crate::optim::sgd::sgd_step;
    use crate::tensor::functional;
    use rand::prelude::*;
    use rand_distr::Normal;

    // let rng = SeedableRng::new(42);
    // let mut rng = thread_rng();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 0.1).unwrap();
    let num_inputs = 4;
    let input_shape = vec![1, 2];
    let output_shape = vec![1, 2];
    // let inputs = (0..num_inputs).iter().map()
    // let mut inputs = Vec::with_capacity(num_inputs);
    type T = f32;
    let mut sample_value = || normal.sample(&mut rng);
    let mut sample_vector = |shape: &[usize]| {
        RcTensor::new(
            (0..shape.iter().product::<usize>())
                .map(|_| sample_value())
                .collect(),
            shape.to_vec(),
        )
    };
    let inputs: Vec<_> = (0..num_inputs)
        .map(|_| (sample_vector(&input_shape), sample_vector(&output_shape)))
        .collect();
    // let inputs = (0..num_inputs).iter().map(|_|  /)
    assert_eq!(inputs.len(), num_inputs);
    for mut mlp in vec![Mlp::new([
        Linear::new(
            // sample_vector(&vec![2, 4]),
            // sample_vector(&vec![1, 4]),
            sample_vector(&vec![2, 32]),
            sample_vector(&vec![1, 32]),
            Some(functional::relu),
        ),
        Linear::new(
            // RcTensor::from([[1.0, -2.0], [-1.1, 0.7], [0.1, -0.2], [0.1, 0.0]]),
            // RcTensor::new_with_filler(vec![1, 2], 1.0),
            sample_vector(&vec![32, 2]),
            sample_vector(&vec![1, 2]),
            Some(functional::tanh),
        ),
    ])] {
        for i in 0..501 {
            let input_idx: usize = rng.gen::<usize>() % num_inputs;
            let (input, expected): (RcTensor<_>, RcTensor<_>) = inputs[input_idx].clone();
            let res = mlp.forward(input.clone());
            // maybe the issue is because expected has no grad?
            let loss = (&res - &expected).abs().sum();
            // let loss = (&res).abs().sum();
            // println!("loss={}", loss);
            loss.backward();

            let step_size = RcTensor::scalar(1e-3);
            sgd_step(&mut mlp, step_size);
            if i % 50 == 0 {
                println!(
                    "
                    mlp.layers[0].weights={}
                    mlp.layers[1].weights={}

                    input_idx={}, input={}, res={}, expected={}, loss={}",
                    &mlp.layers[0].weights,
                    &mlp.layers[1].weights,
                    input_idx,
                    input,
                    res,
                    expected,
                    loss
                );
            }
            expected.zero_grad();
        }
        for (input, expected) in inputs.iter() {
            let res = mlp.forward(input.clone());
            let loss = (res.clone() - expected.clone()).abs().sum();
            assert!(loss.elem() < 0.2, "loss={loss}");
        }
    }
}
