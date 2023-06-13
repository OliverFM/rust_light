use crate::nn::{Linear, Module};
#[allow(unused_imports)] // needed for .abs().sum()
use crate::tensor::TensorLike;
use crate::tensor::{Numeric, RcTensor, TensorList};

// struct FeedForward<T: Numeric> {
//     layers: Vec<dyn Module<T>>,
// }
//
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
