use super::tensor::{Numeric, Tensor, TensorLike};

pub struct LinearLayer<T>
where
    T: Numeric,
{
    weights: Tensor<T>,
    bias: Tensor<T>,
}

impl<T> LinearLayer<T>
where
    T: Numeric,
{
    pub fn forward<U>(&self, batch: &U) -> Tensor<T>
    where
        U: for<'b> TensorLike<'b, Elem = T>,
    {
        let y = &self.weights * batch;
        println!("y.shape()={:?}", y.shape());
        println!("bias.shape()={:?}", self.bias.shape());

        &y + &self.bias
    }
}

#[test]
fn test_layer() {
    let layer = LinearLayer {
        weights: Tensor::new_with_filler(vec![1, 2, 2], 1),
        bias: Tensor::new_with_filler(vec![1, 2, 1], 1),
    };
    let input = Tensor::new(vec![1, 2], vec![2, 1]);
    let res = layer.forward(&input);
    let expected = Tensor::new(vec![4, 4], vec![1, 2, 1]);

    assert_eq!(res, expected);
}
