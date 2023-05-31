use super::tensor::{ElementIterator, Numeric, RcTensor, TensorLike, TensorLikePublic};

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
        let y = self.weights.bmm(batch);
        println!("y.shape()={:?}", y.shape());
        println!("bias.shape()={:?}", self.bias.shape());

        &y + &self.bias
    }
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
