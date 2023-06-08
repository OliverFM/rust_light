use rust_light::nn::linear::Linear;
use rust_light::tensor::{Numeric, RcTensor, TensorLike};

fn main() {
    let layer = Linear::new(
        RcTensor::new_with_filler(vec![1, 2, 2], 1),
        RcTensor::new_with_filler(vec![1, 2, 1], 1),
    );
    let input = RcTensor::new(vec![1, 2], vec![2, 1]);
    let res = layer.forward(&input);
    let expected = RcTensor::new(vec![4, 4], vec![1, 2, 1]);

    assert_eq!(res, expected);
    println!("res={res:?}");
}
