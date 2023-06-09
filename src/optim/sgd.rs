use num::traits::real::Real;

use crate::nn::linear::Linear;
use crate::tensor::{Numeric, RcTensor, Scalar, TensorLike, TensorList};

struct SGD<T: Numeric + Real> {
    step_size: Scalar<T>,
}
//
// impl<T: Numeric + Real> SGD<T> {
//     pub fn new(step_size: T) -> Self {
//         assert!(step_size > T::epsilon(), "Step size must be positive!");
//         SGD {
//             step_size: RcTensor::scalar(step_size),
//         }
//     }
//
//     pub fn zero_grad(&self, module: &mut Linear<T>) {
//         module.params().for_each(|p| p.zero_grad());
//     }
//
//     pub fn step(&self, &mut module: &mut Linear<T>) -> impl Iterator<Item = RcTensor<T>> + '_ {
//         module.params().map(|p| p - (&self.step_size * p.grad()))
//     }
// }
