use num::traits::real::Real;

use crate::nn::linear::Linear;
use crate::tensor::{Numeric, Scalar};

// struct Sgd<T: Numeric + Real> {
//     step_size: Scalar<T>,
//     // module: &'a mut Linear<T>,
// }
//
// impl<T: Numeric + Real> Sgd<T> {
//     pub fn new(step_size: T) -> Self {
//         assert!(step_size > T::epsilon(), "Step size must be positive!");
//         Sgd {
//             step_size: RcTensor::scalar(step_size),
//         }
//     }
//
//     // pub fn zero_grad(&self, module: &mut Linear<T>) {
//     //     module.params().for_each(|p| p.zero_grad());
//     // }
//     //
//     // pub fn step(&self, &mut module: &mut Linear<T>) -> impl Iterator<Item = RcTensor<T>> + '_ {
//     //     module.params().map(|p| p - (&self.step_size * p.grad()))
//     // }
// }
//

pub fn sgd_step<T: Numeric + Real>(layer: &mut Linear<T>, step_size: Scalar<T>) {
    layer.weights = layer.weights.clone() - &step_size * &layer.weights.grad();
}
