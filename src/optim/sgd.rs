use num::traits::real::Real;

use crate::nn::Module;
use crate::tensor::{Numeric, Scalar, TensorLike};

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

pub fn sgd_step<T: Numeric + Real, U: Module<T>>(module: &mut U, step_size: Scalar<T>) {
    // dbg!(&layer.weights.0.);

    let new_params = module
        .params()
        .iter()
        .map(|p| {
            let new = p.clone() - &step_size * p.grad();
            new.zero_grad();
            new
        })
        .collect();
    module.update_params(new_params);
}
