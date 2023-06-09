use crate::tensor::numeric::*;

use crate::tensor::utils::ElementIterator;
use crate::tensor::{RawTensor, RcTensor, TensorLike, TensorList};
use num::traits::real::Real;

#[derive(Debug, PartialEq, Clone)]
pub(in crate::tensor) struct Derivative<T: Numeric> {
    inputs: TensorList<T>,
    // TODO: remove the need to take ownership here
    /// signature: jvp(inputs, jacobians) -> jvp = jacobians @ J(inputs)
    jacobian_vector_product: fn(TensorList<T>, TensorList<T>) -> TensorList<T>,
    debug_info: String,
}

impl<T: Numeric> Derivative<T> {
    pub fn new(
        inputs: TensorList<T>,
        jacobian_vector_product: fn(TensorList<T>, TensorList<T>) -> TensorList<T>,
        debug_info: String,
    ) -> Derivative<T> {
        Derivative {
            inputs,
            jacobian_vector_product,
            debug_info,
        }
    }

    /// Computes the product outer_grads @ JVP(self.input)
    /// Recursively expanding the JVP of the input with the chain rule
    /// e.g. for f(g(h(x))), jvp(1.0) = jvp(1.0 * f'(g(h(x))))
    pub fn compute_jvp(&self, outer_grads: Vec<RcTensor<T>>) {
        // f(g(h(x))) how do i set x.grad if we are now computing f'
        // grad = f'(g(hx)) g'(h(x)) h'(x)
        // f(g(h(x), z)) how do i set x.grad if we are now computing f'
        let self_grads = (self.jacobian_vector_product)(self.inputs.clone(), outer_grads);
        for (grad, input) in self_grads.iter().zip(self.inputs.iter()) {
            dbg!(&grad, &input, &self.debug_info);
            debug_assert_eq!(
                grad.count(),
                input.count(),
                "grad and input must have the same number of elements"
            );
            let shaped_grad = RcTensor::new(grad.0.array.clone(), input.shape().to_vec());
            input.update_grad(shaped_grad);
            if let Some(d) = input.grad_fn.as_ref() {
                d.compute_jvp(vec![grad.clone()]);
            }
        }
    }
}

/// When used for making a jacobian_vector_product function
/// left ought to be the gradient passed in and right is the computed jacobian
pub(in crate::tensor) fn jvp_from_full_jacobians<T: Numeric>(
    left: RcTensor<T>,
    right: RcTensor<T>,
) -> RcTensor<T> {
    if right.is_scalar()
        || right.shape().iter().product::<usize>() == 1
        || left.is_scalar()
        || left.shape().iter().product::<usize>() == 1
    {
        assert!(left.is_scalar() || left.shape().iter().product::<usize>() == 1);
        left * right
    } else {
        left.bmm(&right)
    }
}

pub fn ones<T: Numeric>(tensors: Vec<RcTensor<T>>, grads: Vec<RcTensor<T>>) -> Vec<RcTensor<T>> {
    assert!(tensors.len() == 1);
    assert!(grads.len() == 1);
    let length_1 = tensors[0].shape().iter().product::<usize>();
    let length_2 = grads[0].shape().iter().product::<usize>();
    let raw_tensor = RawTensor::new_with_filler(vec![length_2, length_1], T::one());
    dbg!("sum_backward", vec![RcTensor::from_raw(raw_tensor)]).1
}
