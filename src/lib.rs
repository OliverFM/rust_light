/// Tensor Processing for Machine Learning and Scientific computing in rust.
///
/// The first goal of `rust_light` is to make it easier to train machine learning
/// models in pure rust.
/// The other key goal is to provide a learning experience. This library is
/// only ~3500 lines long, so it gives a fairly good example of a minimal working
/// implementation of autograd and tensors in rust.
///
/// The `tensor` module contains the core functionality for creating and
/// processing tensors. This includes autograd.
///
pub mod tensor;

/// The basic building blocks for making Neural Networks.
pub mod nn;

/// Optimisers!  Currently this only supports SGD.
pub mod optim;
