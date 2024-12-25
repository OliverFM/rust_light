# `rust_light`
A tensor processing library in rust.

## Current status
At the moment I have implemented:
- [x] Parallelised Matrix Multiplication
- [x] Autograd (in parallel)
- [x] SGD
- [x] Neural Network module
- [x] MLPs
- [x] Parallelised Jacobian Vector Product for all operations needed in an MLP
- [x] Script to train a toy NN.
- [ ] GPU Support
- [ ] Image Loading

## Goals
The main goal of this project is to get a simple tensor processing library working.
1. Get something that works.
2. Learn more about rust.
3. Pybind `rust_light`
4. Get `rust_light` to work on a GPU.

### Intended functionality:
1. implement a `Tensor` that supports multiplication and additions -- for scalars and matrices/tensors.
2. Enable full `numpy`-style [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).
3. stretch: add some kind of auto-grad.


## Contributing:
Contributions welcome!
If you'd like to contribute, start by opening an issue.
