use num::PrimInt;
use std::ops::{Add, Mul};

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct Scalar<T>
where
    T: Copy + Clone,
{
    value: T,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Tensor<T>
where
    T: Copy + Clone,
{
    array: Box<Vec<T>>, // later on, I will need unsafe code to replace this with a statically sized type
    shape: Vec<usize>,  // TODO: convert to let this be a slice
}

#[derive(Debug, PartialEq, Clone)]
pub struct TensorView<'a, T>
where
    T: Copy + Clone,
{
    tensor: &'a Tensor<T>,
    shape: Vec<usize>, // TODO: convert to let this be a slice
}

#[derive(Debug, PartialEq, Clone)]
pub struct FrozenTensorView<'a, T>
where
    T: Copy + Clone,
{
    tensor: &'a Tensor<T>,
    shape: &'a Vec<usize>, // TODO: convert to let this be a slice
}

impl<T> Tensor<T>
where
    T: PrimInt + Copy + Clone + Mul + Add,
{
    fn new_empty(shape: Vec<usize>) -> Tensor<T> {
        let mut total = 1;
        for &dim in shape.iter() {
            total *= dim;
        }

        Tensor {
            array: Box::new(Vec::with_capacity(total)),
            shape,
        }
    }
    pub fn new_with_filler(shape: Vec<usize>, filler: T) -> Tensor<T> {
        assert!(shape.len() >= 1);
        let mut total = 1;
        for &dim in shape.iter() {
            total *= dim;
        }

        let mut tensor = Tensor {
            array: Box::new(Vec::with_capacity(total)),
            shape,
        };
        for _ in 0..total {
            tensor.array.push(filler.clone());
        }
        tensor
    }

    pub fn new(array: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let mut len = 1;
        for (i, dim) in shape.iter().enumerate() {
            len *= dim;
        }
        assert_eq!(len, array.len());
        Tensor {
            array: Box::new(array),
            shape,
        }
    }
    // pub fn from_tensor(tensor: &'a Tensor<T>, shape: Vec<usize>) -> TensorView<'a, T> {
    // &FrozenTensorView { tensor, shape }
    // }
    pub fn view<'a>(&'a self, shape: Vec<usize>) -> TensorView<'a, T> {
        TensorView {
            tensor: &self,
            shape,
        }
    }
    pub fn to_view<'a>(&'a self) -> TensorView<'a, T> {
        TensorView {
            tensor: &self,
            shape: self.shape.clone(),
        }
    }
    pub fn freeze<'a>(&'a self) -> FrozenTensorView<'a, T> {
        FrozenTensorView {
            tensor: &self,
            shape: &self.shape,
        }
    }
}
impl<'a, T> TensorLike<'a, T> for Tensor<T>
where
    T: PrimInt + Copy + Clone + Mul + Add,
{
    fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    // fn get(&self, index: &Vec<usize>) -> Result<&T, String> {
    // self.freeze().get(index)
    // }

    fn tensor(&self) -> &Tensor<T> {
        &self
    }
}

// TODO: implement views
// a borrowed Tensor, but with a new shape.
impl<'a, T> TensorView<'a, T>
where
    T: PrimInt + Copy + Clone + Mul + Add,
{
    pub fn freeze<'b>(&'b self) -> FrozenTensorView<'b, T> {
        FrozenTensorView {
            tensor: self.tensor,
            shape: &self.shape,
        }
    }
}

impl<'a, T> FrozenTensorView<'a, T>
where
    T: PrimInt + Copy + Clone + Mul + Add,
{
    pub fn get_global(&self, index: usize) -> Option<&T> {
        if index >= self.tensor.array.len() {
            return None;
        }
        Some(&self.tensor.array[index])
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }
}
pub trait TensorLike<'a, T>
where
    T: PrimInt + Copy + Clone + Mul + Add,
{
    fn shape(&self) -> &Vec<usize>;
    fn tensor(&self) -> &Tensor<T>;
    fn get(&self, index: &Vec<usize>) -> Result<&T, String> {
        let tensor = self.tensor();
        let shape = self.shape();
        if index.len() != shape.len() {
            return Err(format!(
                "shape do not match, expected {} shape but got {} shape!",
                shape.len(),
                index.len(),
            ));
        }
        let mut global_idx = 0;
        let mut multiplier = 1;
        for (i, (dim, idx_dim)) in shape.iter().zip(index.iter()).enumerate().rev() {
            if dim <= idx_dim {
                return Err(format!(
                    "shape do not match -- &TensorView has dimension:\n{:?}\nindex is:\n{:?}\nthe {}th position is out-of-bounds!",
                    shape,
                    index,
                    i,
                ));
            }
            global_idx += idx_dim * multiplier;
            multiplier *= dim;
        }
        Ok(&tensor.array[global_idx])
    }

    fn bmm(&self, right: &dyn TensorLike<T>) -> Tensor<T> {
        // assert!(self.same_shape(right));
        assert!(2 <= self.shape().len() && self.shape().len() <= 3); // For now we can only do Batch matrix
        assert!(right.shape().len() == 2); // rhs must be a matrix
        assert!(self.shape()[self.shape().len() - 1] == right.shape()[right.shape().len() - 2]);

        let new_shape;
        if self.shape().len() == 2 {
            new_shape = vec![1, self.shape()[0], right.shape()[1]];
        } else {
            new_shape = vec![self.shape()[0], self.shape()[1], right.shape()[1]];
        }

        let mut result = Tensor::new_empty(new_shape);

        let mut self_index = self.shape().clone();
        let self_index_len = self_index.len();
        let mut right_index = right.shape().clone();
        for batch_idx in 0..result.shape[0] {
            if self.shape().len() == 3 {
                self_index[0] = batch_idx;
            }
            for i in 0..result.shape[1] {
                self_index[self_index_len - 2] = i;
                for j in 0..result.shape[2] {
                    right_index[1] = j;
                    let mut val = T::zero();
                    for k in 0..right.shape()[0] {
                        self_index[self_index_len - 1] = k;
                        right_index[0] = k;
                        val = val
                            + *self.get(&self_index).unwrap() * *right.get(&right_index).unwrap();
                    }
                    result.array.push(val);
                }
            }
        }
        if self.shape().len() == 2 {
            return Tensor {
                array: result.array,
                shape: result.shape[1..].to_vec(),
            };
        }
        result
    }

    fn dot(&self, other: &dyn TensorLike<T>) -> T {
        //! generalised dot product: returns to acculumulated sum of the elementwise product.
        assert!(self.same_shape(other));
        let mut result = T::zero();
        for i in 0..self.tensor().array.len() {
            result = result + self.tensor().array[i] * other.tensor().array[i];
        }
        result
    }

    fn same_shape(&self, other: &dyn TensorLike<T>) -> bool {
        self.shape() == other.shape()
    }
}

// impl<T> Mul for Tensor<T>
// where
// T: PrimInt + Copy + Clone + Mul + Add,
// {
// type Output = Tensor<T>;

// fn mul(self, right: &Tensor<T>) -> Tensor<T> {
// assert!(self.same_shape(right)); // TODO: add broadcasting
// if self.shape.len() == 1 {
// return self.dot(right);
// }
// return self.bmm(right);
// }
// }
