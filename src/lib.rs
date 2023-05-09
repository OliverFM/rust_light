use num::PrimInt;
use std::ops::{Add, Mul};

#[derive(Debug)]
pub struct Scalar<T>
where
    T: Copy + Clone,
{
    value: T,
}
#[derive(Debug, PartialEq)]
pub struct Tensor<T>
where
    T: Copy + Clone,
{
    array: Box<Vec<T>>, // later on, I will need unsafe code to replace this with a statically sized type
    shape: Vec<usize>,  // TODO: convert to let this be a slice
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

// TODO: implement views
// a borrowed Tensor, but with a new shape.
impl<T> Tensor<T>
where
    T: PrimInt + Copy + Clone + Mul + Add,
{
    pub fn bmm(&self, right: &Tensor<T>) -> Tensor<T> {
        // assert!(self.same_shape(right));
        assert!(2 <= self.shape.len() && self.shape.len() <= 3); // For now we can only do Batch matrix
        assert!(right.shape.len() == 2); // rhs must be a matrix
        assert!(self.shape[self.shape.len() - 1] == right.shape[right.shape.len() - 2]);

        let new_shape;
        if self.shape.len() == 2 {
            new_shape = vec![1, self.shape[0], right.shape[1]];
        } else {
            new_shape = vec![self.shape[0], self.shape[1], right.shape[1]];
        }

        let mut result = Tensor::new_empty(new_shape);

        let mut self_index = self.shape.clone();
        let self_index_len = self_index.len();
        let mut right_index = right.shape.clone();
        for batch_idx in 0..result.shape[0] {
            if self.shape.len() == 3 {
                self_index[0] = batch_idx;
            }
            for i in 0..result.shape[1] {
                self_index[self_index_len - 2] = i;
                for j in 0..result.shape[2] {
                    right_index[1] = j;
                    let mut val = T::zero();
                    for k in 0..right.shape[0] {
                        self_index[self_index_len - 1] = k;
                        right_index[0] = k;
                        val =
                            val + self.get(&self_index).unwrap() * right.get(&right_index).unwrap();
                    }
                    result.array.push(val);
                }
            }
        }
        if self.shape.len() == 2 {
            return Tensor {
                array: result.array,
                shape: result.shape[1..].to_vec(),
            };
        }
        result
    }

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
        Tensor {
            array: Box::new(array),
            shape,
        }
    }

    pub fn get_global(&self, index: usize) -> Option<&T> {
        if index >= self.array.len() {
            return None;
        }
        Some(&self.array[index])
    }

    pub fn get(&self, index: &Vec<usize>) -> Result<T, String> {
        if index.len() != self.shape.len() {
            return Err(format!(
                "shape do not match, expected {} shape but got {} shape!",
                self.shape.len(),
                index.len(),
            ));
        }
        let mut global_idx = 0;
        let mut multiplier = 1;
        for (i, (dim, idx_dim)) in self.shape.iter().zip(index.iter()).enumerate().rev() {
            if dim <= idx_dim {
                return Err(format!(
                    "shape do not match -- Tensor has dimension:\n{:?}\nindex is:\n{:?}\nthe {}th position is out-of-bounds!",
                    self.shape,
                    index,
                    i,
                ));
            }
            global_idx += idx_dim * multiplier;
            multiplier *= dim;
        }
        Ok(self.array[global_idx])
    }

    pub fn dot(&self, other: &Tensor<T>) -> T {
        //! generalised dot product: returns to acculumulated sum of the elementwise product.
        assert!(self.same_shape(other));
        let mut result = T::zero();
        for i in 0..self.array.len() {
            result = result + self.array[i] * other.array[i];
        }
        result
    }

    pub fn same_shape(&self, other: &Tensor<T>) -> bool {
        self.shape == other.shape
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_new_with_filler() {
        let vec = Tensor::new_with_filler(vec![4], 4);
        assert_eq!(vec.shape(), &vec![4]);
        assert_eq!(vec.get_global(2).unwrap(), &4);
    }

    #[test]
    fn test_get_2x2x2() {
        let matrix = Tensor::new(vec![0, 1, 2, 3, 4, 5, 6, 7], vec![2, 2, 2]);
        assert_eq!(matrix.get(&vec![0, 0, 0]).unwrap(), 0);
        assert_eq!(matrix.get(&vec![0, 1, 0]).unwrap(), 2);
        assert_eq!(matrix.get(&vec![1, 1, 1]).unwrap(), 7);
    }

    #[test]
    fn test_get_3x3x4() {
        let matrix = Tensor::new((0..(3 * 3 * 4)).collect(), vec![3, 3, 4]);
        assert_eq!(matrix.get(&vec![0, 0, 0]).unwrap(), 0);
        assert_eq!(matrix.get(&vec![2, 2, 3]).unwrap(), 3 * 3 * 4 - 1);
    }

    #[test]
    fn test_get_3x3() {
        let matrix = Tensor::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], vec![3, 3]);
        let mut prev = -1;
        for i in 0..3 {
            for j in 0..3 {
                let curr = matrix.get(&vec![i, j]).unwrap();
                println!(
                    "prev={prev}, matrix[{i}][{j}]={}",
                    matrix.get(&vec![i, j]).unwrap()
                );
                assert_eq!(prev + 1, curr);
                prev = curr;
            }
        }
        assert_eq!(matrix.get(&vec![0, 0]).unwrap(), 0);
        assert_eq!(matrix.get(&vec![0, 1]).unwrap(), 1);
        assert_eq!(matrix.get(&vec![1, 0]).unwrap(), 3);
        assert_eq!(matrix.get(&vec![2, 2]).unwrap(), 8);
    }

    #[test]
    fn test_dot() {
        let v = vec![0, 1, 2];
        let vec = Tensor::new(v, vec![3]);
        assert_eq!(vec.dot(&vec), 5);
    }
    #[test]
    fn test_creation() {
        let v = vec![0, 1, 2, 3];
        let matrix = Tensor::new(v, vec![2, 2]);

        format!("Matrix: \n{:?}", matrix);

        assert_eq!(matrix.get_global(0).unwrap(), &0);
    }

    #[test]
    fn test_bmm_2x2() {
        let v = vec![0, 1, 2, 3];
        let matrix = Tensor::new(v, vec![2, 2]);
        let shape = vec![2, 1];
        let e1 = Tensor::new(vec![0, 1], vec![2, 1]);
        let e2 = Tensor::new(vec![1, 0], vec![2, 1]);
        let diag = Tensor::new(vec![1, 1], vec![2, 1]);

        let r = matrix.bmm(&diag);
        assert_eq!(r.shape(), &shape);
        assert_eq!(r, Tensor::new(vec![1, 5], shape.clone()));
        assert_eq!(matrix.bmm(&e1), Tensor::new(vec![1, 3], shape.clone()));
        assert_eq!(matrix.bmm(&e2), Tensor::new(vec![0, 2], shape.clone()));
    }
}
