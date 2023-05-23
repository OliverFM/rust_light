use super::numeric::*;
use super::utils::{IndexIterator};
// use super::utils::*;
use crate::tensor::{SliceRange, Tensor, TensorView};

pub trait TensorLike<'a> {
    type Elem: Numeric;
    fn get(&self, index: &Vec<usize>) -> Result<&Self::Elem, String> {
        (*self.tensor()).get(index)
    }

    fn shape(&self) -> &Vec<usize>;

    // fn sum(&self) -> T;
    // fn sum(&self) -> Self::Elem {
    // self.iter_elements().reduce(|elem, acc| elem + acc)
    // }

    fn tensor(&self) -> &Tensor<Self::Elem>;

    fn to_tensor(&self) -> Tensor<Self::Elem>;

    // fn iter_elements<I: Iterator<Item = Self::Elem>>(&self) -> I;

    fn slice(&self, offset: Vec<SliceRange>) -> TensorView<Self::Elem> {
        TensorView::new(self.tensor(), offset)
    }

    fn left_scalar_multiplication(&self, &scalar: &Self::Elem) -> Tensor<Self::Elem> {
        let mut result = Tensor::new_empty((*self.shape()).clone());
        for &elem in self.tensor().array.iter() {
            result.array.push(scalar * elem);
        }
        result
    }

    fn right_scalar_multiplication(&self, &scalar: &Self::Elem) -> Tensor<Self::Elem> {
        let mut result = Tensor::new_empty((*self.shape()).clone());
        for &elem in self.tensor().array.iter() {
            result.array.push(elem * scalar);
        }
        result
    }

    fn dot<U>(&self, other: &U) -> Tensor<Self::Elem>
    where
        U: for<'b> TensorLike<'b, Elem = Self::Elem>,
    {
        //! generalised dot product: returns to acculumulated sum of the elementwise product.
        assert!(self.same_shape(other));
        let mut result = Self::Elem::zero();
        for i in 0..self.tensor().array.len() {
            result = result + self.tensor().array[i] * other.tensor().array[i];
        }
        Tensor {
            array: vec![result],
            shape: vec![1],
        }
    }

    /// A Naive batch matrix multiply operation
    ///
    /// ```
    /// # use rust_light::tensor::*;
    /// let v = vec![0, 1, 2, 3];
    /// let matrix = Tensor::new(v, vec![2, 2]);
    /// let shape = vec![2, 1];
    /// let e1 = Tensor::new(vec![0, 1], vec![2, 1]);
    /// let e2 = Tensor::new(vec![1, 0], vec![2, 1]);
    /// let diag = Tensor::new(vec![1, 1], vec![2, 1]);
    /// let r = matrix.bmm(&diag);
    ///
    /// assert_eq!(r.shape(), &shape);
    /// assert_eq!(r, Tensor::new(vec![1, 5], shape.clone()));
    /// assert_eq!(matrix.bmm(&e1), Tensor::new(vec![1, 3], shape.clone()));
    /// ```
    fn bmm<U>(&self, right: &U) -> Tensor<Self::Elem>
    where
        U: for<'b> TensorLike<'b, Elem = Self::Elem>,
    {
        assert!(2 <= self.shape().len() && self.shape().len() <= 3); // For now we can only do Batch matrix
        assert!(right.shape().len() == 2); // rhs must be a matrix
        assert!(self.shape()[self.shape().len() - 1] == right.shape()[right.shape().len() - 2]);
        let new_shape = if self.shape().len() == 2 {
            vec![1, self.shape()[0], right.shape()[1]]
        } else {
            vec![self.shape()[0], self.shape()[1], right.shape()[1]]
        };

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
                    let mut val = Self::Elem::zero();
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

    fn same_shape<U>(&self, other: &U) -> bool
    where
        U: for<'b> TensorLike<'b, Elem = Self::Elem>,
    {
        self.shape() == other.shape()
    }

    fn broadcastable(&self, new_shape: &[usize]) -> bool {
        // TODO: test this!
        for (&d1, &d2) in self
            .shape()
            .iter()
            .rev()
            .zip(new_shape.to_vec().iter().rev())
        {
            if d1 != d2 {
                if d1 == 1 || d2 == 1 {
                    continue;
                }
                return false;
            }
        }
        true
    }

    fn iter_indices(&self) -> IndexIterator {
        IndexIterator::new(self.shape().clone())
    }
}
