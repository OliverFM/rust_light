use itertools::{EitherOrBoth::*, Itertools};
use num::{One, Zero};
use std::cmp::max;
use std::convert::From;
use std::ops::{Add, Index, Mul};

pub trait Numeric: Zero + One + Copy + Clone + Mul + Add + std::fmt::Debug {}

// https://stackoverflow.com/questions/42381185/specifying-generic-parameter-to-belong-to-a-small-set-of-types
macro_rules! numeric_impl {
    ($($t: ty),+) => {
        $(
            impl Numeric for $t {}
        )+
    }
}

numeric_impl!(usize, u8, u32, u64, u128, i8, i32, i64, i128, f32, f64);

#[derive(Debug, PartialEq, Clone)]
pub struct Tensor<T>
where
    T: Numeric,
{
    array: Vec<T>, // later on, I will need unsafe code to replace this with a statically sized type
    shape: Vec<usize>, // TODO: convert to let this be a slice
}

impl<T> From<T> for Tensor<T>
where
    T: Numeric,
{
    fn from(value: T) -> Self {
        Tensor {
            array: vec![value],
            shape: vec![],
        }
    }
}

// impl<T, U, const N: usize> From<[U; N]> for Tensor<T>
// where
// T: Numeric,
// Tensor<T>: From<U>,
// {
// fn from(value: [U; N]) -> Self {
// Tensor::from::<Vec<_>>(value.to_vec())
// }
// }

impl<T, U> From<Vec<U>> for Tensor<T>
where
    T: Numeric,
    Tensor<T>: From<U>,
{
    fn from(value: Vec<U>) -> Tensor<T> {
        let tensors: Vec<_> = value.into_iter().map(Tensor::from).collect();
        let (arrays, shapes): (Vec<_>, Vec<_>) =
            tensors.into_iter().map(|t| (t.array, t.shape)).unzip();
        let valid = shapes.iter().all(|shape| *shape == shapes[0]);
        assert!(valid);

        let array = arrays.into_iter().flat_map(|arr| arr.into_iter()).collect();
        let mut shape = vec![shapes.len()];
        shape.extend_from_slice(&shapes[0]); // TODO: make this more by chaining iterators before so that we have shapes[0] is a dummy value
        let shape = shape;
        Tensor { array, shape }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct TensorView<'a, T>
where
    T: Numeric,
{
    // TODO: convert this to look at slices of tensors. e.g. tensor[..1]
    tensor: &'a Tensor<T>,
    shape: Vec<usize>, // TODO: convert to let this be a slice
}

#[derive(Debug, PartialEq, Clone)]
pub struct FrozenTensorView<'a, T>
where
    T: Numeric,
{
    tensor: &'a Tensor<T>,
    shape: &'a Vec<usize>, // TODO: convert to let this be a slice
}

impl<T> Tensor<T>
where
    T: Numeric,
{
    fn new_empty(shape: Vec<usize>) -> Tensor<T> {
        let mut total = 1;
        for &dim in shape.iter() {
            total *= dim;
        }

        Tensor {
            array: Vec::with_capacity(total),
            shape,
        }
    }
    pub fn new_with_filler(shape: Vec<usize>, filler: T) -> Tensor<T> {
        assert!(!shape.is_empty());
        let mut total = 1;
        for &dim in shape.iter() {
            total *= dim;
        }

        let mut tensor = Tensor {
            array: Vec::with_capacity(total),
            shape,
        };
        for _ in 0..total {
            tensor.array.push(filler);
        }
        tensor
    }

    pub fn scalar(scalar: T) -> Tensor<T> {
        Tensor {
            array: vec![scalar],
            shape: vec![1],
        }
    }
    pub fn new(array: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let mut len = 1;
        for dim in shape.iter() {
            len *= dim;
        }
        assert_eq!(len, array.len());
        Tensor { array, shape }
    }
    // pub fn from_tensor(tensor: &'a Tensor<T>, shape: Vec<usize>) -> TensorView<'a, T> {
    // &FrozenTensorView { tensor, shape }
    // }
    pub fn view(&self, shape: Vec<usize>) -> TensorView<'_, T> {
        TensorView {
            tensor: self,
            shape,
        }
    }
    pub fn to_view(&self) -> TensorView<'_, T> {
        TensorView {
            tensor: self,
            shape: self.shape.clone(),
        }
    }
    pub fn freeze(&self) -> FrozenTensorView<'_, T> {
        FrozenTensorView {
            tensor: self,
            shape: &self.shape,
        }
    }

    /// ```
    /// # use rust_light::tensor::*;
    /// let v = vec![0, 1, 2, 3];
    /// let matrix = Tensor::new(v, vec![2, 2]);
    /// let tensor = Tensor::new((0..16).collect(), vec![2, 2, 1, 2, 2]);
    ///
    /// assert_eq!(matrix.get(&vec![0,1]),matrix.get(&vec![0,0,1]));
    /// assert_eq!(matrix.get(&vec![0,1]),matrix.get(&vec![0,4,0,1]));
    /// assert_eq!(tensor.get(&vec![0,0,0, 0,1]),
    /// tensor.get(&vec![0,0,10, 0,1])
    /// );
    /// ```
    fn get(&self, index: &Vec<usize>) -> Result<&T, String> {
        match self.get_global_index(index) {
            Ok(global_idx) => Ok(&self.array[global_idx]),
            Err(e) => Err(e),
        }
    }

    fn set(&mut self, index: &Vec<usize>, value: T) -> Result<(), String> {
        match self.get_global_index(index) {
            Ok(global_idx) => {
                self.array[global_idx] = value;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn get_global_index(&self, index: &Vec<usize>) -> Result<usize, String> {
        if index.len() < self.shape().len() {
            return Err(format!(
                "shapes do not match: self.shape={:?}, index={:?}
                Need index to be at least as long as shape.",
                self.shape, index,
            ));
        }
        let mut global_idx = 0;
        let mut multiplier = 1;
        for (i, (&dim, &idx_dim)) in self.shape.iter().rev().zip(index.iter().rev()).enumerate() {
            if dim == 1 {
                // we pick the 0th element during broadcasting
                continue;
            }
            if dim <= idx_dim {
                return Err(format!(
                    "shape do not match -- &TensorView has dimension:\n{:?}\nindex is:\n{:?}\nthe {}th position is out-of-bounds!",
                    self.shape,
                    index,
                    i,
                ));
            }
            global_idx += idx_dim * multiplier;
            multiplier *= dim;
        }
        Ok(global_idx)
    }
}

impl<T> Index<&Vec<usize>> for Tensor<T>
where
    T: Numeric,
{
    type Output = T;

    fn index(&self, index: &Vec<usize>) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<'a, T> TensorLike<'a, T> for Tensor<T>
where
    T: Numeric,
{
    fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    // fn get(&self, index: &Vec<usize>) -> Result<&T, String> {
    // self.freeze().get(index)
    // }

    fn tensor(&self) -> &Tensor<T> {
        self
    }
    fn to_tensor(&self) -> Tensor<T> {
        self.clone()
    }
}

// TODO: implement views
// a borrowed Tensor, but with a new shape.
impl<'a, T> TensorView<'a, T>
where
    T: Numeric,
{
    pub fn to_tensor(&self) -> Tensor<T> {
        (*self.tensor).clone()
    }

    pub fn freeze(&self) -> FrozenTensorView<'_, T> {
        FrozenTensorView {
            tensor: self.tensor,
            shape: &self.shape,
        }
    }
}

impl<'a, T> FrozenTensorView<'a, T>
where
    T: Numeric,
{
    pub fn to_tensor(&self) -> Tensor<T> {
        (*self.tensor).clone()
    }

    pub fn get_global(&self, index: usize) -> Option<&T> {
        if index >= self.tensor.array.len() {
            return None;
        }
        Some(&self.tensor.array[index])
    }

    pub fn shape(&self) -> &Vec<usize> {
        self.shape
    }
}

pub trait TensorLike<'a, T>
where
    T: Numeric,
{
    fn get(&self, index: &Vec<usize>) -> Result<&T, String> {
        (*self.tensor()).get(index)
    }

    fn shape(&self) -> &Vec<usize>;

    fn tensor(&self) -> &Tensor<T>;

    fn to_tensor(&self) -> Tensor<T>;

    fn left_scalar_multiplication(&self, &scalar: &T) -> Tensor<T> {
        let mut result = Tensor::new_empty((*self.shape()).clone());
        for &elem in self.tensor().array.iter() {
            result.array.push(scalar * elem);
        }
        result
    }

    fn right_scalar_multiplication(&self, &scalar: &T) -> Tensor<T> {
        let mut result = Tensor::new_empty((*self.shape()).clone());
        for &elem in self.tensor().array.iter() {
            result.array.push(elem * scalar);
        }
        result
    }

    fn dot<U>(&self, other: &U) -> Tensor<T>
    where
        U: for<'b> TensorLike<'b, T>,
    {
        //! generalised dot product: returns to acculumulated sum of the elementwise product.
        assert!(self.same_shape(other));
        let mut result = T::zero();
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
    fn bmm<U>(&self, right: &U) -> Tensor<T>
    where
        U: for<'b> TensorLike<'b, T>,
    {
        assert!(2 <= self.shape().len() && self.shape().len() <= 3); // For now we can only do Batch matrix
        assert!(right.shape().len() == 2); // rhs must be a matrix
        println!(
            "self.shape()={:?}, right.shape()={:?}",
            self.shape(),
            right.shape()
        );
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

    // Replaced by add
    // fn add_no_broadcast<U>(&self, right: &U) -> Tensor<T>
    // where
    // U: for<'b> TensorLike<'b, T>,
    // {
    // assert!(self.same_shape(right));
    // let mut result = Tensor::new_empty((*self.shape()).clone());
    // for (&x, &y) in self.tensor().array.iter().zip(right.tensor().array.iter()) {
    // result.array.push(x + y);
    // }
    // result
    // }

    fn same_shape<U>(&self, other: &U) -> bool
    where
        U: for<'b> TensorLike<'b, T>,
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
    // fn broadcast(&self, &new_shape: &Vec<usize>) -> Result<FrozenTensorView<T>, Err(String)> {
    // let shape = self.shape();
    // for
    // FrozenTensorView {
    // tensor: self.tensor(),
    // shape: new_shape,
    // }
    // }
}

// TODO: figure out how to get scalar multiplication with correct typing
// impl<T> Add<T> for &Tensor<T>
// where
// T: Numeric,
// {
// type Output = Tensor<T>;
// fn add(self, right: T) -> Tensor<T> {
// Tensor::new(
// self.array.iter().map(|x| *x + right).collect(),
// self.shape.clone(),
// )
// }
// }

impl<T, U> Add<&U> for &Tensor<T>
where
    T: Numeric,
    U: for<'b> TensorLike<'b, T>,
{
    type Output = Tensor<T>;

    /// ```
    /// # use rust_light::tensor::*;
    /// let v = vec![0, 1, 2, 3];
    /// let matrix = Tensor::new(v, vec![2, 2]);
    /// let tensor = Tensor::new((0..16).collect(), vec![2, 2, 1, 2, 2]);
    ///
    /// assert_eq!(matrix.get(&vec![0,1]),matrix.get(&vec![0,0,1]));
    /// assert_eq!(matrix.get(&vec![0,1]),matrix.get(&vec![0,4,0,1]));
    /// assert_eq!(tensor.get(&vec![0,0,0, 0,1]),
    /// tensor.get(&vec![0,0,10, 0,1])
    /// );
    /// ```
    fn add(self, right: &U) -> Tensor<T> {
        assert!(self.broadcastable(right.shape())); // TODO: figure out broadcasting
        let length = max(right.shape().len(), self.shape.len());
        let mut max_shape = Vec::with_capacity(length);

        for pair in self
            .shape
            .iter()
            .rev()
            .zip_longest(right.tensor().shape().iter().rev())
            .rev()
        {
            let dim = match pair {
                Both(&l, &r) => max(l, r),
                Left(&l) => l,
                Right(&r) => r,
            };
            max_shape.push(dim);
        }
        let index_vec = vec![0; length];
        let index_iter = IndexIterator {
            index: index_vec,
            dimensions: max_shape.clone(),
            carry: Default::default(),
        };
        let mut result = Tensor::new_with_filler(max_shape.clone(), T::zero());
        for idx in index_iter {
            let v = self[&idx] + *(*right).get(&idx).unwrap();
            if let Err(e) = result.set(&idx, v) {
                panic!("{}", e)
            }
        }
        result
    }
}

// TODO: figure out how to make this hold references with two lifetimes, and get the iterator to return a reference
#[derive(Default)]
struct IndexIterator {
    index: Vec<usize>,
    dimensions: Vec<usize>,
    carry: usize,
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        for i in (0..self.index.len()).rev() {
            let v = self.index[i];
            let dim = self.dimensions[i];
            while v < dim - 1 && self.carry > 0 {
                self.index[i] = v + 1;
                self.carry -= 1;
            }
            if self.carry == 0 {
                reset_trailing_indices(&mut self.index, i);
                self.carry = 1; // for next iteration
                return Some(self.index.clone());
            }
        }
        None
    }
}

fn reset_trailing_indices(index: &mut [usize], position: usize) {
    for idx in index.iter_mut().skip(position + 1) {
        *idx = 0;
    }
}

#[test]
fn test_index_iterator() {
    let index_iter = IndexIterator {
        index: vec![0, 0, 0],
        dimensions: vec![2, 2, 2],
        carry: Default::default(),
    };
    assert_eq!(
        index_iter.collect::<Vec<_>>(),
        vec![
            [0, 0, 0].to_vec(),
            [0, 0, 1].to_vec(),
            [0, 1, 0].to_vec(),
            [0, 1, 1].to_vec(),
            [1, 0, 0].to_vec(),
            [1, 0, 1].to_vec(),
            [1, 1, 0].to_vec(),
            [1, 1, 1].to_vec(),
        ]
    );
}

impl<T, U> Mul<&U> for &Tensor<T>
where
    T: Numeric,
    U: for<'b> TensorLike<'b, T>,
{
    type Output = Tensor<T>;

    fn mul(self, right: &U) -> Tensor<T> {
        if self.shape.len() == 1 {
            return self.dot(right);
        }
        self.bmm(right)
    }
}
