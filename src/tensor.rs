use itertools::{EitherOrBoth::*, Itertools};
use num::{One, Zero};
use std::cmp::{max, PartialEq};
use std::convert::From;
use std::ops::{Add, Index, Mul};
// use std::ops::{RangeBounds, RangeFrom, RangeFull};

pub trait Numeric: Zero + One + Copy + Clone + Mul + Add + PartialEq + std::fmt::Debug {}

#[derive(Debug, PartialEq, Clone)]
pub struct SliceRange {
    start: usize,
    end: usize,  // TODO: figure out how to make this optional
    skip: usize, // TODO: add this
}

impl SliceRange {
    pub fn new(start: usize, end: usize) -> SliceRange {
        SliceRange {
            start,
            end,
            skip: 0,
        }
    }
}

// https://stackoverflow.com/questions/42381185/specifying-generic-parameter-to-belong-to-a-small-set-of-types
macro_rules! numeric_impl {
    ($($t: ty),+) => {
        $(
            impl Numeric for $t {}
        )+
    }
}

numeric_impl!(usize, u8, u32, u64, u128, i8, i32, i64, i128, f32, f64);

/// The core `struct` in this library.
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
        assert!(valid); // TODO: get rid of the assertion, this method should never fail. Try TryFrom instead

        let array = arrays.into_iter().flat_map(|arr| arr.into_iter()).collect();
        let mut shape = vec![shapes.len()];
        shape.extend_from_slice(&shapes[0]); // TODO: make this more by chaining iterators before so that we have shapes[0] is a dummy value
        let shape = shape;
        Tensor { array, shape }
    }
}

// #[derive(Debug, PartialEq, Clone)]
#[derive(Debug, Clone)]
pub struct TensorView<'a, T>
where
    T: Numeric,
{
    // TODO: convert this to look at slices of tensors. e.g. tensor[..1]
    tensor: &'a Tensor<T>,
    shape: Vec<usize>,
    offset: Vec<SliceRange>,
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
    fn get_global_index(
        &self,
        index: &Vec<usize>,
        offset: Option<&Vec<SliceRange>>,
    ) -> Result<usize, String> {
        if index.len() < self.shape().len() {
            // TODO: allow this case as long as extra dims are 1.
            return Err(format!(
                "shapes do not match: self.shape={:?}, index={:?}
                Need index to be at least as long as shape.",
                self.shape, index,
            ));
        }
        let mut global_idx = 0;
        let mut multiplier = 1;
        // TODO: consider turning this into a fold operation.
        for (i, (&dim, &idx_dim)) in self.shape.iter().rev().zip(index.iter().rev()).enumerate() {
            // Fix the indexing.  We need to have all the reverses index may be shorter than shape.
            let i = self.shape.len() - i - 1;
            let shaped_dim;
            let shifted_idx;
            if dim == 1 {
                // we pick the 0th element during broadcasting
                continue;
            }
            if let Some(range_vec) = offset {
                // println!("setting offset: {:?}", range_vec.clone());
                shaped_dim = range_vec[i].end - range_vec[i].start;
                shifted_idx = range_vec[i].start + idx_dim;
            } else {
                shaped_dim = dim;
                shifted_idx = idx_dim;
            }
            if shaped_dim <= idx_dim {
                return Err(format!(
                    "Trying to index too far into the view! -- &TensorView has dimension:
                    {:?}
                    Offest{:?}
                    index is:
                    {:?}
                    the {}th position is out-of-bounds!
                    shaped_dim={shaped_dim}, shifted_idx={shifted_idx}, dim={dim}, idx_dim={idx_dim}",
                    self.shape,
                    offset,
                    index,
                    i,
                ));
            }

            if dim <= shifted_idx {
                return Err(format!(
                    "shape do not match -- &TensorView has dimension:
                    {:?}
                    Offest{:?}
                    index is:
                    {:?}
                    the {}th position is out-of-bounds!
                    shaped_dim={shaped_dim}, shifted_idx={shifted_idx}, dim={dim}, idx_dim={idx_dim}",
                    self.shape,
                    offset,
                    index,
                    i,
                ));
            }
            global_idx += shifted_idx * multiplier;
            multiplier *= dim;
        }
        Ok(global_idx)
    }
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

    pub fn view(&self, shape: Vec<SliceRange>) -> TensorView<'_, T> {
        TensorView {
            tensor: self,
            offset: shape,
            shape: self.shape.clone(), // TODO: fix this
        }
    }

    // TODO: deprecate
    // pub fn to_view(&self) -> TensorView<'_, T> {
    // TensorView {
    // tensor: self,
    // shape: self.shape.clone(),
    // offset: self.shape.clone(),
    // }
    // }

    pub fn freeze(&self) -> FrozenTensorView<'_, T> {
        FrozenTensorView {
            tensor: self,
            shape: &self.shape,
        }
    }

    fn set(&mut self, index: &Vec<usize>, value: T) -> Result<(), String> {
        match self.get_global_index(index, None) {
            Ok(global_idx) => {
                self.array[global_idx] = value;
                Ok(())
            }
            Err(e) => Err(e),
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
        match self.get_global_index(index, None) {
            Ok(global_idx) => Ok(&self.array[global_idx]),
            Err(e) => Err(e),
        }
    }

    fn get_with_offset(&self, index: &Vec<usize>, offset: &Vec<SliceRange>) -> Result<&T, String> {
        match self.get_global_index(index, Some(offset)) {
            Ok(global_idx) => Ok(&self.array[global_idx]),
            Err(e) => Err(e),
        }
    }
}

impl<'a, T> Index<&Vec<usize>> for TensorView<'a, T>
where
    T: Numeric,
{
    type Output = T;

    fn index(&self, index: &Vec<usize>) -> &Self::Output {
        self.tensor.get_with_offset(index, &self.offset).unwrap()
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

impl<'a, T> TensorLike<'a> for Tensor<T>
where
    T: Numeric,
{
    type Elem = T;
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

impl<'a, T> TensorLike<'a> for TensorView<'a, T>
where
    T: Numeric,
{
    type Elem = T;
    fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    fn tensor(&self) -> &Tensor<T> {
        self.tensor
    }

    fn to_tensor(&self) -> Tensor<T> {
        // self.tensor.clone()
        todo!();
    }

    fn get(&self, index: &Vec<usize>) -> Result<&T, String> {
        let idx = self
            .tensor
            .get_global_index(index, Some(&self.offset))
            .unwrap();
        Ok(&self.tensor.array[idx])
    }
}

impl<'a, T> FrozenTensorView<'a, T>
where
    T: Numeric,
{
    pub fn to_tensor(&self) -> Tensor<T> {
        (*self.tensor).clone()
    }

    pub fn shape(&self) -> &Vec<usize> {
        self.shape
    }
}

pub trait TensorLike<'a> {
    type Elem: Numeric;
    fn get(&self, index: &Vec<usize>) -> Result<&Self::Elem, String> {
        (*self.tensor()).get(index)
    }

    fn shape(&self) -> &Vec<usize>;

    fn tensor(&self) -> &Tensor<Self::Elem>;

    fn to_tensor(&self) -> Tensor<Self::Elem>;

    fn slice(&self, offset: Vec<SliceRange>) -> TensorView<Self::Elem> {
        let mut shape = Vec::with_capacity(offset.len() + 1);
        for slice_range in offset.iter() {
            // NOTE: assuming that all intervals are half open, for now.
            // TODO: add a better parser once I have generalised this
            shape.push(slice_range.end - slice_range.start);
        }
        TensorView {
            tensor: self.tensor(),
            shape,
            offset,
        }
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
                                           // println!(
                                           // "self.shape()={:?}, right.shape()={:?}",
                                           // self.shape(),
                                           // right.shape()
                                           // );
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
    // fn broadcast(&self, &new_shape: &Vec<usize>) -> Result<FrozenTensorView<T>, Err(String)> {
    // let shape = self.shape();
    // for
    // FrozenTensorView {
    // tensor: self.tensor(),
    // shape: new_shape,
    // }
    // }

    fn iter_elements<U>(&self) -> IndexIterator {
        IndexIterator {
            index: vec![0; self.shape().len()],
            dimensions: self.shape().clone(),
            carry: Default::default(),
        }
    }
}

impl<'a, T, U> PartialEq<U> for TensorView<'a, T>
where
    T: Numeric,
    U: TensorLike<'a, Elem = T>,
{
    fn eq(&self, other: &U) -> bool {
        // println!(
        // "self.shape={:?}, other.shape={:?}",
        // self.shape.clone(),
        // other.shape().clone()
        // );
        if other.shape() != &self.shape {
            return false;
        }
        let index_iter = IndexIterator {
            index: vec![0; self.shape.len()],
            dimensions: self.shape.clone(),
            carry: Default::default(),
        };

        for idx in index_iter {
            // println!(
            // "self[{:?}]={:?}, other[{:?}]={:?}",
            // idx.clone(),
            // self.get(&idx),
            // idx.clone(),
            // other.get(&idx)
            // );
            if self.get(&idx) != other.get(&idx) {
                return false;
            }
        }
        true
    }
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
    U: for<'b> TensorLike<'b, Elem = T>,
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
pub struct IndexIterator {
    index: Vec<usize>,
    dimensions: Vec<usize>,
    carry: usize,
}

impl IndexIterator {
    pub fn new(dimensions: Vec<usize>) -> IndexIterator {
        IndexIterator {
            index: vec![0; dimensions.len()],
            dimensions,
            carry: Default::default(),
        }
    }
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

impl<T, U> Mul<&U> for &Tensor<T>
where
    T: Numeric,
    U: for<'b> TensorLike<'b, Elem = T>,
{
    type Output = Tensor<T>;

    fn mul(self, right: &U) -> Tensor<T> {
        if self.shape.len() == 1 {
            return self.dot(right);
        }
        self.bmm(right)
    }
}
