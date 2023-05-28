mod numeric;
mod tensor_like;
mod tensor_view;
mod utils;

use num::traits::real::Real;
pub use numeric::*;
pub use tensor_like::*;
pub use tensor_view::*;
pub use utils::*;

use itertools::{EitherOrBoth::*, Itertools};
use std::cell::{Ref, RefCell};
use std::cmp::{max, PartialEq};
use std::convert::From;
use std::ops::{Add, Index, Mul};
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
pub struct SliceRange {
    /// inclusive
    start: usize,
    /// exclusive
    end: usize, // TODO: figure out how to make this optional
    skip: usize, // TODO: add this
}

impl SliceRange {
    pub fn new(start: usize, end: usize) -> SliceRange {
        assert!(start <= end);
        SliceRange {
            start,
            end,
            skip: 0,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct UserTensor<T: Numeric>(Rc<RefCell<Tensor<T>>>);

impl<T> TensorLike for UserTensor<T>
where
    T: Numeric,
{
    type Elem = T;
    type ShapeReturn<'a> = Ref<'a, Vec<usize>> where Self: 'a;

    fn shape(&self) -> Self::ShapeReturn<'_> {
        Ref::map((*(self.0)).borrow(), |x| x.shape())
    }

    fn sum(&self) -> Self::Elem {
        todo!()
    }

    fn tensor(&self) -> &Tensor<Self::Elem> {
        todo!()
    }

    fn to_tensor(&self) -> Tensor<Self::Elem> {
        (*(self.0)).borrow().clone()
    }
}

// DEBUG: disabling test till more features are in
// #[test]
// fn test_user_tensor_multiplication() {
//     let v = vec![0, 1, 2, 3];
//     let matrix = UserTensor(Rc::new(RefCell::new(Tensor::new(v, vec![2, 2])))); // [[0,1],[2,3]]
//     let shape = vec![2, 1];
//     let e1 = Tensor::new(vec![0, 1], vec![2, 1]);
//
//     assert_eq!(matrix.bmm(&e1), Tensor::new(vec![1, 3], shape.clone()));
// }
//
/// The core `struct` in this library.
#[derive(Debug, PartialEq, Clone)]
pub struct Tensor<T>
where
    T: Numeric,
{
    array: Vec<T>, // later on, I will need unsafe code to replace this with a statically sized type
    shape: Vec<usize>, // TODO: convert to let this be a slice
    grad: Vec<T>,
    requires_grad: bool,
    parents: Vec<UserTensor<T>>,
}

impl<T> Default for Tensor<T>
where
    T: Numeric,
{
    fn default() -> Self {
        Tensor {
            requires_grad: false,
            shape: vec![],
            grad: vec![],
            array: vec![],
            parents: vec![],
        }
    }
}

impl<T> From<T> for Tensor<T>
where
    T: Numeric,
{
    fn from(value: T) -> Self {
        Tensor {
            array: vec![value],
            shape: vec![],
            ..Default::default()
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
        Tensor {
            array,
            shape,
            ..Default::default()
        }
    }
}

impl<T, U, const N: usize> From<[U; N]> for Tensor<T>
where
    T: Numeric,
    Tensor<T>: From<U>,
    U: Clone,
{
    fn from(value: [U; N]) -> Tensor<T> {
        From::from(value.to_vec())
    }
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
    T: Numeric + Real,
{
    fn abs(&self) -> Self {
        let mut result = Tensor::new_empty(self.shape().clone());
        for &elem in self.array.iter() {
            result.array.push(elem.abs())
        }
        result
    }
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
        }
    }
    pub fn new(array: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let mut len = 1;
        for dim in shape.iter() {
            len *= dim;
        }
        assert_eq!(len, array.len());
        Tensor {
            array,
            shape,
            ..Default::default()
        }
    }

    pub fn view(&self, shape: Vec<SliceRange>) -> TensorView<'_, T> {
        TensorView::new(self, shape)
    }

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

impl<T> Index<&Vec<usize>> for Tensor<T>
where
    T: Numeric,
{
    type Output = T;

    fn index(&self, index: &Vec<usize>) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<T> TensorLike for Tensor<T>
where
    T: Numeric,
{
    type Elem = T;
    type ShapeReturn<'a> = &'a Vec<usize> where Self : 'a ;
    fn shape(&self) -> Self::ShapeReturn<'_> {
        &self.shape
    }

    fn sum(&self) -> Self::Elem {
        self.array
            .iter()
            .fold(Self::Elem::zero(), |acc, x| acc + *x)
    }

    fn tensor(&self) -> &Self {
        self
    }

    fn to_tensor(&self) -> Self {
        self.clone()
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

impl<T, U> Add<&U> for &Tensor<T>
where
    T: Numeric,
    U: TensorLike<Elem = T>,
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
        let index_iter = IndexIterator::new(max_shape.clone());
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

impl<T, U> Mul<&U> for &Tensor<T>
where
    T: Numeric,
    U: TensorLike<Elem = T>,
{
    type Output = Tensor<T>;

    fn mul(self, right: &U) -> Tensor<T> {
        if self.shape.len() == 1 {
            return self.dot(right);
        }
        self.bmm(right)
    }
}
