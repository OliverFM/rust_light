use itertools::{EitherOrBoth::*, Itertools};
use num::traits::real::Real;

use std::cell::RefCell;
use std::cmp::{max, PartialEq};
use std::convert::From;
use std::ops::{Add, Deref, Index, Mul, Neg, Sub};

use super::autograd::{self, Derivative};
use super::numeric::*;
use super::rc_tensor::*;
use super::tensor_like::*;
use super::tensor_view::*;
use super::utils::*;

#[derive(Debug, PartialEq, Clone)]
pub struct SliceRange {
    /// inclusive
    pub(in crate::tensor) start: usize,
    /// exclusive
    pub(in crate::tensor) end: usize, // TODO: figure out how to make this optional
    pub(in crate::tensor) skip: usize, // TODO: add this
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

// DEBUG: disabling test till more features are in
// #[ignore]
// #[test]
// fn test_user_tensor_multiplication() {}
//
/// The core `struct` in this library.
#[derive(Debug, Clone)]
pub struct RawTensor<T>
where
    T: Numeric,
{
    // TODO: consider using const generics to switch to type: Box<[T; N]>
    pub(in crate::tensor) array: Vec<T>, // later on, I will need unsafe code to replace this with a statically sized type
    pub(in crate::tensor) shape: Vec<usize>, // TODO: convert to let this be a slice
    pub(in crate::tensor) grad: RefCell<Option<RcTensor<T>>>, // TODO: think about the best representation here
    pub(in crate::tensor) derivative: Option<Derivative<T>>,
}

impl<T: Numeric> PartialEq for RawTensor<T> {
    // TODO: Think if this should change
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }
        if self.array != other.array {
            return false;
        }
        true
    }
}

impl<T> Default for RawTensor<T>
where
    T: Numeric,
{
    fn default() -> Self {
        RawTensor {
            shape: vec![],
            grad: RefCell::new(None),
            array: vec![],
            derivative: None,
        }
    }
}

impl<T, U> From<Vec<U>> for RawTensor<T>
where
    T: Numeric,
    RawTensor<T>: From<U>,
{
    fn from(value: Vec<U>) -> RawTensor<T> {
        let tensors: Vec<_> = value.into_iter().map(RawTensor::from).collect();
        let (arrays, shapes): (Vec<_>, Vec<_>) =
            tensors.into_iter().map(|t| (t.array, t.shape)).unzip();
        let valid = shapes.iter().all(|shape| *shape == shapes[0]);
        assert!(valid); // TODO: get rid of the assertion, this method should never fail. Try TryFrom instead

        let array = arrays.into_iter().flat_map(|arr| arr.into_iter()).collect();
        let mut shape = vec![shapes.len()];
        shape.extend_from_slice(&shapes[0]); // TODO: make this more by chaining iterators before so that we have shapes[0] is a dummy value
        let shape = shape;
        RawTensor {
            array,
            shape,
            ..Default::default()
        }
    }
}

impl<T, U, const N: usize> From<[U; N]> for RawTensor<T>
where
    T: Numeric,
    RawTensor<T>: From<U>,
    U: Clone,
{
    fn from(value: [U; N]) -> RawTensor<T> {
        From::from(value.to_vec())
    }
}

impl<T> RawTensor<T>
where
    T: Numeric + Real,
{
    pub fn abs(&self) -> Self {
        let mut result = RawTensor::new_empty(self.shape().clone());
        for &elem in self.array.iter() {
            result.array.push(elem.abs())
        }
        result
    }
}

impl<T> RawTensor<T>
where
    T: Numeric,
{
    pub fn is_scalar(&self) -> bool {
        self.array.len() == 1 && self.shape.is_empty()
    }

    pub(in crate::tensor) fn get_global_index(
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

    pub(in crate::tensor) fn new_empty(shape: Vec<usize>) -> RawTensor<T> {
        let mut total = 1;
        for &dim in shape.iter() {
            total *= dim;
        }

        RawTensor {
            array: Vec::with_capacity(total),
            shape,
            ..Default::default()
        }
    }

    /// Note! This function will construct Scalars
    pub fn new_with_filler(shape: Vec<usize>, filler: T) -> RawTensor<T> {
        if shape.is_empty() {
            return RawTensor {
                array: vec![filler],
                shape,
                ..Default::default()
            };
        }
        let mut total = 1;
        for &dim in shape.iter() {
            total *= dim;
        }

        let mut tensor = RawTensor {
            array: Vec::with_capacity(total),
            shape,
            ..Default::default()
        };
        for _ in 0..total {
            tensor.array.push(filler);
        }
        tensor
    }

    pub fn scalar(scalar: T) -> RawTensor<T> {
        RawTensor {
            array: vec![scalar],
            shape: vec![],
            ..Default::default()
        }
    }
    pub fn new(array: Vec<T>, shape: Vec<usize>) -> RawTensor<T> {
        let mut len = 1;
        for dim in shape.iter() {
            len *= dim;
        }
        assert_eq!(len, array.len());
        RawTensor {
            array,
            shape,
            ..Default::default()
        }
    }

    // pub fn view(&self, shape: Vec<SliceRange>) -> TensorView<T> {
    //     let tensor = RcTensor::from_raw(self);
    //     TensorView::new(tensor, shape)
    // }

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
    /// let matrix = RcTensor::new(v, vec![2, 2]);
    /// let tensor = RcTensor::new((0..16).collect(), vec![2, 2, 1, 2, 2]);
    ///
    /// assert_eq!(matrix.get(&vec![0,1]),matrix.get(&vec![0,0,1]));
    /// assert_eq!(matrix.get(&vec![0,1]),matrix.get(&vec![0,4,0,1]));
    /// assert_eq!(tensor.get(&vec![0,0,0, 0,1]),
    /// tensor.get(&vec![0,0,10, 0,1])
    /// );
    /// ```
    fn get(&self, index: &Vec<usize>) -> Result<&T, String> {
        match self.get_global_index(index, None) {
            Ok(global_idx) => {
                // dbg!("self={}, index={}, global_idx={}", self, index, global_idx);
                Ok(&self.array[global_idx])
            }
            Err(e) => Err(e),
        }
    }

    pub(super) fn get_with_offset(
        &self,
        index: &Vec<usize>,
        offset: &Vec<SliceRange>,
    ) -> Result<&T, String> {
        match self.get_global_index(index, Some(offset)) {
            Ok(global_idx) => Ok(&self.array[global_idx]),
            Err(e) => Err(e),
        }
    }
}

impl<T> Index<&Vec<usize>> for RawTensor<T>
where
    T: Numeric,
{
    type Output = T;

    fn index(&self, index: &Vec<usize>) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<T> Neg for &RcTensor<T>
where
    T: Numeric + Neg<Output = T>,
{
    type Output = RcTensor<T>;
    fn neg(self) -> Self::Output {
        RcTensor::from_raw(self.0.neg())
    }
}

impl<T> Neg for &RawTensor<T>
where
    T: Numeric + Neg<Output = T>,
{
    type Output = RawTensor<T>;
    fn neg(self) -> Self::Output {
        let mut result = RawTensor::new_empty(self.shape.clone());
        for &v in self.array.iter() {
            result.array.push(-v);
        }
        result
    }
}

impl<T> TensorLikePrivate for RawTensor<T> where T: Numeric {}
impl<T> TensorLike for RawTensor<T>
where
    T: Numeric,
{
    type Elem = T;
    type ShapeReturn<'a> = &'a Vec<usize> where Self : 'a ;
    type TensorRef<'tensor> = &'tensor RawTensor<Self::Elem> where Self : 'tensor;
    type ResultTensorType<'a>= RawTensor<T> where Self: 'a; // &'tensor Tensor<Self::Elem> where Self : 'tensor;
    type SumType = Self;
    type GradType = RcTensor<T>;

    fn set_grad(&self, grad: Self::GradType) {
        *self.grad.borrow_mut() = Some(grad);
    }
    fn shape(&self) -> Self::ShapeReturn<'_> {
        &self.shape
    }

    fn sum(&self) -> Self::SumType {
        let v = self
            .array
            .iter()
            .fold(Self::Elem::zero(), |acc, x| acc + *x);
        let scalar = RawTensor::from(v);
        scalar.set_grad(RcTensor::new_with_filler(
            self.shape.clone(),
            Self::Elem::one(),
        ));
        scalar
    }

    fn tensor(&self) -> Self::TensorRef<'_> {
        self
    }

    fn to_tensor(&self) -> RcTensor<Self::Elem> {
        RcTensor::from_raw(self.clone())
    }
    fn slice(&self, offset: Vec<SliceRange>) -> TensorView<T> {
        TensorView::new(RcTensor::from_raw(self.clone()), offset)
    }
    fn get(&self, index: &Vec<usize>) -> Result<&Self::Elem, String> {
        self.get(index)
    }
    fn bmm<U>(&self, right: &U) -> Self::ResultTensorType<'_>
    where
        U: TensorLike<Elem = Self::Elem>,
    {
        self.bmm_raw(right)
    }
}

impl<T, U> Add<&U> for &RcTensor<T>
where
    T: Numeric,
    U: TensorLike<Elem = T>,
{
    type Output = RcTensor<T>;
    fn add(self, right: &U) -> Self::Output {
        let mut raw_tensor = self.deref().add(right);
        // TODO: set derivative struct for right too
        raw_tensor.derivative = Some(Derivative::new(vec![self.to_tensor()], autograd::ones));
        RcTensor::from_raw(raw_tensor)
    }
}

impl<T, U, V> Sub<&U> for &RcTensor<T>
where
    T: Numeric + Neg,
    U: TensorLike<Elem = T>,
    for<'a> &'a U: Neg<Output = V>,
    V: TensorLike<Elem = T>,
{
    type Output = RcTensor<T>;
    fn sub(self, right: &U) -> Self::Output {
        RcTensor::from_raw(self.0.sub(right))
    }
}

impl<T, U, V> Sub<&U> for &RawTensor<T>
where
    T: Numeric + Neg,
    U: TensorLike<Elem = T>,
    for<'a> &'a U: Neg<Output = V>,
    V: TensorLike<Elem = T>,
{
    type Output = RawTensor<T>;
    fn sub(self, right: &U) -> Self::Output {
        let negative = right.neg();
        self.add(&negative)
    }
}

impl<T, U> Add<&U> for &RawTensor<T>
where
    T: Numeric,
    U: TensorLike<Elem = T>,
{
    type Output = RawTensor<T>;

    /// ```
    /// # use rust_light::tensor::*;
    /// let v = vec![0, 1, 2, 3];
    /// let matrix = RcTensor::new(v, vec![2, 2]);
    /// let tensor = RcTensor::new((0..16).collect(), vec![2, 2, 1, 2, 2]);
    ///
    /// assert_eq!(matrix.get(&vec![0,1]),matrix.get(&vec![0,0,1]));
    /// assert_eq!(matrix.get(&vec![0,1]),matrix.get(&vec![0,4,0,1]));
    /// assert_eq!(tensor.get(&vec![0,0,0, 0,1]),
    /// tensor.get(&vec![0,0,10, 0,1])
    /// );
    /// ```
    fn add(self, right: &U) -> Self::Output {
        assert!(self.broadcastable(right.shape())); // TODO: figure out broadcasting
        let length = max(right.shape().len(), self.shape.len());
        let mut max_shape = Vec::with_capacity(length);

        // TODO: consider getting rid of itertools
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
        let mut result = RawTensor::new_with_filler(max_shape.clone(), T::zero());
        for idx in index_iter {
            let v = self[&idx] + *(*right).get(&idx).unwrap();
            if let Err(e) = result.set(&idx, v) {
                panic!("{}", e)
            }
        }
        result
    }
}

fn element_wise_multiplication<T, U1, V1, U2, V2>(left: U1, right: U2) -> RawTensor<T>
where
    T: Numeric,
    U1: Deref<Target = V1> + std::fmt::Debug + Clone,
    V1: TensorLike<Elem = T>,
    U2: Deref<Target = V2> + Clone + std::fmt::Debug,
    V2: TensorLike<Elem = T>,
{
    let left_shape_vec = left.shape().to_vec();
    assert!(left_shape_vec == right.shape().to_vec());
    let length = left.shape().iter().product();
    let mut array = Vec::with_capacity(length);
    dbg!("left={}, right={},", left.clone(), right.clone());
    for (x, y) in ElementIterator::new(left).zip(ElementIterator::new(right)) {
        array.push(x * y);
    }

    dbg!("array={}, shape={},", &array, &left_shape_vec,);
    let result = RawTensor::new(array, left_shape_vec);
    dbg!("result={:?}", &result);
    result
}

impl<T, U, V> Mul<U> for &RawTensor<T>
where
    T: Numeric,
    U: Deref<Target = V> + Clone + std::fmt::Debug,
    V: TensorLike<Elem = T>,
{
    type Output = RawTensor<T>;

    fn mul(self, right: U) -> RawTensor<T> {
        if self.shape().is_empty() {
            return right.left_scalar_multiplication(&self.array[0]);
        }
        if right.shape().len() == 0 {
            return self.right_scalar_multiplication(right.get_first_elem());
        }
        element_wise_multiplication(self, right)
    }
}

impl<T> From<T> for RawTensor<T>
where
    T: Numeric,
{
    fn from(value: T) -> Self {
        RawTensor {
            array: vec![value],
            shape: vec![],
            ..Default::default()
        }
    }
}
