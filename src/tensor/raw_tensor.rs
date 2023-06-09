use std::cmp::PartialEq;
use std::convert::From;
use std::{
    ops::{Add, Deref, Index, Mul, Neg, Sub},
    sync::RwLock,
};

use super::autograd::Derivative;
use super::functional as F;
use super::functional;
use super::numeric::*;
use super::rc_tensor::*;
use super::tensor_like::*;
use super::tensor_view::*;
use super::utils::*;
use super::IndexType;

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

/// The core `struct` in this library.
#[derive(Debug)]
pub struct RawTensor<T>
where
    T: Numeric,
{
    // TODO: consider using const generics to switch to type: Box<[T; N]>
    pub(in crate::tensor) array: Vec<T>, // later on, I will need unsafe code to replace this with a statically sized type
    pub(in crate::tensor) shape: Vec<usize>, // TODO: convert to let this be a slice
    pub(in crate::tensor) grad: RwLock<Option<RcTensor<T>>>, // TODO: think about the best representation here
    pub(in crate::tensor) grad_fn: Option<Derivative<T>>,
}

impl<T: Numeric> Clone for RawTensor<T> {
    fn clone(&self) -> Self {
        RawTensor {
            array: self.array.clone(),
            shape: self.shape.clone(),
            grad: self.grad.read().unwrap().clone().into(),
            grad_fn: self.grad_fn.clone(),
        }
    }
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
            grad: RwLock::new(None),
            array: vec![],
            grad_fn: None,
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
    T: Numeric,
{
    pub fn is_scalar(&self) -> bool {
        self.array.len() == 1 && self.shape.is_empty()
    }

    pub(in crate::tensor) fn global_index(
        &self,
        index: IndexType,
        offset: Option<&Vec<SliceRange>>,
    ) -> Result<usize, String> {
        global_index(index, &self.shape, offset)
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
        assert_eq!(
            len,
            array.len(),
            "Attempted to create a tensor with array.len()={}, but expected lenght: {len}
            this is because shape is: {shape:?}",
            array.len()
        );
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

    pub(in crate::tensor) fn set(&mut self, index: IndexType, value: T) -> Result<(), String> {
        match self.global_index(index, None) {
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
    fn get(&self, index: IndexType) -> Result<&T, String> {
        match self.global_index(index, None) {
            Ok(global_idx) => {
                // dbg!("self={}, index={}, global_idx={}", self, index, global_idx);
                Ok(&self.array[global_idx])
            }
            Err(e) => Err(e),
        }
    }

    pub(super) fn get_with_offset(
        &self,
        index: IndexType,
        offset: &Vec<SliceRange>,
    ) -> Result<&T, String> {
        match self.global_index(index, Some(offset)) {
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

    fn index(&self, index: IndexType) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<T> Neg for &RawTensor<T>
where
    T: Numeric + Neg<Output = T>,
{
    type Output = RawTensor<T>;
    fn neg(self) -> Self::Output {
        functional::generic_unary_op(self, |x| -x)
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

    fn dot<U, V>(&self, other: U) -> RcTensor<Self::Elem>
    where
        U: Deref<Target = V> + std::fmt::Debug + Clone,
        V: TensorLike<Elem = Self::Elem>,
    {
        F::dot_no_derivative(self, other)
    }

    #[inline]
    fn bmm_rc<U>(&self, right: &U) -> RcTensor<Self::Elem>
    where
        U: TensorLike<Elem = Self::Elem>,
    {
        RcTensor::from_raw(functional::bmm_raw(self, right))
    }

    fn zero_grad(&self) {
        *self.grad.write().unwrap() = None;
    }

    fn update_grad(&self, grad: Self::GradType) {
        let new_grad = match self.grad.read().unwrap().as_ref() {
            None => Some(grad),
            Some(other_grad) => Some(other_grad + &grad),
        };
        *self.grad.write().unwrap() = new_grad;
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
        scalar.update_grad(RcTensor::new_with_filler(
            self.shape.clone(),
            Self::Elem::one(),
        ));
        scalar
    }

    fn tensor(&self) -> Self::TensorRef<'_> {
        self
    }

    fn count(&self) -> usize {
        self.array.len()
    }

    fn to_tensor(&self) -> RcTensor<Self::Elem> {
        RcTensor::from_raw(self.clone())
    }
    fn slice(&self, offset: Vec<SliceRange>) -> TensorView<T> {
        TensorView::new(RcTensor::from_raw(self.clone()), offset)
    }
    fn get(&self, index: IndexType) -> Result<&Self::Elem, String> {
        self.get(index)
    }
    fn bmm<U>(&self, right: &U) -> Self::ResultTensorType<'_>
    where
        U: TensorLike<Elem = Self::Elem>,
    {
        functional::bmm_raw(self, right)
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
        functional::add_raw(self, right)
    }
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
        functional::element_wise_multiplication(self, right)
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
