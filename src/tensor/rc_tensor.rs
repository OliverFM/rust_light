use num::traits::real::Real;


use std::cmp::PartialEq;
use std::convert::From;
use std::{sync::Arc, ops::{Add, Deref, Mul, Sub}};

use super::autograd::{self, Derivative};
use super::numeric::*;
use super::raw_tensor::*;
use super::tensor_like::*;
use super::tensor_view::*;
use crate::tensor::functional;
use crate::tensor::{IndexType, Scalar};

// fn ones<T: Numeric>(tensors: Vec<RcTensor<T>>) -> RcTensor<T> {
//     assert_eq!(tensors.len(), 1);
//     RcTensor::new_with_filler(tensors[0].shape().to_vec(), T::one())
// }

/// Core struct: a reference counted Tensor
///
/// This lets you keep many pointers to the same data, which cna be very handy
/// with using one tensor as an input to many functions.
#[derive(Clone,Debug, PartialEq, )]
pub struct RcTensor<T: Numeric>(pub(super) Arc<RawTensor<T>>);

impl<T> Deref for RcTensor<T>
where
    T: Numeric,
{
    type Target = RawTensor<T>;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<T> RcTensor<T>
where
    T: Numeric + Real,
{
    pub fn abs(&self) -> Self {
        functional::abs(self.clone())
    }
}
impl<T: Numeric> RcTensor<T> {
    pub fn backward(&self) {
        assert_eq!(
            self.0.array.len(),
            1,
            "Can only operate on tensors with exactly one element!"
        );
        self.grad_fn
            .as_ref()
            .unwrap()
            .compute_jvp(vec![RcTensor::scalar(T::one())])
    }
    pub fn is_scalar(&self) -> bool {
        self.0.array.len() == 1 && self.0.shape.is_empty()
    }

    pub(in crate::tensor) fn from_raw(raw_tensor: RawTensor<T>) -> RcTensor<T> {
        RcTensor(Arc::new(raw_tensor))
    }

    // pub(super) fn compute_grad(&self) -> Option<Self> {
    //     // TODO: don't just unwrap, switch to a Result type and deal with the case of no gradient
    //     // appropriately
    //     self.derivative
    //         .as_ref()
    //         .map(|derivative| derivative.compute())
    // }

    pub(in crate::tensor) fn new_empty(shape: Vec<usize>) -> RcTensor<T> {
        let raw_tensor = RawTensor::new_empty(shape);
        RcTensor(Arc::new(raw_tensor))
    }
    pub fn new_with_filler(shape: Vec<usize>, filler: T) -> RcTensor<T> {
        let raw_tensor = RawTensor::new_with_filler(shape, filler);
        RcTensor(Arc::new(raw_tensor))
    }

    pub fn scalar(scalar: T) -> RcTensor<T> {
        let raw_tensor = RawTensor::scalar(scalar);
        RcTensor(Arc::new(raw_tensor))
    }
    pub fn new(array: Vec<T>, shape: Vec<usize>) -> RcTensor<T> {
        let raw_tensor = RawTensor::new(array, shape);
        RcTensor(Arc::new(raw_tensor))
    }

    pub fn view(&self, shape: Vec<SliceRange>) -> TensorView<T> {
        TensorView::new(self.clone(), shape)
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
        self.deref().get(index)
    }

    pub(in crate::tensor) fn get_with_offset(
        &self,
        index: IndexType,
        offset: &Vec<SliceRange>,
    ) -> Result<&T, String> {
        self.deref().get_with_offset(index, offset)
    }

    pub(in crate::tensor) fn get_grad<'a>(&self) -> Option<RcTensor<T>>
    where
        Self: 'a,
    {
        self.0.grad.read().unwrap().clone()
    }

    //TODO: deprecate
    pub fn grad(&self) -> RcTensor<T> {
        self.get_grad().unwrap()
    }
}

impl<T: Numeric> std::fmt::Display for RcTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0.array)
    }
}

impl<T> TensorLikePrivate for RcTensor<T> where T: Numeric {}
impl<T> TensorLike for RcTensor<T>
where
    T: Numeric,
{
    type Elem = T;
    type ShapeReturn<'a> = &'a Vec<usize> where Self: 'a;
    type TensorRef<'a> = RcTensor<Self::Elem> where Self: 'a;
    type ResultTensorType<'a>= RcTensor<T> where Self: 'a; // &'tensor Tensor<Self::Elem> where Self : 'tensor;
    type SumType = Scalar<Self::Elem>;
    type GradType = RcTensor<T>;

    fn update_grad(&self, grad: Self::GradType) {
        //        dbg!("updating grad:", &self, &grad);
        self.0.update_grad(grad);
        //        dbg!("update complete", &self);
    }

    fn zero_grad(&self) {
        //        dbg!("zeroing grad", &self);
        self.0.zero_grad();
    }

    fn dot<U, V>(&self, _other: U) -> RcTensor<Self::Elem>
    where
        U: Deref<Target = V> + std::fmt::Debug + Clone,
        V: TensorLike<Elem = Self::Elem>,
    {
        todo!()
    }

    fn shape(&self) -> Self::ShapeReturn<'_> {
        self.deref().shape()
    }

    fn get(&self, index: IndexType) -> Result<&Self::Elem, String> {
        self.deref().get(index)
    }

    fn sum(&self) -> Self::SumType {
        let mut raw_scalar = self.0.sum();
        raw_scalar.grad_fn = Some(Derivative::new(
            vec![self.clone()],
            autograd::ones,
            format!("sum, file: {}, line: {}", file!(), line!(),),
        ));
        Scalar::from_raw(raw_scalar)
    }

    fn tensor(&self) -> Self::TensorRef<'_> {
        self.clone()
    }

    fn to_tensor(&self) -> RcTensor<Self::Elem> {
        self.clone()
    }

    fn count(&self) -> usize {
        self.0.array.len()
    }

    fn slice(&self, offset: Vec<SliceRange>) -> TensorView<T> {
        TensorView::new(self.clone(), offset)
    }
    fn bmm<U>(&self, right: &U) -> Self::ResultTensorType<'_>
    where
        U: TensorLike<Elem = Self::Elem>,
    {
        self.bmm_rc(right)
    }

    #[inline]
    fn bmm_rc<U>(&self, right: &U) -> RcTensor<Self::Elem>
    where
        U: TensorLike<Elem = Self::Elem>,
    {
        let mut raw_tensor = functional::bmm_raw(self, right);
        raw_tensor.grad_fn = Some(Derivative::new(
            vec![self.clone(), right.to_tensor()],
            functional::bmm_jvp,
            format!("bmm_rc, file: {}, line: {}", file!(), line!(),),
        ));
        RcTensor::from_raw(raw_tensor)
    }
}

impl<T> From<T> for RcTensor<T>
where
    T: Numeric,
{
    fn from(value: T) -> Self {
        RcTensor::from_raw(RawTensor::from(value))
    }
}

impl<T, U> From<Vec<U>> for RcTensor<T>
where
    T: Numeric,
    RawTensor<T>: From<U>,
{
    fn from(value: Vec<U>) -> RcTensor<T> {
        let tmp = <RawTensor<T> as From<Vec<U>>>::from(value);
        RcTensor::from_raw(tmp)
    }
}

impl<T, U, const N: usize> From<[U; N]> for RcTensor<T>
where
    T: Numeric,
    RawTensor<T>: From<U>,
    U: Clone,
{
    fn from(value: [U; N]) -> RcTensor<T> {
        let raw_tensor = From::from(value.to_vec());
        RcTensor::from_raw(raw_tensor)
    }
}

impl<T, U> Add<U> for &RcTensor<T>
where
    T: Numeric,
    U: Deref<Target = RcTensor<T>> + Clone + std::fmt::Debug,
{
    type Output = RcTensor<T>;
    fn add(self, right: U) -> Self::Output {
        let mut raw_tensor = functional::add_raw(self, right.deref().clone());
        raw_tensor.grad_fn = Some(Derivative::new(
            vec![self.clone(), right.deref().clone()],
            functional::add_jvp,
            format!("add, file: {}, line: {}", file!(), line!(),),
        ));
        RcTensor::from_raw(raw_tensor)
    }
}

impl<T> Add<RcTensor<T>> for RcTensor<T>
where
    T: Numeric,
    // U: Deref<Target = V> + Clone + std::fmt::Debug,
    // V: TensorLike<Elem = T>,
{
    type Output = RcTensor<T>;
    fn add(self, right: RcTensor<T>) -> Self::Output {
        let mut raw_tensor = functional::add_raw(&self, &right);
        raw_tensor.grad_fn = Some(Derivative::new(
            vec![self, right],
            functional::add_jvp,
            format!("add, file: {}, line: {}", file!(), line!(),),
        ));
        RcTensor::from_raw(raw_tensor)
        // let result = functional::add(self.clone(), right.clone());
        //        // dbg!(self.clone(), right.clone(), result.clone());
        // result
    }
}

impl<T, U> Sub<U> for &RcTensor<T>
where
    T: Numeric + Neg<Output = T>,
    U: Deref<Target = RcTensor<T>> + Clone + std::fmt::Debug,
{
    type Output = RcTensor<T>;
    fn sub(self, right: U) -> Self::Output {
        let neg_right: RcTensor<T> = right.deref().clone().neg();
        self.add(&neg_right)
    }
}

impl<T> Sub<RcTensor<T>> for RcTensor<T>
where
    T: Numeric + Neg<Output = T>,
{
    type Output = RcTensor<T>;
    fn sub(self, right: RcTensor<T>) -> Self::Output {
        self.add(right.neg())
    }
}
impl<T> Sub<&RcTensor<T>> for RcTensor<T>
where
    T: Numeric + Neg<Output = T>,
{
    type Output = RcTensor<T>;
    fn sub(self, right: &RcTensor<T>) -> Self::Output {
        self.add(right.neg())
    }
}

impl<T> Neg for &RcTensor<T>
where
    T: Numeric + Neg<Output = T>,
{
    type Output = RcTensor<T>;
    fn neg(self) -> Self::Output {
        let mut raw_tensor = self.0.neg();
        raw_tensor.grad_fn = Some(Derivative::new(
            vec![self.clone()],
            |tensors, grads| {
                functional::generic_unary_jvp(&tensors[0], &grads[0], |_| T::one().neg())
            },
            format!("neg, file: {}, line: {}", file!(), line!(),),
        ));
        RcTensor::from_raw(raw_tensor)
    }
}

impl<T, U, V> Mul<U> for RcTensor<T>
where
    T: Numeric,
    U: Deref<Target = V> + Clone + std::fmt::Debug,
    V: TensorLike<Elem = T>,
{
    type Output = RcTensor<T>;

    fn mul(self, right: U) -> Self::Output {
        let raw_tensor = self.0.deref().mul(right);
        RcTensor::from_raw(raw_tensor)
    }
}

impl<T, U, V> Mul<U> for &RcTensor<T>
where
    T: Numeric,
    U: Deref<Target = V> + Clone + std::fmt::Debug,
    V: TensorLike<Elem = T>,
{
    type Output = RcTensor<T>;

    fn mul(self, right: U) -> Self::Output {
        let raw_tensor = self.0.deref().mul(right);
        RcTensor::from_raw(raw_tensor)
    }
}

#[test]
fn test_update_grad() {
    let tensor = RcTensor::from([1.0]);
    let grad = RcTensor::from([7.0]);
    tensor.clone().update_grad(grad.clone());
    assert_eq!(tensor.grad(), grad);
}

#[test]
fn test_neg() {
    let tensor = RcTensor::from([1.0]);
    tensor.neg().backward();
    assert_eq!(tensor.grad(), RcTensor::from([-1.0]));
}

#[test]
fn test_add_two_inputs() {
    let tensor1 = RcTensor::from([111.0]);
    let tensor2 = RcTensor::from([-12.0]);
    (&tensor1 + &tensor2).backward();
    //    dbg!(&tensor2);
    assert_eq!(tensor1.grad(), RcTensor::from([1.0]));
    assert_eq!(tensor2.grad(), RcTensor::from([1.0]));
}

#[test]
fn test_add_two_inputs_cloned() {
    let tensor1 = RcTensor::from([111.0]);
    let tensor2 = RcTensor::from([-12.0]);
    (tensor1.clone() + tensor2.clone()).backward();
    //    dbg!(&tensor2);
    assert_eq!(tensor1.grad(), RcTensor::from([1.0]));
    assert_eq!(tensor2.grad(), RcTensor::from([1.0]));
}

#[test]
fn test_add_3() {
    let tensor1 = RcTensor::from([1.0]);
    let tensor2 = RcTensor::from([1.0]);
    let tensor3 = RcTensor::from([1.0]);
    (&tensor1.clone().neg() + &(tensor2.clone() + tensor3.clone())).backward();
    assert_eq!(tensor1.grad(), RcTensor::from([-1.0]));
    assert_eq!(tensor2.grad(), RcTensor::from([1.0]));
    assert_eq!(tensor3.grad(), RcTensor::from([1.0]));
}

#[test]
fn test_element_wise_multiplication() {
    let left = RcTensor::from([1, 2, 3]);
    let right = RcTensor::from([7, 2, 8]);
    //    //    dbg!("left={}, right={},", &left, &right);
    assert_eq!(&left * &right, RcTensor::from([7, 4, 24]));
}

#[test]
fn test_element_wise_multiplication_on_rc_tensor_directly() {
    let left = RcTensor::from([1, 2, 3]);
    let right = RcTensor::from([7, 2, 8]);
    //    //    dbg!("left={}, right={},", &left, &right);
    assert_eq!(left * right, RcTensor::from([7, 4, 24]));
}
