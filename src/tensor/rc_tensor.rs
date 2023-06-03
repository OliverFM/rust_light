use std::rc::Rc;

use std::cell::RefCell;
use std::cmp::PartialEq;
use std::convert::From;
use std::ops::{Deref, Mul};

use super::autograd::{self, Derivative};

use super::numeric::*;
use super::raw_tensor::*;
use super::tensor_like::*;
use super::tensor_view::*;

// fn ones<T: Numeric>(tensors: Vec<RcTensor<T>>) -> RcTensor<T> {
//     assert_eq!(tensors.len(), 1);
//     RcTensor::new_with_filler(tensors[0].shape().to_vec(), T::one())
// }

#[derive(Debug, PartialEq, Clone)]
pub struct RcTensor<T: Numeric>(pub(super) Rc<RawTensor<T>>);

// TODO: make this a separate struct
pub type Scalar<T> = RcTensor<T>;

impl<T> Deref for RcTensor<T>
where
    T: Numeric,
{
    type Target = RawTensor<T>;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<T: Numeric> RcTensor<T> {
    pub fn is_scalar(&self) -> bool {
        self.0.array.len() == 1 && self.0.shape.is_empty()
    }

    pub(in crate::tensor) fn from_raw(raw_tensor: RawTensor<T>) -> RcTensor<T> {
        RcTensor(Rc::new(raw_tensor))
    }

    pub(super) fn compute_grad(&self) -> Option<Self> {
        // TODO: don't just unwrap, switch to a Result type and deal with the case of no gradient
        // appropriately
        self.derivative
            .as_ref()
            .map(|derivative| derivative.compute())
    }

    pub(in crate::tensor) fn new_empty(shape: Vec<usize>) -> RcTensor<T> {
        let raw_tensor = RawTensor::new_empty(shape);
        RcTensor(Rc::new(raw_tensor))
    }
    pub fn new_with_filler(shape: Vec<usize>, filler: T) -> RcTensor<T> {
        let raw_tensor = RawTensor::new_with_filler(shape, filler);
        RcTensor(Rc::new(raw_tensor))
    }

    pub fn scalar(scalar: T) -> RcTensor<T> {
        let raw_tensor = RawTensor::scalar(scalar);
        RcTensor(Rc::new(raw_tensor))
    }
    pub fn new(array: Vec<T>, shape: Vec<usize>) -> RcTensor<T> {
        let raw_tensor = RawTensor::new(array, shape);
        RcTensor(Rc::new(raw_tensor))
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
    fn get(&self, index: &Vec<usize>) -> Result<&T, String> {
        self.deref().get(index)
    }

    pub(in crate::tensor) fn get_with_offset(
        &self,
        index: &Vec<usize>,
        offset: &Vec<SliceRange>,
    ) -> Result<&T, String> {
        self.deref().get_with_offset(index, offset)
    }

    pub(in crate::tensor) fn get_grad<'a>(&self) -> RefCell<Option<RcTensor<T>>>
    where
        Self: 'a,
    {
        self.0.grad.clone()
    }
}

fn sum_backward<T: Numeric>(inputs: Vec<RcTensor<T>>, grads: Vec<RcTensor<T>>) -> Vec<RcTensor<T>> {
    assert_eq!(inputs.len(), 1);
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].0.array.len(), 1);
    vec![RcTensor::new_with_filler(
        inputs[0].shape().to_vec(),
        *grads[0].get_first_elem(),
    )]
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

    fn set_grad(&self, grad: Self::GradType) {
        let grad_clone = grad.clone();
        *self.grad.borrow_mut() = Some(grad);
        dbg!("setting grads:", self.clone(), grad_clone);
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

    fn get(&self, index: &Vec<usize>) -> Result<&Self::Elem, String> {
        self.deref().get(index)
    }

    fn sum(&self) -> Self::SumType {
        let mut raw_scalar = self.0.sum();
        raw_scalar.derivative = Some(Derivative::new(
            vec![self.clone()],
            autograd::ones,
            Some(sum_backward),
        ));
        Scalar::from_raw(raw_scalar)
    }

    fn tensor(&self) -> Self::TensorRef<'_> {
        self.clone()
    }

    fn to_tensor(&self) -> RcTensor<Self::Elem> {
        self.clone()
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
fn test_element_wise_multiplication() {
    let left = RcTensor::from([1, 2, 3]);
    let right = RcTensor::from([7, 2, 8]);
    dbg!("left={}, right={},", &left, &right);
    assert_eq!(&left * &right, RcTensor::from([7, 4, 24]));
}
// TODO: get this test to work: not sure why it doesn't work since RcTensor implements
// Deref<RawTensor>
#[test]
fn test_element_wise_multiplication_on_rc_tensor_directly() {
    let left = RcTensor::from([1, 2, 3]);
    let right = RcTensor::from([7, 2, 8]);
    dbg!("left={}, right={},", &left, &right);
    assert_eq!(left * right, RcTensor::from([7, 4, 24]));
}
