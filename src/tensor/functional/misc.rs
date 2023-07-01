use super::super::numeric::*;
use crate::tensor::autograd::Derivative;
use crate::tensor::utils::{global_index, ElementIterator};
use crate::tensor::{autograd, RawTensor, RcTensor, TensorLike};

use crossbeam::channel::{unbounded, Receiver, Sender};
use std::{iter, ops::Deref, sync::Arc};

use rayon;

pub(crate) fn todo_backward<T: Numeric>(
    _inputs: Vec<RcTensor<T>>,
    _grads: Vec<RcTensor<T>>,
) -> Vec<RcTensor<T>> {
    todo!()
}
pub(crate) fn todo_deriv<T: Numeric>(
    _inputs: Vec<RcTensor<T>>,
    _: Vec<RcTensor<T>>,
) -> Vec<RcTensor<T>> {
    todo!()
}
#[inline]
pub(crate) fn dot_raw<T, U1, U2, V1, V2>(left: U1, right: U2) -> RawTensor<T>
where
    T: Numeric,
    U1: Deref<Target = V1> + std::fmt::Debug + Clone,
    V1: TensorLike<Elem = T>,
    U2: Deref<Target = V2> + std::fmt::Debug + Clone,
    V2: TensorLike<Elem = T>,
{
    //! generalised dot product: returns to acculumulated sum of the elementwise product.
    assert!(left.same_shape(&right));
    let mut result = T::zero();
    for i in 0..left.tensor().array.len() {
        result += left.tensor().array[i] * right.tensor().array[i];
    }
    RawTensor {
        array: vec![result],
        shape: vec![1],
        ..Default::default()
    }
}

#[inline]
pub(crate) fn bmm_raw<T, U1, U2, V1, V2>(left: U1, right: U2) -> RawTensor<T>
where
    T: Numeric,
    // U1: Deref<Target = V1> + std::fmt::Debug + Clone  ,
    U1: Deref<Target = V1> + std::fmt::Debug + Clone + Send + Sync,
    V1: TensorLike<Elem = T>,
    // U2: Deref<Target = V2> + std::fmt::Debug + Clone  ,
    U2: Deref<Target = V2> + std::fmt::Debug + Clone + Send + Sync,
    V2: TensorLike<Elem = T>,
{
    // assert!(2 <= self.shape().len() && self.shape().len() <= 3); // For now we can only do Batch matrix
    //    dbg!(left.shape(), right.shape());
    assert!(2 <= left.shape().len()); // For now we can only do Batch matrix
    assert!(right.shape().len() == 2); // rhs must be a matrix
                                       //
    assert!(left.shape()[left.shape().len() - 1] == right.shape()[right.shape().len() - 2]);
    let new_shape: Vec<usize> = if left.shape().len() == 2 {
        vec![1, left.shape()[0], right.shape()[1]]
    } else {
        vec![left.shape()[0], left.shape()[1], right.shape()[1]]
    };

    // let mut result_array = Vec::with_capacity(new_shape.iter().product::<usize>());
    let mut result_array = vec![T::zero(); new_shape.iter().product()];

    let mut left_index = left.shape().clone();
    let left_index_len = left_index.len();
    let right_index = right.shape().clone();
    let right_shape = &right.shape().clone();
    let mut remaining_slice = &mut result_array[..];
    for batch_idx in 0..new_shape[0] {
        if left.shape().len() == 3 {
            left_index[0] = batch_idx;
        }
        for i in 0..new_shape[1] {
            let mut slice;
            (slice, remaining_slice) = remaining_slice.split_at_mut(new_shape[2]);
            // dbg!(&remaining_slice.len());
            assert_eq!(slice.len(), new_shape[2]);
            left_index[left_index_len - 2] = i;

            rayon::in_place_scope(|s| {
                // std::thread::scope(|s| {
                // dbg!(&new_shape[]);
                for j in 0..new_shape[2] {
                    let slot;
                    (slot, slice) = slice.split_at_mut(1);
                    let mut right_index2 = right_index.clone();
                    let mut left_index2 = left_index.clone();
                    let left2 = &left;
                    let right2 = &right;
                    let mut closure = move || {
                        // dbg!(rayon::current_thread_index());
                        // dbg!(std::thread::current().id(), std::thread::current().name(),);
                        right_index2[1] = j;
                        let mut val = T::zero();
                        for k in 0..right_shape[0] {
                            // dbg!(&k, rayon::current_thread_index().unwrap());
                            left_index2[left_index_len - 1] = k;
                            right_index2[0] = k;
                            val += *left2.get(&left_index2).unwrap()
                                * (*right2.get(&right_index2).unwrap());
                        }

                        slot[0] = val;
                    };
                    if right.shape()[0] > 0 {
                        s.spawn(move |_| closure()); // Rayon version

                    // s.spawn(move || closure());
                    } else {
                        closure();
                    }
                }
            });
            // dbg!(&remaining_slice);
        }
    }
    // dbg!(&result_array);
    if left.shape().len() == 2 {
        RawTensor {
            array: result_array,
            shape: new_shape[1..].to_vec(),
            ..Default::default()
        }
    } else {
        RawTensor {
            array: result_array,
            shape: new_shape,

            ..Default::default()
        }
    }
}

struct SplitArrayWriter<'a, T: Numeric> {
    segments: Vec<&'a mut [T]>,
    bit_width: usize,
    width: usize,
    splits: usize,
    senders: Option<Arc<Vec<Sender<(usize, T)>>>>,
    receivers: Vec<Receiver<(usize, T)>>,
}

impl<'a, T: Numeric> SplitArrayWriter<'a, T> {
    fn new(slice: &'a mut [T]) -> Self {
        let bit_width = 5;
        let width = 1 << bit_width;
        let splits = slice.len() / width;
        let segments: Vec<_> = slice.rchunks_mut(width).collect();
        let (senders, receivers) = iter::repeat_with(unbounded).take(segments.len()).unzip();
        SplitArrayWriter {
            segments,
            bit_width,
            width,
            splits,
            senders: Some(Arc::new(senders)),
            receivers,
        }
    }

    pub fn get_writer(
        &self,
    ) -> impl Fn(usize, T) -> Result<(), crossbeam::channel::SendError<(usize, T)>> {
        let senders = self.senders.clone().unwrap();
        let bit_width = self.bit_width;
        // let mask = (1 << bit_width) - 1;
        move |global_idx, val| {
            let sender_idx = global_idx >> bit_width;
            senders[sender_idx].send((global_idx, val))
        }
    }

    pub fn start_then_join<'b>(mut self, scope: &rayon::Scope<'b>)
    where
        Self: 'b,
    {
        self.senders = None;
        scope.spawn(move |inner_scope| {
            self.segments
                .into_iter()
                .zip(self.receivers.into_iter())
                .enumerate()
                .for_each(move |(slice_idx, (slice, rx))| {
                    inner_scope.spawn(move |_| {
                        rx.iter().for_each(|(global_idx, value)| {
                            let local_idx: usize = global_idx - slice_idx * self.width;
                            assert!(local_idx < slice.len(), 
                                "got invalid index: global_idx={global_idx}, local_idx={local_idx}, slice.len()={}", slice.len());
                            slice[local_idx] += value;
                        })
                    })
                })
        })
    }
}

/// c[i,j] = dot(A[i, ..], B[..,j])
/// so A[i,k] * B[k,j] appears for all j -> so J_A[ø[i,j], ø[i,k], ] = B[k,j]
/// so A[i,k] * B[k,j] appears for all j -> so J_A[ø_J[i,j], ø_B[k,j], ] = A[i,k]
/// : Key thing to note here is that J
/// is over a flat input, and we are thinking about matrices, which makes this a bit weird.
/// So we need to have some map ø(i,j)->k where i,j are matrix coords, and k is the param vector
/// coords
/// need to compute jacobians[0] @ J_A, jacobians[0] @ J_B with J_A being a matrix
pub(crate) fn bmm_jvp<T: Numeric>(
    inputs: Vec<RcTensor<T>>,
    jacobians: Vec<RcTensor<T>>,
) -> Vec<RcTensor<T>> {
    // assert!(match std::env::var("RAYON_NUM_THREADS") {
    //     Ok(s) => match s.parse::<usize>() {
    //         Ok(n) => n >= 2,
    //         _ => false,
    //     },
    //     _ => true,
    // });
    assert!(
        inputs.len() == 2 && jacobians.len() == 1,
        "inputs.len()={}, jacobians.len()={}",
        inputs.len(),
        jacobians.len()
    );

    assert_eq!(
        inputs[0].shape()[1],
        inputs[1].shape()[0],
        "inputs[0].shape()={:?}, inputs[1].shape()={:?}",
        inputs[0].shape(),
        inputs[1].shape()
    );
    let bmm_output_shape = vec![inputs[0].shape()[0], inputs[1].shape()[1]];
    let left_jacobian_shape = vec![
        bmm_output_shape[0] * bmm_output_shape[1],
        inputs[0].shape()[0] * inputs[0].shape()[1], // we are only doing this for the left input
    ];
    let left_output_shape = vec![jacobians[0].shape()[0], left_jacobian_shape[1]];
    let left_output_shape_ref = &left_output_shape;
    let left_length = left_output_shape_ref[0] * left_output_shape_ref[1]; // jacobians[0].shape()[0] * inputs[1].shape()[1];
    let mut left_array = vec![T::zero(); left_length];

    let right_jacobian_shape = vec![
        bmm_output_shape[0] * bmm_output_shape[1],
        inputs[1].shape()[0] * inputs[1].shape()[1], // we are only doing this for the left input
    ];
    let right_output_shape = vec![jacobians[0].shape()[0], right_jacobian_shape[1]];
    let right_output_shape_ref = &right_output_shape;
    let right_length = right_output_shape_ref[0] * right_output_shape_ref[1]; // jacobians[0].shape()[0] * inputs[1].shape()[1];
    let mut right_array = vec![T::zero(); right_length];

    // currently: loop through all the non-zero values:
    // J_A[ø[i,j], ø[i,k]] = B[k,j]
    // Consider seeing if there is a way to get this to work such that we also build the array as
    rayon::in_place_scope(|s| {
        let mut left_split_writer = SplitArrayWriter::new(&mut left_array[..]);
        let mut right_split_writer = SplitArrayWriter::new(&mut right_array[..]);
        let left_split_writer_ref = &mut left_split_writer;
        let right_split_writer_ref = &mut right_split_writer;
        for i in 0..inputs[0].shape()[0] {
            for k in 0..inputs[0].shape()[1] {
                for j in 0..inputs[1].shape()[1] {
                    let left_writer = left_split_writer_ref.get_writer();
                    let right_writer = right_split_writer_ref.get_writer();
                    let inputs_ref = &inputs;
                    let jacobians_ref = &jacobians;
                    let bmm_output_shape_ref = &bmm_output_shape;
                    let left_jacobian_shape_ref = &left_jacobian_shape;
                    s.spawn(move |_| {
                        let self_jac_idx0 =
                            global_index(&vec![i, j], bmm_output_shape_ref, None).unwrap();
                        let left_jac_idx1 =
                            global_index(&vec![i, k], inputs_ref[0].shape(), None).unwrap(); // for J_A
                        let right_jac_idx1 =
                            global_index(&vec![k, j], inputs_ref[1].shape(), None).unwrap(); // for J_A
                                                                                             // println!(
                                                                                             //     "i={i}, k={k}, j={j}, self_jac_idx0={:?}, self_jac_idx1={:?}",
                                                                                             //     &self_jac_idx0, &left_jac_idx1,
                                                                                             // );
                        assert!(self_jac_idx0 < left_jacobian_shape_ref[0]);
                        assert!(self_jac_idx0 < jacobians_ref[0].shape()[1]);
                        assert!(left_jac_idx1 < left_jacobian_shape_ref[1]);

                        for input_jac_idx in 0..jacobians_ref[0].shape()[0] {
                            let tmp_left = match global_index(
                                &vec![input_jac_idx, left_jac_idx1],
                                left_output_shape_ref,
                                None,
                            ) {
                                Ok(t) => t,
                                Err(e) => {
                                    panic!("{e}")
                                }
                            };

                            let left_val = jacobians_ref[0][&vec![input_jac_idx, self_jac_idx0]]
                                * inputs_ref[1][&vec![k, j]];
                            left_writer(tmp_left, left_val).unwrap();

                            let tmp_right = match global_index(
                                &vec![input_jac_idx, right_jac_idx1],
                                right_output_shape_ref,
                                None,
                            ) {
                                Ok(t) => t,
                                Err(e) => {
                                    panic!("{e}")
                                }
                            };
                            // println!("tmp_right={tmp_right:?}");
                            let right_val = jacobians_ref[0][&vec![input_jac_idx, self_jac_idx0]]
                                * inputs_ref[0][&vec![i, k]];
                            right_writer(tmp_right, right_val).unwrap();
                        }
                    });

                    // s.spawn(closure);
                }
            }
        }
        right_split_writer.start_then_join(s);
        left_split_writer.start_then_join(s);
    });
    vec![
        RcTensor::new(left_array, left_output_shape),
        RcTensor::new(right_array, right_output_shape),
    ]
}

pub fn element_wise_multiplication<T, U1, V1, U2, V2>(left: U1, right: U2) -> RawTensor<T>
where
    T: Numeric,
    U1: Deref<Target = V1> + std::fmt::Debug + Clone,
    V1: TensorLike<Elem = T>,
    U2: Deref<Target = V2> + Clone + std::fmt::Debug,
    V2: TensorLike<Elem = T>,
{
    let left_shape_vec = left.shape().to_vec();
    assert!(left.same_shape(&right));
    let array = ElementIterator::new(left)
        .zip(ElementIterator::new(right))
        .map(|(x, y)| x * y)
        .collect();

    RawTensor::new(array, left_shape_vec)
}

pub fn dot_no_derivative<T, U1, U2, V1, V2>(left: U1, right: U2) -> RcTensor<T>
where
    T: Numeric,
    U1: Deref<Target = V1> + std::fmt::Debug + Clone,
    V1: TensorLike<Elem = T>,
    U2: Deref<Target = V2> + Clone + std::fmt::Debug,
    V2: TensorLike<Elem = T>,
{
    RcTensor::from_raw(dot_raw(left, right))
}

fn dot_jvp<T: Numeric>(inputs: Vec<RcTensor<T>>, jacobians: Vec<RcTensor<T>>) -> Vec<RcTensor<T>> {
    assert!(
        inputs.len() == 2 && jacobians.len() == 1,
        "inputs.len()={}, jacobians.len()={}",
        inputs.len(),
        jacobians.len()
    );
    let (left, right) = (inputs[0].clone(), inputs[1].clone());
    let jacobian = jacobians[0].clone();
    let left_jvp = autograd::jvp_from_full_jacobians(jacobian.clone(), right);
    let right_jvp = autograd::jvp_from_full_jacobians(jacobian, left);
    vec![left_jvp, right_jvp]
}

#[test]
fn test_dot_autograd() {
    let left = RcTensor::from([1.0, 2.0, 3.0]);
    let right = RcTensor::from([4.0, 5.0, 6.0]);
    dot(&left, &right).backward();
    assert_eq!(right, left.get_grad().unwrap());
    assert_eq!(left, right.get_grad().unwrap());
}

// TODO: generalise this to views
pub fn dot<T>(left: &RcTensor<T>, right: &RcTensor<T>) -> RcTensor<T>
where
    T: Numeric,
{
    let mut raw_tensor = dot_raw(left, right);
    raw_tensor.grad_fn = Some(Derivative::new(
        vec![left.clone(), right.clone()],
        dot_jvp,
        format!("dot, file: {}, line: {}", file!(), line!(),),
    ));

    RcTensor::from_raw(raw_tensor)
}

#[test]
fn test_dot() {
    let v = vec![0, 1, 2];
    let vec = RcTensor::new(v, vec![3]);
    assert_eq!(dot(&vec, &vec), RcTensor::new(vec![5], vec![1]));
}

#[test]
fn test_bmm_jvp() {
    // for (matrix_a, matrix_b)
    let matrix_a = RcTensor::from([[1.0, 2.0], [3.0, 4.0]]);
    let matrix_b = RcTensor::from([[10.0, 3.0], [42.0, -7.0]]);

    // calculated by hand and checked against pytorch
    let expected_jvp_a = RcTensor::from([[13.0, 42.0 + -7.0], [10.0 + 3.0, 42.0 + -7.0]]);
    let expected_jvp_b = RcTensor::from([[4., 4.], [6., 6.]]);
    matrix_a.bmm(&matrix_b).sum().backward();
    assert_eq!(matrix_a.get_grad().unwrap(), expected_jvp_a);
    assert_eq!(matrix_b.get_grad().unwrap(), expected_jvp_b);
}

#[test]
fn test_bmm_jvp_differing_shapes() {
    let matrix_a = RcTensor::from([[1.0, 2.0], [3.0, 4.0], [123.4, 1e-3]]);
    let matrix_b = RcTensor::from([[10.0, 1e-2, -12.0], [42.0, 3.142, -7.0]]);

    // calculated by hand and checked against pytorch
    let expected_jvp_a =
        RcTensor::from([[-1.9900, 38.1420], [-1.9900, 38.1420], [-1.9900, 38.1420]]);
    let expected_jvp_b = RcTensor::from([[127.4500, 127.4500, 127.4500], [6.0010, 6.0010, 6.0010]]);
    matrix_a.bmm(&matrix_b).sum().backward();
    assert!(
        (matrix_a.get_grad().unwrap() - &expected_jvp_a)
            .sum()
            .elem()
            <= 1e-3
    );

    assert!(
        (matrix_b.get_grad().unwrap() - &expected_jvp_b)
            .sum()
            .elem()
            <= 1e-3
    );
}

#[test]
fn test_add_grad() {
    for (left, right, left_grad, _right_grad) in vec![(
        RcTensor::from([1.0, 2.0, 3.0]),
        RcTensor::from([10.0, 42.0, -5.0]),
        RcTensor::from([1.0, 1.0, 1.0]),
        RcTensor::from([1.0, 1.0, 1.0]),
    )] {
        (&left + &right).sum().backward();

        let computed_grad = left.get_grad().unwrap().deref().clone();
        let diff = &computed_grad - &left_grad;
        //        dbg!(&computed_grad, &left_grad, &diff);
        assert!(diff.sum().elem() <= 1e-3);

        let computed_grad = right.get_grad().unwrap().deref().clone();
        let diff = &computed_grad - &left_grad;
        //        dbg!(&computed_grad, &right_grad, &diff);
        assert!(diff.sum().elem() <= 1e-3);
    }
}

#[test]
fn test_bmm_2x2() {
    let v = vec![0, 1, 2, 3];
    let matrix = RcTensor::new(v, vec![2, 2]); // [[0,1],[2,3]]
    let shape = vec![2, 1];
    let e1 = RcTensor::new(vec![0, 1], vec![2, 1]);
    let e2 = RcTensor::new(vec![1, 0], vec![2, 1]);
    let diag = RcTensor::new(vec![1, 1], vec![2, 1]);

    let r = matrix.bmm(&diag);
    assert_eq!(r.shape(), &shape);
    assert_eq!(r, RcTensor::new(vec![1, 5], shape.clone()));
    matrix.zero_grad();
    let r = matrix.bmm(&e1);
    r.sum().backward();
    assert_eq!(r, RcTensor::new(vec![1, 3], shape.clone()));
    matrix.grad();
    matrix.zero_grad();
    r.sum().backward();
    let r = matrix.bmm(&e2);
    matrix.grad();
    assert_eq!(r, RcTensor::new(vec![0, 2], shape.clone()));
}

#[test]
fn test_bmm_runs() {
    use rand::random;
    for (left_shape, right_shape) in vec![
        (vec![2, 2], vec![2, 2]),
        (vec![2, 2], vec![2, 1]),
        (vec![8, 2], vec![2, 1]),
        // (vec![2, 8, 2], vec![2, 1]), // to get this to work we need to allow tensor
        // views/reshaping
    ] {
        let length = left_shape.iter().product();
        let mut left_array: Vec<f32> = Vec::with_capacity(length);
        for _ in 0..length {
            left_array.push(random());
        }
        let left_array = left_array;
        //        dbg!(&left_array, &left_shape, &length);
        let length = right_shape.iter().product();
        let mut right_array: Vec<f32> = Vec::with_capacity(length);
        for _ in 0..length {
            right_array.push(random());
        }

        let left = RcTensor::new(left_array, left_shape);
        let right = RcTensor::new(right_array, right_shape);
        left.bmm(&right).sum().backward();
    }
}
