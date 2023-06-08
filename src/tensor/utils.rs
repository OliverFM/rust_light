use crate::tensor::numeric::Numeric;
use crate::tensor::SliceRange;
use crate::tensor::TensorLike;

pub struct ElementIterator<T, U, V>
where
    U: Deref<Target = V>,
    V: TensorLike<Elem = T>,
    T: Numeric,
{
    index: Vec<usize>,
    tensor_like: U,
    first: bool,
}

impl<T, U, V> ElementIterator<T, U, V>
where
    T: Numeric,
    U: Deref<Target = V> + Clone,
    V: TensorLike<Elem = T>,
{
    pub fn new(tensor_like: U) -> ElementIterator<T, U, V> {
        ElementIterator {
            index: vec![0; tensor_like.deref().shape().len()],
            tensor_like: tensor_like.clone(),
            first: true,
        }
    }
}

impl<T, U, V> Iterator for ElementIterator<T, U, V>
where
    T: Numeric,
    U: Deref<Target = V>,
    V: TensorLike<Elem = T>,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.first {
            self.first = false;
            let &elem = self.tensor_like.get(&self.index).unwrap();
            return Some(elem);
        }
        if increment_index(&mut self.index, self.tensor_like.shape()) {
            let &elem = self.tensor_like.get(&self.index).unwrap();
            return Some(elem);
        }
        None
    }
}

#[test]
fn test_element_iterator() {
    use crate::tensor::{RcTensor, SliceRange};
    let v = [1, 2, 3];
    let tensor = RcTensor::from(v);
    let view = tensor.view(vec![SliceRange::new(0, 3)]);
    let tensor_element_iterator = ElementIterator::new(&tensor);
    let element_iterator = ElementIterator::new(&view);
    for ((view_elem, tensor_elem), &expected) in
        element_iterator.zip(tensor_element_iterator).zip(v.iter())
    {
        // println!("elem={elem:?}, expected={expected:?}");
        assert_eq!(view_elem, expected);
        assert_eq!(tensor_elem, expected);
    }
}

pub struct IndexIterator {
    index: Vec<usize>,
    dimensions: Vec<usize>,
    first: bool,
}

impl IndexIterator {
    pub fn new(dimensions: Vec<usize>) -> IndexIterator {
        IndexIterator {
            index: vec![0; dimensions.len()],
            dimensions,
            first: true,
        }
    }
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.first {
            self.first = false;
            return Some(self.index.clone());
        }
        if increment_index(&mut self.index, &self.dimensions) {
            return Some(self.index.clone());
        }
        None
    }
}

use std::ops::Deref;
pub fn increment_index<V: Deref<Target = Vec<usize>>>(index: &mut [usize], shape: V) -> bool {
    let mut carry = 1;
    for i in (0..index.len()).rev() {
        let v = index[i];
        let dim = shape[i];
        while v < dim - 1 && carry > 0 {
            index[i] = v + 1;
            carry -= 1;
        }
        if carry == 0 {
            reset_trailing_indices(index, i);
            return true;
        }
    }
    false
}

fn reset_trailing_indices(index: &mut [usize], position: usize) {
    for idx in index.iter_mut().skip(position + 1) {
        *idx = 0;
    }
}

pub(in crate::tensor) fn tensor_index(
    global_index: usize,
    shape: &[usize],
) -> Result<Vec<usize>, String> {
    let mut index = Vec::with_capacity(shape.len());
    match tensor_index_inplace(global_index, shape, &mut index) {
        Err(e) => Err(e),
        Ok(()) => Ok(index),
    }
}

pub(in crate::tensor) fn tensor_index_inplace(
    global_index: usize,
    shape: &[usize],
    index: &mut Vec<usize>,
) -> Result<(), String> {
    index.clear();
    // shape.iter().fold(global_index, |acc, dim|  )

    for _ in 0..shape.len() {
        index.push(0);
    }

    let mut global_index = global_index;
    for (idx, &dim) in shape.iter().enumerate().rev() {
        index[idx] = global_index % dim;
        global_index /= dim;
    }

    if global_index > 0 {
        Err(format!("index is too big: {global_index}"))
    } else {
        Ok(())
    }
}

#[test]
fn test_tensor_index() {
    let mut index = Vec::new();
    for shape in [
        vec![1, 2, 3, 1],
        vec![3, 12, 7, 4, 1, 1, 2, 13],
        vec![1, 2, 3, 4, 5, 7],
    ] {
        for i in 0..shape.iter().product::<usize>() {
            tensor_index_inplace(i, &shape, &mut index).unwrap();

            println!("i={i}, index={index:?}");
            assert_eq!(i, global_index(&index, &shape, None).unwrap());
        }
    }
}

pub(in crate::tensor) fn global_index(
    index: &Vec<usize>,
    shape: &[usize],
    offset: Option<&Vec<SliceRange>>,
) -> Result<usize, String> {
    if index.len() < shape.len() {
        // TODO: allow this case as long as extra dims are 1.
        return Err(format!(
            "shapes do not match: self.shape={:?}, index={:?}
                Need index to be at least as long as shape.",
            shape, index,
        ));
    }
    let mut global_idx = 0;
    let mut multiplier = 1;
    // TODO: consider turning this into a fold operation.
    for (i, (&dim, &idx_dim)) in shape.iter().rev().zip(index.iter().rev()).enumerate() {
        // Fix the indexing.  We need to have all the reverses index may be shorter than shape.
        let i = shape.len() - i - 1;
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
                    shape,
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
                    shape,
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

// #[test]
// fn test_global_index() {
//     let m = 4;
//     let n = 4;
//     for i in 0..m {
//         for j in 0..n {
//             let idx = global_index(&vec![i, j], &vec![4, 4], None).unwrap();
//             assert!(idx[0] <= m, "idx[0]={idx[0]}, m={m}");
//             assert!(idx[1] <= n, "idx[1]={idx[1]}, n={n}");
//         }
//     }
// }

#[test]
fn test_increment_index() {
    let mut index = vec![0, 0, 0];
    let dimensions = vec![2, 3, 2];
    let indices = vec![
        [0, 0, 1].to_vec(),
        [0, 1, 0].to_vec(),
        [0, 1, 1].to_vec(),
        [0, 2, 0].to_vec(),
        [0, 2, 1].to_vec(),
        [1, 0, 0].to_vec(),
        [1, 0, 1].to_vec(),
        [1, 1, 0].to_vec(),
        [1, 1, 1].to_vec(),
        [1, 2, 0].to_vec(),
        [1, 2, 1].to_vec(),
    ];
    for expected_idx in indices.into_iter() {
        let valid = increment_index(&mut index, &dimensions);
        assert!(valid);
        assert_eq!(index, expected_idx);
    }
}

#[test]
fn test_index_iterator() {
    let index_iter = IndexIterator::new(vec![2, 2, 2]);
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
