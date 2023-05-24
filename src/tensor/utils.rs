use crate::tensor::numeric::Numeric;
use crate::tensor::TensorLike;

pub struct ElementIterator<'b, T, U>
where
    U: for<'a> TensorLike<'a, Elem = T>,
    T: Numeric,
{
    index: Vec<usize>,
    tensor_like: &'b U,
    first: bool,
}

impl<'b, T, U> ElementIterator<'b, T, U>
where
    U: for<'a> TensorLike<'a, Elem = T>,
    T: Numeric,
{
    pub fn new(tensor_like: &'b U) -> ElementIterator<'b, T, U> {
        ElementIterator {
            index: vec![0; tensor_like.shape().len()],
            tensor_like,
            first: true,
        }
    }
}

impl<'b, T: 'b, U> Iterator for ElementIterator<'b, T, U>
where
    U: for<'a> TensorLike<'a, Elem = T>,
    T: Numeric,
{
    type Item = &'b T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.first {
            self.first = false;
            let elem = self.tensor_like.get(&self.index).unwrap();
            return Some(elem);
        }
        if increment_index(&mut self.index, self.tensor_like.shape()) {
            let elem = self.tensor_like.get(&self.index).unwrap();
            return Some(elem);
        }
        None
    }
}

#[test]
fn test_element_iterator() {
    use crate::tensor::{SliceRange, Tensor};
    let v = [1, 2, 3];
    let tensor = Tensor::from(v);
    let view = tensor.view(vec![SliceRange::new(0, 3)]);
    let tensor_element_iterator = ElementIterator::new(&tensor);
    let element_iterator = ElementIterator::new(&view);
    for ((view_elem, tensor_elem), expected) in
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

pub fn increment_index(index: &mut [usize], shape: &[usize]) -> bool {
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
