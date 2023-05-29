use rust_light::tensor::*;

#[test]
fn test_slicing() {
    let tensor1 = RawTensor::from(vec![vec![0, 1, 2], vec![3, 4, 5]]);
    let tensor2 = RawTensor::from(vec![vec![1, 2], vec![4, 5]]);
    println!(
        "{:?}\n{:?}",
        tensor1.slice(vec![SliceRange::new(0, 2), SliceRange::new(1, 2),]),
        tensor2.slice(vec![SliceRange::new(0, 2), SliceRange::new(0, 2),])
    );
    assert_eq!(
        tensor1.slice(vec![SliceRange::new(0, 2), SliceRange::new(1, 3),]),
        tensor2.slice(vec![SliceRange::new(0, 2), SliceRange::new(0, 2),])
    );

    // assert_eq!(tensor1.slice(vec![.., 1..]), tensor2.slice(vec![.., ..]));
    // let slice0 = tensor1.slice(vec![SliceRange::new(0, 2), SliceRange::new(1, 2)]);
    let slice1 = tensor1.slice(vec![SliceRange::new(0, 2), SliceRange::new(1, 3)]);
    let slice2 = tensor1.slice(vec![SliceRange::new(0, 2), SliceRange::new(0, 3)]);

    println!("Slice: {:?}", slice1);
    println!("Slice[&vec![0,0,0]]= {:?}", slice1[&vec![0, 0]]);
    assert_eq!(slice1[&vec![0, 0]], 1);
    assert_ne!(slice1, slice2);

    let tensor = RawTensor::new((0..32).collect(), vec![2, 4, 4]);
    let slice = tensor.slice(vec![
        SliceRange::new(1, 2),
        SliceRange::new(2, 3),
        SliceRange::new(1, 4),
    ]);
    assert_eq!(slice.shape(), &vec![1, 1, 3]);
    assert_eq!(
        slice,
        RawTensor::from([[[25, 26, 27].to_vec()].to_vec()].to_vec())
    );
}

#[test]
fn test_from_array() {
    let tensor1 = RawTensor::from([[0, 1, 2], [3, 4, 5]]);
    let tensor2 = RawTensor::new((0..6).collect(), vec![2, 3]);
    assert_eq!(tensor1, tensor2);
}

#[test]
fn test_from_vec() {
    let tensor1 = RawTensor::from(vec![vec![0, 1, 2], vec![3, 4, 5]]);
    let tensor2 = RawTensor::new((0..6).collect(), vec![2, 3]);
    assert_eq!(tensor1, tensor2);
}

// #[ignore]
#[test]
fn test_new_with_filler() {
    let vec = RawTensor::new_with_filler(vec![4], 4);
    let shape = vec.shape();
    assert_eq!(shape, &vec![4]);
    assert_eq!(vec.get(&vec![0]).unwrap(), &4);
}

// #[ignore]
#[test]
fn test_get_2x2x2() {
    let matrix = RawTensor::new(vec![0, 1, 2, 3, 4, 5, 6, 7], vec![2, 2, 2]);
    assert_eq!(*matrix.get(&vec![0, 0, 0]).unwrap(), 0);
    assert_eq!(*matrix.get(&vec![0, 1, 0]).unwrap(), 2);
    assert_eq!(*matrix.get(&vec![1, 1, 1]).unwrap(), 7);
}

// #[ignore]
#[test]
fn test_get_3x3x4() {
    let matrix = RawTensor::new((0..(3 * 3 * 4)).collect(), vec![3, 3, 4]);
    assert_eq!(*matrix.get(&vec![0, 0, 0]).unwrap(), 0);
    assert_eq!(*matrix.get(&vec![2, 2, 3]).unwrap(), 3 * 3 * 4 - 1);
}

// #[ignore]
#[test]
fn test_get_3x3() {
    let matrix = RawTensor::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], vec![3, 3]);
    let mut prev = -1;
    for i in 0..3 {
        for j in 0..3 {
            let &curr = matrix.get(&vec![i, j]).unwrap();
            println!(
                "prev={prev}, matrix[{i}][{j}]={}",
                matrix.get(&vec![i, j]).unwrap()
            );
            assert_eq!(prev + 1, curr);
            prev = curr;
        }
    }
    assert_eq!(matrix.get(&vec![0, 0]).unwrap(), &0);
    assert_eq!(matrix.get(&vec![0, 1]).unwrap(), &1);
    assert_eq!(matrix.get(&vec![1, 0]).unwrap(), &3);
    assert_eq!(matrix.get(&vec![2, 2]).unwrap(), &8);
}

// #[ignore]
#[test]
fn test_add_scalar() {
    let val = 42;
    let tensor1 = RawTensor::new((0..32).collect(), vec![2, 4, 4]);
    let tensor2 = RawTensor::new((42..(32 + 42)).collect(), vec![2, 4, 4]);
    let scalar = RawTensor::scalar(val);
    assert_eq!(&tensor1 + &scalar, tensor2);
    assert_eq!(&scalar + &tensor1, tensor2);
    // assert_eq!(scalar, Tensor::from(val));
    // assert_eq!(&tensor1 + val, tensor2); // Need to figure out multiple implementations first
}

// #[ignore]
#[test]
fn test_add() {
    let tensor1 = RawTensor::new_with_filler(vec![4, 4], 1);
    let tensor2 = RawTensor::new((0..32).collect(), vec![2, 4, 4]);
    let tensor3 = RawTensor::new((1..33).collect(), vec![2, 4, 4]);
    assert_eq!(&tensor2 + &tensor1, tensor3);
    assert_eq!(&tensor1 + &tensor2, tensor3);
}

// #[ignore]
#[test]
fn test_dot() {
    let v = vec![0, 1, 2];
    let vec = RawTensor::new(v, vec![3]);
    assert_eq!(vec.dot(&vec), RawTensor::new(vec![5], vec![1]));
}

#[test]
fn test_creation() {
    let v = vec![0, 1, 2, 3];
    let matrix = RawTensor::new(v, vec![2, 2]);

    format!("Matrix: \n{:?}", matrix);

    assert_eq!(matrix.get(&vec![0, 0]).unwrap(), &0);
}

// #[ignore]
#[test]
fn test_bmm_2x2() {
    let v = vec![0, 1, 2, 3];
    let matrix = RawTensor::new(v, vec![2, 2]); // [[0,1],[2,3]]
    let shape = vec![2, 1];
    let e1 = RawTensor::new(vec![0, 1], vec![2, 1]);
    let e2 = RawTensor::new(vec![1, 0], vec![2, 1]);
    let diag = RawTensor::new(vec![1, 1], vec![2, 1]);

    let r = matrix.bmm(&diag);
    assert_eq!(r.shape(), &shape);
    assert_eq!(r, RawTensor::new(vec![1, 5], shape.clone()));
    assert_eq!(matrix.bmm(&e1), RawTensor::new(vec![1, 3], shape.clone()));
    assert_eq!(matrix.bmm(&e2), RawTensor::new(vec![0, 2], shape.clone()));
}

// #[ignore]
#[test]
fn test_right_scalar_multiplication() {
    let vec = RawTensor::new_with_filler(vec![4], 1);
    assert_eq!(
        vec.right_scalar_multiplication(&42),
        RawTensor::new(vec![42, 42, 42, 42], vec![4])
    );
}
