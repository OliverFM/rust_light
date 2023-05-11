use super::*;
#[test]
fn test_new_with_filler() {
    let vec = Tensor::new_with_filler(vec![4], 4);
    assert_eq!(vec.shape(), &vec![4]);
    assert_eq!(vec.get(&vec![0]).unwrap(), &4);
}

#[test]
fn test_get_2x2x2() {
    let matrix = Tensor::new(vec![0, 1, 2, 3, 4, 5, 6, 7], vec![2, 2, 2]);
    assert_eq!(*matrix.get(&vec![0, 0, 0]).unwrap(), 0);
    assert_eq!(*matrix.get(&vec![0, 1, 0]).unwrap(), 2);
    assert_eq!(*matrix.get(&vec![1, 1, 1]).unwrap(), 7);
}

#[test]
fn test_get_3x3x4() {
    let matrix = Tensor::new((0..(3 * 3 * 4)).collect(), vec![3, 3, 4]);
    assert_eq!(*matrix.get(&vec![0, 0, 0]).unwrap(), 0);
    assert_eq!(*matrix.get(&vec![2, 2, 3]).unwrap(), 3 * 3 * 4 - 1);
}

#[test]
fn test_get_3x3() {
    let matrix = Tensor::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], vec![3, 3]);
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

#[test]
fn test_dot() {
    let v = vec![0, 1, 2];
    let vec = Tensor::new(v, vec![3]);
    assert_eq!(vec.dot(&vec), 5);
}
#[test]
fn test_creation() {
    let v = vec![0, 1, 2, 3];
    let matrix = Tensor::new(v, vec![2, 2]);

    format!("Matrix: \n{:?}", matrix);

    assert_eq!(matrix.get(&vec![0, 0]).unwrap(), &0);
}

#[test]
fn test_bmm_2x2() {
    let v = vec![0, 1, 2, 3];
    let matrix = Tensor::new(v, vec![2, 2]);
    let shape = vec![2, 1];
    let e1 = Tensor::new(vec![0, 1], vec![2, 1]);
    let e2 = Tensor::new(vec![1, 0], vec![2, 1]);
    let diag = Tensor::new(vec![1, 1], vec![2, 1]);

    let r = matrix.bmm(&diag);
    assert_eq!(r.shape(), &shape);
    assert_eq!(r, Tensor::new(vec![1, 5], shape.clone()));
    assert_eq!(matrix.bmm(&e1), Tensor::new(vec![1, 3], shape.clone()));
    assert_eq!(matrix.bmm(&e2), Tensor::new(vec![0, 2], shape.clone()));
}
