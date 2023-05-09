use rust_light::Tensor;

fn main() {
    let v = vec![0, 1, 2, 3];
    let matrix = Tensor::new(v, vec![2, 2]);

    println!("Hello, world!");
    println!("Matrix: \n{:?}", matrix);
}
