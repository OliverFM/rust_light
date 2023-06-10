use std::cmp::PartialEq;
pub use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use num::traits::real::Real;
pub use num::{One, Zero};

pub trait FpNumeric: Numeric + Real + From<f32> {}

pub trait Numeric:
    Add
    + AddAssign
    + Copy
    + Clone
    + One
    + Mul
    + Sub
    + SubAssign
    + PartialEq
    + PartialOrd
    + Zero
    + std::fmt::Debug
{
}
// https://stackoverflow.com/questions/42381185/specifying-generic-parameter-to-belong-to-a-small-set-of-types
macro_rules! numeric_impl {
    ($($t: ty),+) => {
        $(
            impl Numeric for $t {}
        )+
    }
}

numeric_impl!(usize, u8, u32, u64, u128, i8, i32, i64, i128, f32, f64);
