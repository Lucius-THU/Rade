use crate::device::Device;
use crate::tensor::Tensor;
use num_traits::{Num, Pow, ToPrimitive};
use rand::distributions::uniform::SampleUniform;
use std::fmt::Display;
use std::ops::{Add, Index};

pub trait Type: Num + SampleUniform + Copy + Display + PartialOrd + Add<Output = Self> {
    fn atol() -> Self {
        Self::zero()
    }

    fn abs(&self) -> Self {
        if *self < Self::zero() {
            Self::zero() - *self
        } else {
            *self
        }
    }
}

impl Type for f32 {
    fn atol() -> Self {
        1e-5
    }
}

impl Type for f64 {
    fn atol() -> Self {
        1e-8
    }
}

macro_rules! impl_type {
    ($($t:ty),*) => {
        $(
            impl Type for $t {}
        )*
    };
}

impl_type!(usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128);

pub trait Len<T>: Index<usize, Output = T> {
    fn ptr(&self) -> *mut T;
    fn len(&self) -> usize;
}

macro_rules! impl_len {
    ($($len:expr),*) => {
        $(
            impl<T> Len<T> for [T; $len] {
                fn ptr(&self) -> *mut T {
                    self.as_ptr() as *mut T
                }

                fn len(&self) -> usize {
                    $len
                }
            }
        )*
    };
}

impl_len!(
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
);

pub trait Float: Type + num_traits::Float + Pow<Self, Output = Self> {
    fn powt<'a, D: Device>(self, rhs: &Tensor<'a, Self, D>) -> Tensor<'a, Self, D>;
}

pub trait Unsigned: Type + num_traits::Unsigned + ToPrimitive {}

impl<T: Type + num_traits::Unsigned + ToPrimitive> Unsigned for T {}
