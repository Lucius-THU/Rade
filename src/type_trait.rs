use num_traits::Num;
use rand::distributions::uniform::SampleUniform;
use std::fmt::Display;
use std::ops::{Add, Index};

pub trait Type: Num + SampleUniform + Copy + Display + Add<Output = Self> {}

impl<T: Num + SampleUniform + Copy + Display + Add<Output = Self>> Type for T {}

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
    27, 28, 29, 30
);
