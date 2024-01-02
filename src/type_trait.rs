use crate::device::Device;
use crate::operation::ScalarPow;
use crate::tensor::Tensor;
use bincode::{Decode, Encode};
use num_traits::{Bounded, Num, NumCast, Pow, ToPrimitive};
use rand::distributions::uniform::SampleUniform;
use std::fmt::Display;
use std::ops::{AddAssign, Index, MulAssign};

pub trait Type:
    'static
    + Encode
    + Decode
    + Num
    + Bounded
    + SampleUniform
    + Copy
    + Display
    + PartialOrd
    + AddAssign
    + MulAssign
{
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

impl_type!(usize, u8, u16, u32, u64, isize, i8, i16, i32, i64);

pub trait Len<T>: Index<usize, Output = T> {
    fn ptr(&self) -> *mut T;
    fn len(&self) -> usize;
}

impl<T, const N: usize> Len<T> for [T; N] {
    fn ptr(&self) -> *mut T {
        self.as_ptr() as *mut T
    }

    fn len(&self) -> usize {
        N
    }
}

pub trait Float: Signed + num_traits::Float + Pow<Self, Output = Self> {
    fn powt<D: Device<Self>>(self, rhs: &Tensor<Self, D>) -> Tensor<Self, D> {
        Tensor::calc(ScalarPow(self), vec![rhs.clone()])
    }
}

impl<T: Signed + num_traits::Float + Pow<Self, Output = Self>> Float for T {}

pub trait Signed: Type + num_traits::Signed + ToPrimitive {}

impl<T: Type + num_traits::Signed + ToPrimitive> Signed for T {}

pub trait Unsigned: Type + num_traits::Unsigned + ToPrimitive + NumCast {}

impl<T: Type + num_traits::Unsigned + ToPrimitive + NumCast> Unsigned for T {}
