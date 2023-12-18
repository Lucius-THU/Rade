use crate::ndarray::NDArray;
use crate::type_trait::{Float, Type, Unsigned};
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use num_traits::Pow;

pub trait Device: Clone {
    fn new<T: Type>(data: *mut T, shape: &[usize]) -> NDArray<T, Self>;

    fn ones<T: Type>(shape: &[usize]) -> NDArray<T, Self>;

    fn zeros<T: Type>(shape: &[usize]) -> NDArray<T, Self>;

    fn one_hot<T: Type, U: Unsigned>(
        indices: &NDArray<U, Self>,
        num_classes: usize,
    ) -> NDArray<T, Self>;

    fn rand<T: Type>(shape: &[usize], low: T, high: T) -> NDArray<T, Self>;

    fn add<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn sub<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn add_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn scalar_sub<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn mul<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn mul_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn eq<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> bool;

    fn pow<T: Type + Pow<T, Output = T>>(
        &self,
        lhs: &NDArray<T, Self>,
        rhs: &NDArray<T, Self>,
    ) -> NDArray<T, Self>;

    fn pow_scalar<U: Type, T: Type + Pow<U, Output = T>>(
        &self,
        lhs: &NDArray<T, Self>,
        rhs: U,
    ) -> NDArray<T, Self>;

    fn scalar_pow<T: Type + Pow<T, Output = T>>(
        &self,
        lhs: T,
        rhs: &NDArray<T, Self>,
    ) -> NDArray<T, Self>;

    fn div<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn div_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn scalar_div<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn ln<T: Float>(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn sqrt<T: Float>(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn maximum_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn gt_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn matmul<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn sum<T: Type>(
        &self,
        lhs: &NDArray<T, Self>,
        shape: Vec<usize>,
        reduce_dims: usize,
    ) -> NDArray<T, Self>;

    fn max<T: Type>(
        &self,
        lhs: &NDArray<T, Self>,
        shape: Vec<usize>,
        reduce_dims: usize,
    ) -> NDArray<T, Self>;

    fn equal<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn contiguous<T: Type>(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn data<T: Type>(lhs: &NDArray<T, Self>) -> Vec<T>;

    fn encode<T: Type, E: Encoder>(
        encoder: &mut E,
        lhs: &NDArray<T, Self>,
    ) -> Result<(), EncodeError>;

    fn decode<T: Type, D: Decoder>(decoder: &mut D) -> Result<NDArray<T, Self>, DecodeError>;
}
