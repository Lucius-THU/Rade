use crate::ndarray::NDArray;
use crate::type_trait::{Float, Type, Unsigned};
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use num_traits::Pow;
use rand_distr::{Distribution, StandardNormal};

pub trait Device<T: Type>: Clone {
    fn new(data: *mut T, shape: &[usize]) -> NDArray<T, Self>;

    fn ones(shape: &[usize]) -> NDArray<T, Self>;

    fn zeros(shape: &[usize]) -> NDArray<T, Self>;

    fn arange(start: T, end: T, step: T) -> NDArray<T, Self>;

    fn one_hot<U: Unsigned, D: Device<U>>(
        indices: &NDArray<U, D>,
        num_classes: usize,
    ) -> NDArray<T, Self>;

    fn rand(shape: &[usize], low: T, high: T) -> NDArray<T, Self>;

    fn randn(shape: &[usize], mean: T, std: T) -> NDArray<T, Self>
    where
        T: Float,
        StandardNormal: Distribution<T>;

    fn add(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn sub(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn add_scalar(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn scalar_sub(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn mul(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn mul_scalar(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn eq(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> bool;

    fn pow(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>
    where
        T: Pow<T, Output = T>;

    fn pow_scalar<U: Type>(&self, lhs: &NDArray<T, Self>, rhs: U) -> NDArray<T, Self>
    where
        T: Pow<U, Output = T>;

    fn scalar_pow(&self, lhs: T, rhs: &NDArray<T, Self>) -> NDArray<T, Self>
    where
        T: Pow<T, Output = T>;

    fn div(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn div_scalar(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn scalar_div(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn ln(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>
    where
        T: Float;

    fn sqrt(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>
    where
        T: Float;

    fn maximum_scalar(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn gt_scalar(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn matmul(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn sum(&self, lhs: &NDArray<T, Self>, shape: Vec<usize>, dims: usize) -> NDArray<T, Self>;

    fn max(&self, lhs: &NDArray<T, Self>, shape: Vec<usize>, dims: usize) -> NDArray<T, Self>;

    fn equal(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn contiguous(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn index<U: Unsigned, F: Device<U>>(
        &self,
        lhs: &NDArray<T, Self>,
        index: NDArray<U, F>,
    ) -> NDArray<T, Self>;

    fn index_rev<U: Unsigned, F: Device<U>>(
        &self,
        lhs: &NDArray<T, Self>,
        index: NDArray<U, F>,
        dim: usize,
    ) -> NDArray<T, Self>;

    fn cat(&self, args: &[NDArray<T, Self>], dim: usize, shape: Vec<usize>) -> NDArray<T, Self>;

    fn sin(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>
    where
        T: Float;

    fn cos(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>
    where
        T: Float;

    fn data(lhs: &NDArray<T, Self>) -> Vec<T>;

    fn encode<E: Encoder>(encoder: &mut E, lhs: &NDArray<T, Self>) -> Result<(), EncodeError>;

    fn decode<D: Decoder>(decoder: &mut D) -> Result<NDArray<T, Self>, DecodeError>;
}
