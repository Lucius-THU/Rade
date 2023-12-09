use crate::ndarray::NDArray;
use crate::type_trait::Type;
use num_traits::{Float, Pow};

pub trait Device: Clone {
    fn new<T: Type>(data: *mut T, shape: &[usize]) -> NDArray<T, Self>;

    fn ones<T: Type>(shape: &[usize]) -> NDArray<T, Self>;

    fn zeros<T: Type>(shape: &[usize]) -> NDArray<T, Self>;

    fn rand<T: Type>(shape: &[usize], low: T, high: T) -> NDArray<T, Self>;

    fn add<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn add_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

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

    fn ln<T: Type + Float>(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn max_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn gt_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn matmul<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn sum<T: Type>(
        &self,
        lhs: &NDArray<T, Self>,
        shape: Vec<usize>,
        reduce_dims: usize,
    ) -> NDArray<T, Self>;

    fn contiguous<T: Type>(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>;
}
