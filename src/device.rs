use crate::ndarray::NDArray;
use crate::type_trait::Type;

pub trait Device<T: Type>: Clone {
    fn new(data: *mut T, shape: &[usize]) -> NDArray<T, Self>;

    fn ones(shape: &[usize]) -> NDArray<T, Self>;

    fn rand(shape: &[usize], low: T, high: T) -> NDArray<T, Self>;

    fn add(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn add_scalar(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn mul(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>;

    fn mul_scalar(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self>;

    fn eq(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> bool;

    fn sum(
        &self,
        lhs: &NDArray<T, Self>,
        shape: Vec<usize>,
        reduce_dims: usize,
    ) -> NDArray<T, Self>;

    fn contiguous(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>;
}
