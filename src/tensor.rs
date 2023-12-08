use crate::device::Device;
use crate::ndarray::NDArray;
use crate::operation::{
    AddScalar, Broadcast, EWiseAdd, EWiseMul, MulScalar, Operation, Reshape, Summation,
};
use crate::type_trait::{Len, Type};
use lazy_static::lazy_static;
use num_traits::{One, Zero};
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::{Arc, RwLock};

lazy_static! {
    pub static ref LAZY_MODE: bool = false;
}

pub(crate) struct Value<'a, T: Type, D: Device<T>> {
    cached_data: Option<NDArray<T, D>>,
    pub inputs: Vec<Tensor<'a, T, D>>,
    op: Option<Box<dyn Operation<'a, T, D> + 'a>>,
    requires_grad: bool,
    grad: Option<Tensor<'a, T, D>>,
}

#[derive(Clone)]
pub struct Tensor<'a, T: Type, D: Device<T>>(pub(crate) Arc<RwLock<Value<'a, T, D>>>);

impl<'a, T: Type, D: Device<T>> Tensor<'a, T, D> {
    pub fn new1d<A: Len<T>>(data: A, requires_grad: bool) -> Self {
        Self::make(
            Some(D::new(data.ptr(), &[data.len()])),
            &[],
            None,
            requires_grad,
        )
    }

    pub fn new2d<A: Len<T>, B: Len<A>>(data: B, requires_grad: bool) -> Self {
        Self::make(
            Some(D::new(data.ptr() as *mut T, &[data.len(), data[0].len()])),
            &[],
            None,
            requires_grad,
        )
    }

    pub fn new3d<A: Len<T>, B: Len<A>, C: Len<B>>(data: C, requires_grad: bool) -> Self {
        Self::make(
            Some(D::new(
                data.ptr() as *mut T,
                &[data.len(), data[0].len(), data[0][0].len()],
            )),
            &[],
            None,
            requires_grad,
        )
    }

    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        Self::make(Some(D::ones(shape)), &[], None, requires_grad)
    }

    pub fn rand(shape: &[usize], low: T, high: T, requires_grad: bool) -> Self {
        Self::make(Some(D::rand(shape, low, high)), &[], None, requires_grad)
    }

    pub(crate) fn make(
        cached_data: Option<NDArray<T, D>>,
        inputs: &[Tensor<'a, T, D>],
        op: Option<Box<dyn Operation<'a, T, D> + 'a>>,
        requires_grad: bool,
    ) -> Self {
        Self(Arc::new(RwLock::new(Value {
            cached_data,
            inputs: inputs.to_vec(),
            op,
            requires_grad,
            grad: None,
        })))
    }

    pub fn detach(&self) -> Self {
        Self::make(Some(self.realize_cached_data()), &[], None, false)
    }

    pub fn shape(&self) -> Vec<usize> {
        self.realize_cached_data().0.shape.to_vec()
    }

    pub fn broadcast(&self, shape: &[usize]) -> Self {
        Self::calc(Broadcast(shape.to_vec()), &[self])
    }

    pub fn sum(&self, axes: Option<Vec<usize>>, keep_dims: bool) -> Self {
        Self::calc(Summation(axes, keep_dims), &[self])
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        Self::calc(Reshape(shape), &[self])
    }

    pub(crate) fn realize_cached_data(&self) -> NDArray<T, D> {
        let mut value = self.0.write().unwrap();
        if value.cached_data.is_none() {
            if let Some(op) = &value.op {
                value.cached_data = Some(
                    op.compute(
                        &value
                            .inputs
                            .iter()
                            .map(|x| x.realize_cached_data())
                            .collect::<Vec<_>>(),
                    ),
                );
            } else {
                panic!("No cached data and no op")
            }
        }
        value.cached_data.clone().unwrap()
    }

    fn calc(op: impl Operation<'a, T, D> + 'a, args: &[&Tensor<'a, T, D>]) -> Self {
        let inputs = args.iter().map(|&x| x.clone()).collect::<Vec<_>>();
        let requires_grad = inputs.iter().any(|x| x.0.read().unwrap().requires_grad);
        let mut output = Self::make(None, &inputs, Some(Box::new(op)), requires_grad);
        if !*LAZY_MODE {
            if !requires_grad {
                output = output.detach();
            } else {
                output.realize_cached_data();
            }
        }
        output
    }
}

impl<'a, T: Type, D: Device<T>> Add for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::calc(EWiseAdd, &[self, rhs])
    }
}

impl<'a, T: Type + 'a, D: Device<T>> Add<T> for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn add(self, rhs: T) -> Self::Output {
        Tensor::calc(AddScalar(rhs), &[self])
    }
}

impl<'a, T: Type, D: Device<T>> Mul for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::calc(EWiseMul, &[self, rhs])
    }
}

impl<'a, T: Type + 'a, D: Device<T>> Mul<T> for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor::calc(MulScalar(rhs), &[self])
    }
}

impl<'a, T: Type, D: Device<T>> Sub for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::calc(EWiseAdd, &[self, &(rhs * (T::zero() - T::one()))])
    }
}

impl<'a, T: Type + 'a, D: Device<T>> Sub<T> for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn sub(self, rhs: T) -> Self::Output {
        Tensor::calc(AddScalar(T::zero() - rhs), &[self])
    }
}

impl<'a, T: Type + 'a, D: Device<T>> Neg for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn neg(self) -> Self::Output {
        Tensor::calc(MulScalar(T::zero() - T::one()), &[self])
    }
}

impl<'a, T: Type + 'a, D: Device<T>> PartialEq for Tensor<'a, T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.realize_cached_data() == other.realize_cached_data()
    }
}

macro_rules! impl_add {
    ($($t:ty),*) => {
        $(
            impl<'a, D: Device<$t>> Add<&Tensor<'a, $t, D>> for $t {
                type Output = Tensor<'a, $t, D>;

                fn add(self, rhs: &Tensor<'a, $t, D>) -> Self::Output {
                    Tensor::calc(AddScalar(self), &[rhs])
                }
            }
        )*
    };
}

macro_rules! impl_mul {
    ($($t:ty),*) => {
        $(
            impl<'a, D: Device<$t>> Mul<&Tensor<'a, $t, D>> for $t {
                type Output = Tensor<'a, $t, D>;

                fn mul(self, rhs: &Tensor<'a, $t, D>) -> Self::Output {
                    Tensor::calc(MulScalar(self), &[rhs])
                }
            }
        )*
    };
}

macro_rules! impl_sub {
    ($($t:ty),*) => {
        $(
            impl<'a, D: Device<$t>> Sub<&Tensor<'a, $t, D>> for $t {
                type Output = Tensor<'a, $t, D>;

                fn sub(self, rhs: &Tensor<'a, $t, D>) -> Self::Output {
                    Tensor::calc(AddScalar(self), &[&(rhs * (<$t as Zero>::zero() - <$t as One>::one()))])
                }
            }
        )*
    };
}

impl_add!(isize, i8, i16, i32, i64, i128, f32, f64);

impl_mul!(isize, i8, i16, i32, i64, i128, f32, f64);

impl_sub!(isize, i8, i16, i32, i64, i128, f32, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CPU;

    #[test]
    fn test_add() {
        let a = Tensor::<f32, CPU>::new1d([1.0, 2.0, 3.0], false);
        let b = Tensor::new2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], false);
        assert!(&a + &b == Tensor::new2d([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]], false));
        assert!(&a + 1.0 == Tensor::new1d([2.0, 3.0, 4.0], false));
        assert!(2.0 + &b == Tensor::new2d([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], false));
    }

    #[test]
    fn test_new() {
        let a = Tensor::<f32, CPU>::new1d([1.0, 2.0, 3.0], false);
        let b = Tensor::<f32, CPU>::new2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], false);
        let c = Tensor::<f32, CPU>::new3d([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], false);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(c.shape(), &[1, 3, 2]);
    }

    #[test]
    fn test_broadcast() {
        let a = Tensor::<f32, CPU>::new1d([1.0, 2.0, 3.0], false);
        let b = Tensor::<f32, CPU>::new2d([[1.0], [4.0]], false);
        assert_eq!(a.broadcast(&[2, 3]).shape(), &[2, 3]);
        assert_eq!(b.broadcast(&[1, 2, 3]).shape(), &[1, 2, 3]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::<f32, CPU>::new1d([1.0, 2.0, 3.0], false);
        let b = Tensor::new2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], false);
        assert!(&a * &b == Tensor::new2d([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]], false));
        assert!(&a * 2.0 == Tensor::new1d([2.0, 4.0, 6.0], false));
        assert!(3.0 * &b == Tensor::new2d([[3.0, 6.0, 9.0], [12.0, 15.0, 18.0]], false));
    }

    #[test]
    fn test_sub() {
        let a = Tensor::<f32, CPU>::new1d([1.0, 2.0, 3.0], false);
        let b = Tensor::new2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], false);
        assert!(&a - &b == Tensor::new2d([[0.0, 0.0, 0.0], [-3.0, -3.0, -3.0]], false));
        assert!(&a - 1.0 == Tensor::new1d([0.0, 1.0, 2.0], false));
        assert!(2.0 - &b == Tensor::new2d([[1.0, 0.0, -1.0], [-2.0, -3.0, -4.0]], false));
    }

    #[test]
    fn test_sum() {
        let a = Tensor::<f32, CPU>::new2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], false);
        assert!(a.sum(None, false) == Tensor::new1d([21.0], false));
        assert!(a.sum(Some(vec![0]), true) == Tensor::new2d([[5.0, 7.0, 9.0]], false));
        assert!(a.sum(Some(vec![1]), false) == Tensor::new1d([6.0, 15.0], false));
        assert!(a.sum(Some(vec![0, 1]), true) == Tensor::new2d([[21.0]], false));
    }

    #[test]
    fn test_reshape() {
        let a = Tensor::<f32, CPU>::new2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], false);
        assert!(
            a.reshape(vec![3, 2]) == Tensor::new2d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], false)
        );
        assert!(a.reshape(vec![6]) == Tensor::new1d([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false));

        let b = Tensor::<f32, CPU>::new1d([1.0, 2.0, 3.0], true).broadcast(&[2, 3]);
        assert!(b.reshape(vec![3, 2]) == Tensor::new2d([[1.0, 2.0], [3.0, 1.0], [2.0, 3.0]], true));
    }
}
