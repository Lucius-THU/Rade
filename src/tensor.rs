use crate::device::Device;
use crate::ndarray::NDArray;
use crate::operation::{
    AddScalar, Broadcast, EWiseAdd, EWiseMul, MulScalar, Operation, Reshape, Summation,
};
use crate::type_trait::{Len, Type};
use lazy_static::lazy_static;
use std::ops::{Add, Mul};
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
        Self::make(self.0.read().unwrap().cached_data.clone(), &[], None, false)
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

impl_add!(usize, u8, u16, u32, u64, isize, i8, i16, i32, i64, f32, f64);

impl_mul!(usize, u8, u16, u32, u64, isize, i8, i16, i32, i64, f32, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CPU;

    #[test]
    fn test_add() {
        let a = Tensor::<f32, CPU>::rand(&[2, 3], 0.0, 1.0, true);
        let b = Tensor::rand(&[3], 0.0, 1.0, true);
        let c = &(&a + &b).realize_cached_data().0;
        assert_eq!(c.shape, vec![2, 3]);
        assert_eq!(c.strides, vec![3, 1]);

        let d = &(&a + 1.0).realize_cached_data().0;
        let e = &(1.0 + &b).realize_cached_data().0;
        assert_eq!(d.shape, vec![2, 3]);
        assert_eq!(d.strides, vec![3, 1]);
        assert_eq!(e.shape, vec![3]);
        assert_eq!(e.strides, vec![1]);
    }

    #[test]
    fn test_new() {
        let a = Tensor::<f32, CPU>::new1d([1.0, 2.0, 3.0], true);
        let b = Tensor::<f32, CPU>::new2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], true);
        let c = Tensor::<f32, CPU>::new3d([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], true);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(c.shape(), &[1, 3, 2]);
    }

    #[test]
    fn test_broadcast() {
        let a = Tensor::<f32, CPU>::new1d([1.0, 2.0, 3.0], true);
        let b = Tensor::<f32, CPU>::new2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], true);
        let c = Tensor::<f32, CPU>::new3d([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], true);
        let d = Tensor::<f32, CPU>::ones(&[3], true);
        let e = Tensor::<f32, CPU>::ones(&[2, 3], true);
        let f = Tensor::<f32, CPU>::ones(&[1, 3, 1], true);
        assert_eq!(a.broadcast(&[2, 3]).shape(), &[2, 3]);
        assert_eq!(b.broadcast(&[1, 2, 3]).shape(), &[1, 2, 3]);
        assert_eq!(c.broadcast(&[2, 3, 2]).shape(), &[2, 3, 2]);
        assert_eq!(d.broadcast(&[2, 3]).shape(), &[2, 3]);
        assert_eq!(e.broadcast(&[1, 2, 3]).shape(), &[1, 2, 3]);
        assert_eq!(f.broadcast(&[2, 3, 2]).shape(), &[2, 3, 2]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::<f32, CPU>::rand(&[2, 3], 0.0, 1.0, true);
        let b = Tensor::rand(&[3], 0.0, 1.0, true);
        let c = &a * &b;
        assert_eq!(c.shape(), vec![2, 3]);
        assert_eq!(c.realize_cached_data().0.strides, vec![3, 1]);

        let d = &a * 2.0;
        let e = 3.0 * &b;
        assert_eq!(d.shape(), vec![2, 3]);
        assert_eq!(e.shape(), vec![3]);
    }

    #[test]
    fn test_sum() {
        let a = Tensor::<f32, CPU>::new2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], true);
        let ans0 = a.sum(None, false);
        let ans1 = a.sum(Some(vec![0]), false);
        let ans2 = a.sum(Some(vec![1]), false);
        let ans3 = a.sum(Some(vec![0, 1]), true);
        let ans4 = a.sum(Some(vec![0]), true);
        let ans5 = a.sum(Some(vec![1]), true);

        assert_eq!(ans0.shape(), &[1]);
        assert_eq!(ans1.shape(), &[3]);
        assert_eq!(ans2.shape(), &[2]);
        assert_eq!(ans3.shape(), &[1, 1]);
        assert_eq!(ans4.shape(), &[1, 3]);
        assert_eq!(ans5.shape(), &[2, 1]);
    }

    #[test]
    fn test_reshape() {
        let a = Tensor::<f32, CPU>::new2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], true);
        let ans0 = a.reshape(vec![3, 2]);
        let ans1 = a.reshape(vec![1, 6]);
        let ans2 = a.reshape(vec![6, 1]);
        let ans3 = a.reshape(vec![6]);
        assert_eq!(ans0.shape(), &[3, 2]);
        assert_eq!(ans1.shape(), &[1, 6]);
        assert_eq!(ans2.shape(), &[6, 1]);
        assert_eq!(ans3.shape(), &[6]);

        let b = Tensor::<f32, CPU>::new1d([1.0, 2.0, 3.0], true).broadcast(&[2, 3]);
        let ans4 = b.reshape(vec![3, 2]);
        let ans5 = b.reshape(vec![1, 6]);
        assert_eq!(ans4.shape(), &[3, 2]);
        assert_eq!(ans5.shape(), &[1, 6]);

        let ans6 = b.sum(Some(vec![0]), false);
        let ans7 = b.sum(Some(vec![1]), false);
        let ans8 = b.sum(Some(vec![0, 1]), true);
        assert_eq!(ans6.shape(), &[3]);
        assert_eq!(ans7.shape(), &[2]);
        assert_eq!(ans8.shape(), &[1, 1]);
    }
}
