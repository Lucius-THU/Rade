use crate::device::Device;
use crate::ndarray::NDArray;
use crate::operation::{
    AddScalar, Broadcast, EWiseAdd, EWiseMul, EWisePow, Equal, GTScalar, Ln, Matmul, Max,
    MaximumScalar, MulScalar, Operation, PowScalar, Reshape, ScalarPow, Summation, Transpose,
};
use crate::type_trait::{Float, Len, Type, Unsigned};
use lazy_static::lazy_static;
use num_traits::{NumCast, One, Pow, Zero};
use std::collections::{HashMap, HashSet};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Arc, Mutex, RwLock};

lazy_static! {
    pub static ref LAZY_MODE: bool = false;
    pub static ref TENSOR_CNT: Mutex<usize> = Mutex::new(0);
}

pub(crate) struct Value<'a, T: Type, D: Device> {
    cached_data: Option<NDArray<T, D>>,
    pub inputs: Vec<Tensor<'a, T, D>>,
    op: Option<Box<dyn Operation<'a, T, D> + 'a>>,
    pub requires_grad: bool,
    grad: Option<Tensor<'a, T, D>>,
}

#[derive(Clone)]
pub struct Tensor<'a, T: Type, D: Device>(pub(crate) Arc<RwLock<Value<'a, T, D>>>, usize);

impl<'a, T: Type, D: Device> Tensor<'a, T, D> {
    pub fn new_with_shape(data: &[T], shape: &[usize], requires_grad: bool) -> Self {
        Self::make(
            Some(D::new(data.as_ptr() as *mut T, shape)),
            vec![],
            None,
            requires_grad,
        )
    }

    pub fn new1d<A: Len<T>>(data: A, requires_grad: bool) -> Self {
        Self::make(
            Some(D::new(data.ptr(), &[data.len()])),
            vec![],
            None,
            requires_grad,
        )
    }

    pub fn new2d<A: Len<T>, B: Len<A>>(data: B, requires_grad: bool) -> Self {
        Self::make(
            Some(D::new(data.ptr() as *mut T, &[data.len(), data[0].len()])),
            vec![],
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
            vec![],
            None,
            requires_grad,
        )
    }

    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        Self::make(Some(D::ones(shape)), vec![], None, requires_grad)
    }

    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        Self::make(Some(D::zeros(shape)), vec![], None, requires_grad)
    }

    pub fn ones_like(&self, requires_grad: bool) -> Self {
        Self::make(Some(D::ones(&self.shape())), vec![], None, requires_grad)
    }

    pub fn zeros_like(&self, requires_grad: bool) -> Self {
        Self::make(Some(D::zeros(&self.shape())), vec![], None, requires_grad)
    }

    pub fn one_hot<U: Unsigned>(
        labels: &Tensor<'a, U, D>,
        num_classes: usize,
        requires_grad: bool,
    ) -> Self {
        Self::make(
            Some(D::one_hot(&labels.realize_cached_data(), num_classes)),
            vec![],
            None,
            requires_grad,
        )
    }

    pub fn rand(shape: &[usize], low: T, high: T, requires_grad: bool) -> Self {
        Self::make(Some(D::rand(shape, low, high)), vec![], None, requires_grad)
    }

    pub(crate) fn make(
        cached_data: Option<NDArray<T, D>>,
        inputs: Vec<Tensor<'a, T, D>>,
        op: Option<Box<dyn Operation<'a, T, D> + 'a>>,
        requires_grad: bool,
    ) -> Self {
        let mut cnt = TENSOR_CNT.lock().unwrap();
        let tensor = Self(
            Arc::new(RwLock::new(Value {
                cached_data,
                inputs,
                op,
                requires_grad,
                grad: None,
            })),
            *cnt,
        );
        *cnt += 1;
        tensor
    }

    pub fn detach(&self, requires_grad: bool) -> Self {
        Self::make(
            Some(self.realize_cached_data()),
            vec![],
            None,
            requires_grad,
        )
    }

    pub fn shape(&self) -> Vec<usize> {
        self.realize_cached_data().shape()
    }

    pub fn grad(&self) -> Option<Self> {
        self.0.read().unwrap().grad.clone()
    }

    pub fn zero_grad(&self) {
        self.0.write().unwrap().grad = None;
    }

    pub fn data(&self) -> Option<NDArray<T, D>> {
        self.0.read().unwrap().cached_data.clone()
    }

    pub fn set_data(&self, data: Tensor<'a, T, D>) {
        self.0.write().unwrap().cached_data = Some(data.realize_cached_data());
    }

    /// TODO: This function is not efficient and elegant.
    pub fn underlying_data(&self) -> Vec<T> {
        D::data(&self.realize_cached_data())
    }

    pub fn backward(&self) {
        let mut visited = HashSet::new();
        let mut stack = vec![];
        let mut grads = HashMap::new();
        grads.insert(self.1, vec![Tensor::ones_like(self, true)]);
        topo_sort(self, &mut visited, &mut stack);
        for node in stack.iter().rev() {
            let grad = grads[&node.1].iter().fold(None, |acc, x| {
                if acc.is_none() {
                    Some(x.clone())
                } else {
                    Some(&acc.unwrap() + x)
                }
            });
            {
                let value = node.0.read().unwrap();
                if let Some(op) = &value.op {
                    let in_grads = op.gradient(grad.as_ref().unwrap(), &node);
                    for (i, input) in value.inputs.iter().enumerate() {
                        grads
                            .entry(input.1)
                            .or_insert_with(|| vec![])
                            .push(in_grads[i].clone());
                    }
                    continue;
                }
            }
            node.0.write().unwrap().grad = grad;
        }
    }

    pub fn broadcast(&self, shape: &[usize]) -> Self {
        Self::calc(Broadcast(shape.to_vec()), vec![self.clone()])
    }

    pub fn sum(&self, axes: Option<Vec<usize>>, keep_dims: bool) -> Self {
        Self::calc(Summation(axes, keep_dims), vec![self.clone()])
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        Self::calc(Reshape(shape), vec![self.clone()])
    }

    pub fn transpose(&self, axes: Option<(usize, usize)>) -> Self {
        Self::calc(Transpose(axes), vec![self.clone()])
    }

    pub fn relu(&self) -> Self {
        Self::calc(MaximumScalar(T::zero()), vec![self.clone()])
    }

    pub fn gt(&self, rhs: T) -> Self {
        Self::calc(GTScalar(rhs), vec![self.clone()])
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
        Self::calc(Matmul, vec![self.clone(), rhs.clone()])
    }

    pub fn equal(&self, rhs: &Self) -> Self {
        Self::calc(Equal, vec![self.clone(), rhs.clone()])
    }

    pub fn max(&self, axes: Option<Vec<usize>>, keep_dims: bool) -> Self {
        Self::calc(Max(axes, keep_dims), vec![self.clone()])
    }

    pub(crate) fn realize_cached_data(&self) -> NDArray<T, D> {
        {
            let value = self.0.read().unwrap();
            if let Some(data) = &value.cached_data {
                return data.clone();
            }
        }
        let mut value = self.0.write().unwrap();
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
        value.cached_data.clone().unwrap()
    }

    fn calc(op: impl Operation<'a, T, D> + 'a, args: Vec<Tensor<'a, T, D>>) -> Self {
        let requires_grad = args.iter().any(|x| x.0.read().unwrap().requires_grad);
        let mut output = Self::make(None, args, Some(Box::new(op)), requires_grad);
        if !*LAZY_MODE {
            if !requires_grad {
                output = output.detach(false);
            } else {
                output.realize_cached_data();
            }
        }
        output
    }
}

impl<'a, T: Float, D: Device> Tensor<'a, T, D> {
    pub fn ln(&self) -> Self {
        Self::calc(Ln, vec![self.clone()])
    }

    pub fn exp(&self) -> Self {
        T::exp(T::one()).powt(self)
    }

    pub fn kaiming_uniform(fan_in: usize, fan_out: usize, requires_grad: bool) -> Self {
        let bound = T::sqrt(T::from(6.).unwrap() / T::from(fan_in).unwrap());
        Self::rand(&[fan_in, fan_out], -bound, bound, requires_grad)
    }
}

impl<'a, T: Type, D: Device> Add for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::calc(EWiseAdd, vec![self.clone(), rhs.clone()])
    }
}

impl<'a, T: Type + 'a, D: Device> Add<T> for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn add(self, rhs: T) -> Self::Output {
        Tensor::calc(AddScalar(rhs), vec![self.clone()])
    }
}

impl<'a, T: Type, D: Device> Mul for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::calc(EWiseMul, vec![self.clone(), rhs.clone()])
    }
}

impl<'a, T: Type + 'a, D: Device> Mul<T> for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor::calc(MulScalar(rhs), vec![self.clone()])
    }
}

impl<'a, T: Type, D: Device> Sub for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &(rhs * (T::zero() - T::one()))
    }
}

impl<'a, T: Type + 'a, D: Device> Sub<T> for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn sub(self, rhs: T) -> Self::Output {
        self + (T::zero() - rhs)
    }
}

impl<'a, T: Float + 'a, D: Device> Div for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn div(self, rhs: Self) -> Self::Output {
        self * &rhs.pow(T::from(-1).unwrap())
    }
}

impl<'a, T: Type + 'a, D: Device> Div<T> for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn div(self, rhs: T) -> Self::Output {
        self * (T::one() / rhs)
    }
}

impl<'a, T: Type + 'a, D: Device> Neg for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn neg(self) -> Self::Output {
        Tensor::calc(MulScalar(T::zero() - T::one()), vec![self.clone()])
    }
}

impl<'a, T: Type + 'a, D: Device> PartialEq for Tensor<'a, T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.realize_cached_data() == other.realize_cached_data()
    }
}

impl<'a, T: Float + 'a, D: Device> Pow<&Tensor<'a, T, D>> for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn pow(self, rhs: &Tensor<'a, T, D>) -> Self::Output {
        Tensor::calc(EWisePow, vec![self.clone(), rhs.clone()])
    }
}

impl<'a, T: Float + 'a, D: Device> Pow<T> for &Tensor<'a, T, D> {
    type Output = Tensor<'a, T, D>;

    fn pow(self, rhs: T) -> Self::Output {
        Tensor::calc(PowScalar(rhs), vec![self.clone()])
    }
}

macro_rules! impl_div {
    ($($t:ty),*) => {
        $(
            impl<'a, D: Device> Div<&Tensor<'a, $t, D>> for $t {
                type Output = Tensor<'a, $t, D>;

                fn div(self, rhs: &Tensor<'a, $t, D>) -> Self::Output {
                    self * &rhs.pow(<$t as NumCast>::from(-1).unwrap())
                }
            }
        )*
    };
}

impl_div!(f32, f64);

macro_rules! impl_pow_scalar {
    ($t:ty, $u:ty) => {
        impl<'a, D: Device> Pow<$u> for &Tensor<'a, $t, D> {
            type Output = Tensor<'a, $t, D>;

            fn pow(self, rhs: $u) -> Self::Output {
                Tensor::calc(PowScalar(rhs), vec![self.clone()])
            }
        }
    };
}

macro_rules! impl_scalar_pow {
    ($($t:ty),*) => {
        $(
            impl Float for $t {
                fn powt<'a, D: Device>(self, rhs: &Tensor<'a, $t, D>) -> Tensor<'a, $t, D> {
                    Tensor::calc(ScalarPow(self), vec![rhs.clone()])
                }
            }
        )*
    };
}

impl_pow_scalar!(f32, i32);
impl_pow_scalar!(f64, i32);
impl_pow_scalar!(isize, u32);
impl_pow_scalar!(i8, u32);
impl_pow_scalar!(i16, u32);
impl_pow_scalar!(i32, u32);
impl_pow_scalar!(i64, u32);
impl_pow_scalar!(i128, u32);

impl_scalar_pow!(f32, f64);

macro_rules! impl_add {
    ($($t:ty),*) => {
        $(
            impl<'a, D: Device> Add<&Tensor<'a, $t, D>> for $t {
                type Output = Tensor<'a, $t, D>;

                fn add(self, rhs: &Tensor<'a, $t, D>) -> Self::Output {
                    Tensor::calc(AddScalar(self), vec![rhs.clone()])
                }
            }
        )*
    };
}

macro_rules! impl_mul {
    ($($t:ty),*) => {
        $(
            impl<'a, D: Device> Mul<&Tensor<'a, $t, D>> for $t {
                type Output = Tensor<'a, $t, D>;

                fn mul(self, rhs: &Tensor<'a, $t, D>) -> Self::Output {
                    Tensor::calc(MulScalar(self), vec![rhs.clone()])
                }
            }
        )*
    };
}

macro_rules! impl_sub {
    ($($t:ty),*) => {
        $(
            impl<'a, D: Device> Sub<&Tensor<'a, $t, D>> for $t {
                type Output = Tensor<'a, $t, D>;

                fn sub(self, rhs: &Tensor<'a, $t, D>) -> Self::Output {
                    self + &(rhs * (<$t as Zero>::zero() - <$t as One>::one()))
                }
            }
        )*
    };
}

impl_add!(isize, i8, i16, i32, i64, i128, f32, f64);

impl_mul!(isize, i8, i16, i32, i64, i128, f32, f64);

impl_sub!(isize, i8, i16, i32, i64, i128, f32, f64);

fn topo_sort<'a, T: Type, D: Device>(
    node: &Tensor<'a, T, D>,
    visited: &mut HashSet<usize>,
    stack: &mut Vec<Tensor<'a, T, D>>,
) {
    if !visited.contains(&node.1) && node.0.read().unwrap().requires_grad {
        visited.insert(node.1);
        for input in &node.0.read().unwrap().inputs {
            topo_sort(input, visited, stack);
        }
        stack.push(node.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CPU;

    #[test]
    fn test_new() {
        let a = Tensor::<f32, CPU>::new2d([[1., 2., 3.], [4., 5., 6.]], false);
        let b = Tensor::<f32, CPU>::kaiming_uniform(2, 3, false);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_add() {
        let a = Tensor::<f32, CPU>::new1d([1., 2., 3.], false);
        let b = Tensor::new2d([[1., 2., 3.], [4., 5., 6.]], false);
        assert!(&a + &b == Tensor::new2d([[2., 4., 6.], [5., 7., 9.]], false));
        assert!(&a + 1. == Tensor::new1d([2., 3., 4.], false));
        assert!(2. + &b == Tensor::new2d([[3., 4., 5.], [6., 7., 8.]], false));
    }

    #[test]
    fn test_mul() {
        let a = Tensor::<f32, CPU>::new1d([1., 2., 3.], false);
        let b = Tensor::new2d([[1., 2., 3.], [4., 5., 6.]], false);
        assert!(&a * &b == Tensor::new2d([[1., 4., 9.], [4., 10., 18.]], false));
        assert!(&a * 2. == Tensor::new1d([2., 4., 6.], false));
        assert!(3. * &b == Tensor::new2d([[3., 6., 9.], [12., 15., 18.]], false));
    }

    #[test]
    fn test_sub() {
        let a = Tensor::<f32, CPU>::new1d([1., 2., 3.], false);
        let b = Tensor::new2d([[1., 2., 3.], [4., 5., 6.]], false);
        assert!(&a - &b == Tensor::new2d([[0., 0., 0.], [-3., -3., -3.]], false));
        assert!(&a - 1. == Tensor::new1d([0., 1., 2.], false));
        assert!(2. - &b == Tensor::new2d([[1., 0., -1.], [-2., -3., -4.]], false));
    }

    #[test]
    fn test_div() {
        let a = Tensor::<f32, CPU>::new1d([1., 2., 3.], false);
        let b = Tensor::new2d([[1., 2., 3.], [4., 5., 6.]], false);
        assert!(&a / &b == Tensor::new2d([[1., 1., 1.], [0.25, 0.4, 0.5]], false));
        assert!(&a / 2. == Tensor::new1d([0.5, 1., 1.5], false));
        assert!(2. / &b == Tensor::new2d([[2., 1., 2. / 3.], [0.5, 0.4, 1. / 3.]], false));
    }

    #[test]
    fn test_neg() {
        let a = Tensor::<f32, CPU>::new1d([1., 2., 3.], false);
        assert!(-&a == Tensor::new1d([-1., -2., -3.], false));
    }

    #[test]
    fn test_pow() {
        let a = Tensor::<f32, CPU>::new1d([1., 2., 3.], false);
        let b = Tensor::new2d([[1., 2., 3.], [4., 5., 6.]], false);
        assert!(a.pow(&b) == Tensor::new2d([[1., 4., 27.], [1., 32., 729.]], false));
        assert!(a.pow(2.) == Tensor::new1d([1., 4., 9.], false));
        assert!(2.0.powt(&b) == Tensor::new2d([[2., 4., 8.], [16., 32., 64.]], false));
    }

    #[test]
    fn test_ln() {
        let a = Tensor::<f32, CPU>::new1d([1., 2., 3.], false);
        assert!(a.ln() == Tensor::new1d([0., std::f32::consts::LN_2, 1.0986123], false));
    }

    #[test]
    fn test_sum() {
        let a = Tensor::<f32, CPU>::new2d([[1., 2., 3.], [4., 5., 6.]], false);
        assert!(a.sum(None, false) == Tensor::new1d([21.], false));
        assert!(a.sum(Some(vec![0]), true) == Tensor::new2d([[5., 7., 9.]], false));
        assert!(a.sum(Some(vec![1]), false) == Tensor::new1d([6., 15.], false));
        assert!(a.sum(Some(vec![0, 1]), true) == Tensor::new2d([[21.]], false));
    }

    #[test]
    fn test_broadcast() {
        let a = Tensor::<f32, CPU>::new1d([1., 2., 3.], false);
        let b = Tensor::<f32, CPU>::new2d([[1.], [4.]], false);
        assert_eq!(a.broadcast(&[2, 3]).shape(), &[2, 3]);
        assert_eq!(b.broadcast(&[1, 2, 3]).shape(), &[1, 2, 3]);
    }

    #[test]
    fn test_reshape() {
        let a = Tensor::<f32, CPU>::new2d([[1., 2., 3.], [4., 5., 6.]], false);
        assert!(a.reshape(vec![3, 2]) == Tensor::new2d([[1., 2.], [3., 4.], [5., 6.]], false));
        assert!(a.reshape(vec![6]) == Tensor::new1d([1., 2., 3., 4., 5., 6.], false));

        let b = Tensor::<f32, CPU>::new1d([1., 2., 3.], false).broadcast(&[2, 3]);
        assert!(b.reshape(vec![3, 2]) == Tensor::new2d([[1., 2.], [3., 1.], [2., 3.]], false));
    }

    #[test]
    fn test_transpose() {
        let a = Tensor::<f32, CPU>::new2d([[1., 2., 3.], [4., 5., 6.]], false);
        assert!(a.transpose(None) == Tensor::new2d([[1., 4.], [2., 5.], [3., 6.]], false));
        assert!(a.transpose(Some((1, 0))) == Tensor::new2d([[1., 4.], [2., 5.], [3., 6.]], false));
    }

    #[test]
    fn test_relu() {
        let a = Tensor::<f32, CPU>::new2d([[1., -2., 3.], [-4., 5., -6.]], false);
        assert!(a.relu() == Tensor::new2d([[1., 0., 3.], [0., 5., 0.]], false));
    }

    #[test]
    fn test_max() {
        let a = Tensor::<f32, CPU>::new2d([[1., -2., 3.], [-4., 5., -6.]], false);
        assert!(a.max(Some(vec![0]), false) == Tensor::new1d([1., 5., 3.], false));
        assert!(a.max(Some(vec![1]), true) == Tensor::new2d([[3.], [5.]], false));
    }

    #[test]
    fn test_matmul_and_backward() {
        let a = Tensor::<f32, CPU>::new2d([[1., 2., 3.], [4., 5., 6.]], true);
        let b = Tensor::new2d([[-1., 2., -3.], [4., -5., 6.]], true);
        let c = Tensor::new3d(
            [
                [[-0.1, -2.4, 3.7], [-4.1, -2.5, 1.8]],
                [[-3.4, -4.3, 2.8], [-1.5, -1.1, 1.2]],
            ],
            true,
        );
        let d = &(&a + &b) * &b;
        let e = &(&d.matmul(&c.transpose(None)) / 85.7)
            .pow(3.)
            .sum(Some(vec![1]), false)
            - 1.2;
        assert!(
            e == Tensor::new2d(
                [[27.7565333, -1.21271657], [5.02653611e-03, -1.11112233]],
                false
            )
        );

        e.backward();
        assert!(
            a.grad().unwrap()
                == Tensor::new2d(
                    [
                        [0.02772277, -0.0672842, -0.07850484],
                        [-0.73267158, 4.88346243, 8.07030603]
                    ],
                    false
                )
        );
        assert!(
            b.grad().unwrap()
                == Tensor::new2d(
                    [
                        [0.02772277, -0.20185259, -0.07850484],
                        [-2.19801474, 4.88346243, 24.21091808]
                    ],
                    false
                )
        );
        assert!(
            c.grad().unwrap()
                == Tensor::new3d(
                    [
                        [
                            [10.5657550, 0.0140563070, 23.7729488],
                            [3.90452972e-04, 0.0152520692, 8.78519186e-04]
                        ],
                        [
                            [1.31348380, 0.0451217215, 2.95533854],
                            [0.224900912, 2.95280060e-03, 0.506027051]
                        ]
                    ],
                    false
                )
        );
    }
}
