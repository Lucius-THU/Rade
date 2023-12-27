use crate::device::Device;
use crate::type_trait::{Float, Type, Unsigned};
use num_traits::Pow;
use std::ops::{Add, Div, Index, Mul, Sub};
use std::sync::Arc;

pub enum Storage<T> {
    CPU(Vec<T>),
    CUDA(*mut T),
}

pub struct Idx<'a> {
    pub idx: Vec<usize>,
    pub shape: &'a [usize],
    strides: &'a [usize],
}

#[derive(Clone)]
pub struct NDArray<T: Type, D: Device<T>>(pub(crate) Arc<_NDArray<T, D>>);

#[derive(Clone)]
pub(crate) struct _NDArray<T: Type, D: Device<T>> {
    pub data: Arc<Storage<T>>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
    pub device: D,
}

impl<T: Type, D: Device<T>> NDArray<T, D> {
    pub(crate) fn make(
        data: Storage<T>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
        device: D,
    ) -> Self {
        NDArray(Arc::new(_NDArray {
            data: Arc::new(data),
            shape,
            strides,
            offset,
            device,
        }))
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub(crate) fn ndim(&self) -> usize {
        self.0.shape.len()
    }

    pub(crate) fn len(&self) -> usize {
        self.0.shape.iter().product()
    }

    /// Broadcast the `NDArray` to the given `shape`.
    /// **Warning**: This function does **not** check if the `NDArray` can be broadcast to the given `shape`!
    pub fn broadcast(&self, shape: &[usize]) -> Self {
        let mut strides = vec![0; shape.len()];
        let p = shape.len() - self.0.shape.len();
        for (i, &dim) in self.0.shape.iter().enumerate() {
            if shape[i + p] == dim {
                strides[i + p] = self.0.strides[i];
            }
        }
        let mut ret = self.clone();
        Arc::make_mut(&mut ret.0).strides = strides;
        Arc::make_mut(&mut ret.0).shape = shape.to_vec();
        ret
    }

    /// Permute the axes of the `NDArray`.
    /// **Warning**: `axes` must be a permutation of `0..self.ndim()`.
    pub fn permute(&self, axes: &[usize]) -> Self {
        let mut shape = vec![0; axes.len()];
        let mut strides = vec![0; axes.len()];
        for (i, &axis) in axes.iter().enumerate() {
            shape[i] = self.0.shape[axis];
            strides[i] = self.0.strides[axis];
        }
        let mut ret = self.clone();
        Arc::make_mut(&mut ret.0).shape = shape;
        Arc::make_mut(&mut ret.0).strides = strides;
        ret
    }

    fn reduce_axes(&self, axes: &[usize], keep_dims: bool) -> (NDArray<T, D>, Vec<usize>) {
        let mut permutation = axes.to_vec();
        let mut shape = vec![];
        for i in 0..self.ndim() {
            if !axes.contains(&i) {
                permutation.push(i);
                shape.push(self.0.shape[i]);
            } else if keep_dims {
                shape.push(1);
            }
        }
        let mut perm = self.permute(&permutation);
        if shape.is_empty() {
            shape.push(1);
        }
        if perm.ndim() == axes.len() {
            Arc::make_mut(&mut perm.0).shape.push(1);
            Arc::make_mut(&mut perm.0).strides.push(1);
        }
        (perm, shape)
    }

    pub fn sum(&self, axis: Option<Vec<usize>>, keep_dims: bool) -> Self {
        let axis = axis.unwrap_or((0..self.ndim()).collect::<Vec<_>>());
        let (perm, shape) = self.reduce_axes(&axis, keep_dims);
        self.0.device.sum(&perm, shape, axis.len())
    }

    pub fn max(&self, axis: Option<Vec<usize>>, keep_dims: bool) -> Self {
        let axis = axis.unwrap_or((0..self.ndim()).collect::<Vec<_>>());
        let (perm, shape) = self.reduce_axes(&axis, keep_dims);
        self.0.device.max(&perm, shape, axis.len())
    }

    pub fn reshape(&self, shape: &[usize]) -> Self {
        let mut ret = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()
        };
        let strides = compact_strides(shape);
        Arc::make_mut(&mut ret.0).shape = shape.to_vec();
        Arc::make_mut(&mut ret.0).strides = strides;
        ret
    }

    pub fn scalar_sub(&self, rhs: T) -> Self {
        self.0.device.scalar_sub(self, rhs)
    }

    pub fn scalar_div(&self, rhs: T) -> Self {
        self.0.device.scalar_div(self, rhs)
    }

    pub fn max_scalar(&self, rhs: T) -> Self {
        self.0.device.maximum_scalar(self, rhs)
    }

    pub fn gt_scalar(&self, rhs: T) -> Self {
        self.0.device.gt_scalar(self, rhs)
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
        self.0.device.matmul(self, rhs)
    }

    pub fn equal(&self, rhs: &Self) -> Self {
        self.0.device.equal(self, rhs)
    }

    pub fn is_contiguous(&self) -> bool {
        self.0.strides == compact_strides(&self.0.shape)
    }

    pub fn contiguous(&self) -> Self {
        self.0.device.contiguous(self)
    }

    pub fn index<U: Unsigned, F: Device<U>>(&self, index: NDArray<U, F>) -> Self {
        self.0.device.index(self, index)
    }

    pub fn index_rev<U: Unsigned, F: Device<U>>(&self, index: NDArray<U, F>, dim: usize) -> Self {
        self.0.device.index_rev(self, index, dim)
    }

    pub fn split(&self, dim: usize, start: usize, len: usize) -> Self {
        self.0.device.split(self, dim, start, len)
    }

    pub fn cat(args: &[Self], dim: usize, shape: Vec<usize>) -> Self {
        args[0].0.device.cat(args, dim, shape)
    }
}

impl<T: Float, D: Device<T>> NDArray<T, D> {
    pub fn ln(&self) -> Self {
        self.0.device.ln(self)
    }

    pub fn sqrt(&self) -> Self {
        self.0.device.sqrt(self)
    }
}

impl<T: Type + Pow<T, Output = T>, D: Device<T>> NDArray<T, D> {
    pub fn scalar_pow(&self, rhs: T) -> Self {
        self.0.device.scalar_pow(rhs, self)
    }
}

impl<T: Type, D: Device<T>> Add for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn add(self, rhs: Self) -> Self::Output {
        self.0.device.add(self, rhs)
    }
}

impl<T: Type, D: Device<T>> Add<T> for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn add(self, rhs: T) -> Self::Output {
        self.0.device.add_scalar(self, rhs)
    }
}

impl<T: Type, D: Device<T>> Sub for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.0.device.sub(self, rhs)
    }
}

impl<T: Type, D: Device<T>> Mul for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.0.device.mul(self, rhs)
    }
}

impl<T: Type, D: Device<T>> Mul<T> for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn mul(self, rhs: T) -> Self::Output {
        self.0.device.mul_scalar(self, rhs)
    }
}

impl<T: Type, D: Device<T>> PartialEq for NDArray<T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.device.eq(self, other)
    }
}

impl<T: Type + Pow<T, Output = T>, D: Device<T>> Pow<&NDArray<T, D>> for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn pow(self, rhs: &NDArray<T, D>) -> Self::Output {
        self.0.device.pow(self, rhs)
    }
}

impl<U: Type, T: Type + Pow<U, Output = T>, D: Device<T>> Pow<U> for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn pow(self, rhs: U) -> Self::Output {
        self.0.device.pow_scalar(self, rhs)
    }
}

impl<T: Type, D: Device<T>> Div<&NDArray<T, D>> for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn div(self, rhs: &NDArray<T, D>) -> Self::Output {
        self.0.device.div(self, rhs)
    }
}

impl<T: Type, D: Device<T>> Div<T> for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn div(self, rhs: T) -> Self::Output {
        self.0.device.div_scalar(self, rhs)
    }
}

impl<'a, T: Type, D: Device<T>> Index<&Idx<'a>> for NDArray<T, D> {
    type Output = T;

    fn index(&self, index: &Idx) -> &Self::Output {
        if index.idx.len() != self.0.shape.len() {
            panic!("Index length mismatched with Tensor dimension.")
        }
        match self.0.data.as_ref() {
            Storage::CPU(data) => &data[index.get(self.0.offset)],
            _ => panic!("Tensor Storage mismatched with Device."),
        }
    }
}

impl<'a> Idx<'a> {
    pub fn new<T: Type, D: Device<T>>(array: &'a NDArray<T, D>, ndim: usize) -> Self {
        Self {
            idx: vec![0; ndim],
            shape: &array.0.shape[..ndim],
            strides: &array.0.strides[..ndim],
        }
    }

    pub fn get(&self, offset: usize) -> usize {
        self.idx
            .iter()
            .zip(self.strides)
            .fold(offset, |acc, (&idx, &stride)| acc + idx * stride)
    }

    pub fn next(&mut self) -> bool {
        for (id, &dim) in self.idx.iter_mut().zip(self.shape).rev() {
            *id += 1;
            if *id < dim {
                return true;
            } else {
                *id = 0;
            }
        }
        false
    }
}

pub(crate) fn compact_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1;
    for &dim in shape.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }
    strides.reverse();
    strides
}
