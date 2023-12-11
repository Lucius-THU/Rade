use crate::device::Device;
use crate::type_trait::Type;
use num_traits::{Float, Pow};
use std::ops::{Add, Mul};
use std::sync::Arc;

pub enum Storage<T> {
    CPU(Vec<T>),
    CUDA(*mut T),
}

pub struct Idx {
    pub(crate) idx: Vec<usize>,
    pub(crate) shape: Vec<usize>,
}

#[derive(Clone)]
pub struct NDArray<T: Type, D: Device>(pub(crate) Arc<_NDArray<T, D>>);

#[derive(Clone)]
pub(crate) struct _NDArray<T: Type, D: Device> {
    pub data: Arc<Storage<T>>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
    pub device: D,
}

impl<T: Type, D: Device> NDArray<T, D> {
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

    pub(crate) fn shape(&self) -> Vec<usize> {
        self.0.shape.to_vec()
    }

    pub(crate) fn ndim(&self) -> usize {
        self.0.shape.len()
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
        let mut permutation = vec![];
        let mut shape = vec![];
        for i in 0..self.ndim() {
            if !axes.contains(&i) {
                permutation.push(i);
                shape.push(self.0.shape[i]);
            } else if keep_dims {
                shape.push(1);
            }
        }
        if shape.is_empty() {
            shape.push(1);
        }
        permutation.append(&mut axes.to_vec());
        let perm = self.permute(&permutation);
        (perm, shape)
    }

    pub fn sum(&self, axis: Option<Vec<usize>>, keep_dims: bool) -> Self {
        let axis = axis.unwrap_or((0..self.ndim()).collect::<Vec<_>>());
        let (perm, shape) = self.reduce_axes(&axis, keep_dims);
        self.0.device.sum(&perm, shape, self.ndim() - axis.len())
    }

    pub fn max(&self, axis: Option<Vec<usize>>, keep_dims: bool) -> Self {
        let axis = axis.unwrap_or((0..self.ndim()).collect::<Vec<_>>());
        let (perm, shape) = self.reduce_axes(&axis, keep_dims);
        self.0.device.max(&perm, shape, self.ndim() - axis.len())
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

    fn is_contiguous(&self) -> bool {
        self.0.strides == compact_strides(&self.0.shape)
    }

    fn contiguous(&self) -> Self {
        self.0.device.contiguous(self)
    }
}

impl<T: Type + Float, D: Device> NDArray<T, D> {
    pub fn ln(&self) -> Self {
        self.0.device.ln(self)
    }
}

impl<T: Type + Pow<T, Output = T>, D: Device> NDArray<T, D> {
    pub fn scalar_pow(&self, rhs: T) -> Self {
        self.0.device.scalar_pow(rhs, self)
    }
}

impl<T: Type, D: Device> Add for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn add(self, rhs: Self) -> Self::Output {
        self.0.device.add(self, rhs)
    }
}

impl<T: Type, D: Device> Add<T> for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn add(self, rhs: T) -> Self::Output {
        self.0.device.add_scalar(self, rhs)
    }
}

impl<T: Type, D: Device> Mul for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.0.device.mul(self, rhs)
    }
}

impl<T: Type, D: Device> Mul<T> for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn mul(self, rhs: T) -> Self::Output {
        self.0.device.mul_scalar(self, rhs)
    }
}

impl<T: Type, D: Device> PartialEq for NDArray<T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.device.eq(self, other)
    }
}

impl<T: Type + Pow<T, Output = T>, D: Device> Pow<&NDArray<T, D>> for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn pow(self, rhs: &NDArray<T, D>) -> Self::Output {
        self.0.device.pow(self, rhs)
    }
}

impl<U: Type, T: Type + Pow<U, Output = T>, D: Device> Pow<U> for &NDArray<T, D> {
    type Output = NDArray<T, D>;

    fn pow(self, rhs: U) -> Self::Output {
        self.0.device.pow_scalar(self, rhs)
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

impl Idx {
    pub fn new(shape: &[usize]) -> Self {
        Self {
            idx: vec![0; shape.len()],
            shape: shape.to_vec(),
        }
    }

    pub fn next(&mut self) -> bool {
        for (i, &dim) in self.shape.iter().enumerate().rev() {
            self.idx[i] += 1;
            if self.idx[i] == dim {
                self.idx[i] = 0;
            } else {
                return true;
            }
        }
        false
    }

    pub fn next_in_dim(&mut self, dim: usize) -> bool {
        for (i, &d) in self.shape[dim..].iter().enumerate().rev() {
            self.idx[i + dim] += 1;
            if self.idx[i + dim] == d {
                self.idx[i + dim] = 0;
            } else {
                return true;
            }
        }
        false
    }

    pub fn next_out_dim(&mut self, dim: usize) -> bool {
        for (i, &d) in self.shape[..dim].iter().enumerate().rev() {
            self.idx[i] += 1;
            if self.idx[i] == d {
                self.idx[i] = 0;
            } else {
                return true;
            }
        }
        false
    }
}
