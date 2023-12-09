use crate::device::Device;
use crate::ndarray::{self, Idx, NDArray, Storage};
use crate::tensor::Tensor;
use crate::type_trait::Type;
use num_traits::{Float, Pow};
use rand::distributions::{Distribution, Uniform};
use std::fmt::{Display, Formatter, Result};
use std::ops::Index;
use std::slice;

#[derive(Clone)]
pub struct CPU;

impl Device for CPU {
    fn new<T: Type>(data: *mut T, shape: &[usize]) -> NDArray<T, Self> {
        let strides = ndarray::compact_strides(shape);
        let len = shape[0] * strides[0];
        NDArray::make(
            Storage::CPU(unsafe { slice::from_raw_parts(data, len).to_vec() }),
            shape.to_vec(),
            strides.to_vec(),
            0,
            Self,
        )
    }

    fn ones<T: Type>(shape: &[usize]) -> NDArray<T, Self> {
        let strides = ndarray::compact_strides(shape);
        NDArray::make(
            Storage::CPU(vec![T::one(); shape[0] * strides[0]]),
            shape.to_vec(),
            strides.to_vec(),
            0,
            Self,
        )
    }

    fn zeros<T: Type>(shape: &[usize]) -> NDArray<T, Self> {
        let strides = ndarray::compact_strides(shape);
        NDArray::make(
            Storage::CPU(vec![T::zero(); shape[0] * strides[0]]),
            shape.to_vec(),
            strides.to_vec(),
            0,
            Self,
        )
    }

    fn rand<T: Type>(shape: &[usize], low: T, high: T) -> NDArray<T, Self> {
        let strides = ndarray::compact_strides(shape);
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(low, high);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        for _ in 0..shape[0] * strides[0] {
            data.push(uniform.sample(&mut rng));
        }
        NDArray::make(
            Storage::CPU(data),
            shape.to_vec(),
            strides.to_vec(),
            0,
            Self,
        )
    }

    fn add<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        let shape = &lhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(lhs[&idx] + rhs[&idx]);
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), idx.shape, strides, 0, Self)
    }

    fn add_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        let shape = &lhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(lhs[&idx] + rhs);
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), idx.shape, strides, 0, Self)
    }

    fn mul<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        let shape = &lhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(lhs[&idx] * rhs[&idx]);
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), idx.shape, strides, 0, Self)
    }

    fn mul_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        let shape = &lhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(lhs[&idx] * rhs);
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), idx.shape, strides, 0, Self)
    }

    fn eq<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> bool {
        let mut idx = Idx::new(&lhs.0.shape);
        loop {
            if (lhs[&idx] - rhs[&idx]).abs() > T::atol() {
                return false;
            }
            if !idx.next() {
                break;
            }
        }
        true
    }

    fn pow<T: Type + Pow<T, Output = T>>(
        &self,
        lhs: &NDArray<T, Self>,
        rhs: &NDArray<T, Self>,
    ) -> NDArray<T, Self> {
        let shape = &lhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(lhs[&idx].pow(rhs[&idx]));
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), idx.shape, strides, 0, Self)
    }

    fn pow_scalar<U: Type, T: Type + Pow<U, Output = T>>(
        &self,
        lhs: &NDArray<T, Self>,
        rhs: U,
    ) -> NDArray<T, Self> {
        let shape = &lhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(lhs[&idx].pow(rhs));
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), idx.shape, strides, 0, Self)
    }

    fn scalar_pow<T: Type + Pow<T, Output = T>>(
        &self,
        lhs: T,
        rhs: &NDArray<T, Self>,
    ) -> NDArray<T, Self> {
        let shape = &rhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(lhs.pow(rhs[&idx]));
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), idx.shape, strides, 0, Self)
    }

    fn ln<T: Type + Float>(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        let shape = &lhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(lhs[&idx].ln());
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), shape.to_vec(), strides, 0, Self)
    }

    fn max_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        let shape = &lhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(if lhs[&idx] > rhs { lhs[&idx] } else { rhs });
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), shape.to_vec(), strides, 0, Self)
    }

    fn gt_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        let shape = &lhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(if lhs[&idx] > rhs { T::one() } else { T::zero() });
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), shape.to_vec(), vec![1], 0, Self)
    }

    fn matmul<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        let len = lhs.ndim();
        let mut axes = (0..len).collect::<Vec<_>>();
        axes.swap(len - 1, len - 2);
        let rhs = &rhs.permute(&axes);
        let mut shape = lhs.0.shape.clone();
        let [n, m, k] = [shape[len - 2], shape[len - 1], rhs.0.shape[len - 2]];
        shape[len - 1] = k;
        let mut idx = Idx::new(&shape);
        let strides = ndarray::compact_strides(&shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut lhs_idx = Idx::new(&lhs.0.shape);
        let mut rhs_idx = Idx::new(&rhs.0.shape);
        loop {
            let mut stop = false;
            for _ in 0..n {
                for _ in 0..k {
                    let mut sum = T::zero();
                    for _ in 0..m {
                        sum = sum + lhs[&lhs_idx] * rhs[&rhs_idx];
                        lhs_idx.next_in_dim(len - 1);
                        rhs_idx.next_in_dim(len - 2);
                    }
                    data.push(sum);
                    if !idx.next() {
                        stop = true;
                    }
                }
                lhs_idx.next_out_dim(len - 1);
            }
            rhs_idx.next_out_dim(len - 2);
            if stop {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), shape, strides, 0, Self)
    }

    fn sum<T: Type>(
        &self,
        lhs: &NDArray<T, Self>,
        shape: Vec<usize>,
        reduce_dims: usize,
    ) -> NDArray<T, Self> {
        let strides = ndarray::compact_strides(&shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(&lhs.0.shape);
        let reduce_lens = idx.shape[reduce_dims..lhs.ndim()].iter().product::<usize>();
        loop {
            let mut sum = T::zero();
            let mut stop = false;
            for _ in 0..reduce_lens {
                sum = sum + lhs[&idx];
                stop = !idx.next();
            }
            data.push(sum);
            if stop {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), shape, strides, 0, Self)
    }

    fn contiguous<T: Type>(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        let shape = &lhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(lhs[&idx]);
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), shape.to_vec(), strides, 0, Self)
    }
}

impl<T: Type> Index<&Idx> for NDArray<T, CPU> {
    type Output = T;

    fn index(&self, index: &Idx) -> &Self::Output {
        let mut idx = 0;
        for (i, &dim) in index.idx.iter().enumerate() {
            idx += dim * self.0.strides[i];
        }
        if let Storage::CPU(data) = &self.0.data.as_ref() {
            &data[idx + self.0.offset]
        } else {
            panic!("Tensor Storage mismatched with Device")
        }
    }
}

impl<'a, T: Type> Display for Tensor<'a, T, CPU> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "CPU(")?;
        let data = &self.realize_cached_data();
        let mut idx = Idx::new(&data.0.shape);
        let mut start = true;
        loop {
            let mut ident = ", ".to_string();
            for &i in idx.idx.iter().rev() {
                if i == 0 {
                    ident = format!("]{}[", ident);
                } else {
                    break;
                }
            }
            if start {
                ident = ident[data.ndim() + 2..].to_string();
                start = false;
            }
            write!(f, "{ident}{}", data[&idx])?;
            if !idx.next() {
                break;
            }
        }
        let mut ending = "".to_string();
        for _ in 0..data.ndim() {
            ending = format!("{}]", ending);
        }
        write!(f, "{ending})")
    }
}
