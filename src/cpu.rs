use crate::device::Device;
use crate::ndarray::{self, Idx, NDArray, Storage};
use crate::tensor::Tensor;
use crate::type_trait::{Type, Unsigned};
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode};
use num_traits::{Float, Pow};
use rand::distributions::{Distribution, Uniform};
use std::cmp::min;
use std::fmt::{self, Display, Formatter};
use std::ops::Index;
use std::slice;

pub(crate) mod ops;

#[derive(Clone)]
pub struct CPU;

impl CPU {
    fn scalar_op<T: Type, U: Type>(
        &self,
        lhs: &NDArray<T, Self>,
        rhs: U,
        op: impl Fn(T, U) -> T,
    ) -> NDArray<T, Self> {
        if let Storage::CPU(lhs_data) = &lhs.0.data.as_ref() {
            let data = lhs_data[lhs.0.offset..]
                .iter()
                .map(|&x| op(x, rhs))
                .collect::<Vec<_>>();
            NDArray::make(
                Storage::CPU(data),
                lhs.0.shape.clone(),
                lhs.0.strides.clone(),
                0,
                Self,
            )
        } else {
            panic!("Tensor Storage mismatched with Device.")
        }
    }
}

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

    fn one_hot<T: Type, U: Unsigned>(
        indices: &NDArray<U, Self>,
        num_classes: usize,
    ) -> NDArray<T, Self> {
        let mut shape = indices.0.shape.clone();
        let strides = ndarray::compact_strides(&shape);
        let mut data = vec![T::zero(); shape[0] * strides[0] * num_classes];
        let mut idx = Idx::new(&shape);
        let mut i = 0;
        loop {
            data[i * num_classes + indices[&idx].to_usize().unwrap()] = T::one();
            i += 1;
            if !idx.next() {
                break;
            }
        }
        shape.push(num_classes);
        let strides = ndarray::compact_strides(&shape);
        NDArray::make(Storage::CPU(data), shape, strides, 0, Self)
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
        self.scalar_op(lhs, rhs, |x, y| x + y)
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
        self.scalar_op(lhs, rhs, |x, y| x * y)
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
        self.scalar_op(lhs, rhs, |x, y| x.pow(y))
    }

    fn scalar_pow<T: Type + Pow<T, Output = T>>(
        &self,
        lhs: T,
        rhs: &NDArray<T, Self>,
    ) -> NDArray<T, Self> {
        self.scalar_op(rhs, lhs, |x, y| y.pow(x))
    }

    fn div<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        let shape = &lhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(lhs[&idx] / rhs[&idx]);
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), idx.shape, strides, 0, Self)
    }

    fn div_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| x / y)
    }

    fn scalar_div<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| y / x)
    }

    fn ln<T: Type + Float>(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        self.scalar_op(lhs, T::zero(), |x, _| x.ln())
    }

    fn maximum_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| if x > y { x } else { y })
    }

    fn gt_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| if x > y { T::one() } else { T::zero() })
    }

    fn matmul<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        let len = lhs.ndim();
        let mut shape = lhs.0.shape.clone();
        let dims = [shape[len - 2], shape[len - 1], rhs.0.shape[len - 1]];
        shape[len - 1] = dims[2];
        let inner_size = dims[0] * dims[2];
        let strides = ndarray::compact_strides(&shape);
        let total = shape[0] * strides[0];
        let mut data = vec![T::zero(); total];
        let mut lhs_idx = Idx::new(&lhs.0.shape);
        let mut rhs_idx = Idx::new(&rhs.0.shape);
        let tile = T::LANES;
        let tiled_dims = dims
            .iter()
            .map(|x| (x + tile - 1) / tile * tile)
            .collect::<Vec<_>>();
        for o in 0..total / inner_size {
            let outer_offset = o * inner_size;
            let mut temp_rhs = vec![T::zero(); tiled_dims[1] * tiled_dims[2]];
            for i in 0..dims[1] {
                for j in 0..dims[2] {
                    let col_tiled = j / tile * tile;
                    let col_rem = j - col_tiled;
                    temp_rhs[col_tiled * tiled_dims[1] + i * tile + col_rem] = rhs[&rhs_idx];
                    rhs_idx.next();
                }
            }
            for m in (0..dims[0]).step_by(tile) {
                let mut temp_lhs = vec![T::zero(); tile * tiled_dims[1]];
                let r = min(tile, dims[0] - m);
                for i in 0..r {
                    for j in 0..dims[1] {
                        let col_tiled = j / tile * tile;
                        let col_rem = j - col_tiled;
                        temp_lhs[(col_tiled + i) * tile + col_rem] = lhs[&lhs_idx];
                        lhs_idx.next();
                    }
                }
                for n in (0..dims[2]).step_by(tile) {
                    let c = min(tile, dims[2] - n);
                    let inner_offset = outer_offset + m * dims[2] + n;
                    for v in (0..dims[1]).step_by(tile) {
                        let t1 = v * tile;
                        let t2 = (v + tile) * tile;
                        let offset = n * tiled_dims[1];
                        let tile_lhs = &temp_lhs[t1..t2];
                        let tile_rhs = &temp_rhs[t1 + offset..t2 + offset];
                        let temp_ans = T::tiled_matmul(tile_lhs, tile_rhs);
                        for i in 0..r {
                            let i_offset = i * tile;
                            for j in 0..c {
                                data[inner_offset + i * dims[2] + j] += temp_ans[i_offset + j];
                            }
                        }
                    }
                }
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

    fn max<T: Type>(
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
            let mut max = lhs[&idx];
            let mut stop = !idx.next();
            for _ in 1..reduce_lens {
                if lhs[&idx] > max {
                    max = lhs[&idx];
                }
                stop = !idx.next();
            }
            data.push(max);
            if stop {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), shape, strides, 0, Self)
    }

    fn equal<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        let shape = &lhs.0.shape;
        let strides = ndarray::compact_strides(shape);
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        let mut idx = Idx::new(shape);
        loop {
            data.push(if lhs[&idx] == rhs[&idx] {
                T::one()
            } else {
                T::zero()
            });
            if !idx.next() {
                break;
            }
        }
        NDArray::make(Storage::CPU(data), idx.shape, strides, 0, Self)
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
        NDArray::make(Storage::CPU(data), shape.clone(), strides, 0, Self)
    }

    fn data<T: Type>(lhs: &NDArray<T, Self>) -> Vec<T> {
        if let Storage::CPU(data) = &lhs.0.data.as_ref() {
            data.clone()
        } else {
            panic!("Tensor Storage mismatched with Device")
        }
    }

    fn encode<T: Type, E: Encoder>(
        encoder: &mut E,
        lhs: &NDArray<T, Self>,
    ) -> Result<(), EncodeError> {
        if let Storage::CPU(data) = &lhs.0.data.as_ref() {
            Encode::encode(data, encoder)?;
            Encode::encode(&lhs.0.shape, encoder)?;
            Ok(())
        } else {
            Err(EncodeError::Other(
                "Storage type does not compatible with CPU.",
            ))
        }
    }

    fn decode<T: Type, D: Decoder>(decoder: &mut D) -> Result<NDArray<T, Self>, DecodeError> {
        let data = Decode::decode(decoder)?;
        let shape: Vec<usize> = Decode::decode(decoder)?;
        let strides = ndarray::compact_strides(&shape);
        Ok(NDArray::make(Storage::CPU(data), shape, strides, 0, Self))
    }
}

impl<T: Type> Index<&Idx> for NDArray<T, CPU> {
    type Output = T;

    fn index(&self, index: &Idx) -> &Self::Output {
        let mut idx = self.0.offset;
        for (dim, stride) in index.idx.iter().zip(&self.0.strides) {
            idx += dim * stride;
        }
        if let Storage::CPU(data) = &self.0.data.as_ref() {
            &data[idx]
        } else {
            panic!("Tensor Storage mismatched with Device.")
        }
    }
}

impl<T: Type> Display for Tensor<T, CPU> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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
