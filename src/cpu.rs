use crate::cpu::ops::CPUType;
use crate::device::Device;
use crate::ndarray::{self, Idx, NDArray, Storage};
use crate::tensor::Tensor;
use crate::type_trait::{Float, Type, Unsigned};
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode};
use num_traits::Pow;
use rand_distr::{Distribution, Normal, StandardNormal, Uniform};
use std::fmt::{self, Display, Formatter};
use std::slice;

pub(crate) mod ops;

#[derive(Clone)]
pub struct CPU;

impl CPU {
    fn scalar_op<T: CPUType, U: Type>(
        &self,
        lhs: &NDArray<T, Self>,
        rhs: U,
        op: impl Fn(T, U) -> T,
    ) -> NDArray<T, Self> {
        if let Storage::CPU(lhs_data) = lhs.0.data.as_ref() {
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

    fn ewise_op<T: CPUType>(
        &self,
        lhs: &NDArray<T, Self>,
        rhs: &NDArray<T, Self>,
        op: impl Fn(T, T) -> T,
    ) -> NDArray<T, Self> {
        if !lhs.is_contiguous() {
            return self.ewise_op(&CPU.contiguous(lhs), rhs, op);
        }
        if !rhs.is_contiguous() {
            return self.ewise_op(lhs, &CPU.contiguous(rhs), op);
        }
        if let (Storage::CPU(lhs_data), Storage::CPU(rhs_data)) =
            (lhs.0.data.as_ref(), rhs.0.data.as_ref())
        {
            let len = lhs.len();
            let data = lhs_data[lhs.0.offset..lhs.0.offset + len]
                .iter()
                .zip(&rhs_data[rhs.0.offset..rhs.0.offset + len])
                .map(|(&x, &y)| op(x, y))
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

    fn reduce_op<T: CPUType>(
        &self,
        lhs: &NDArray<T, Self>,
        shape: Vec<usize>,
        dims: usize,
        init: T,
        op: impl Fn(T, T) -> T,
    ) -> NDArray<T, Self> {
        if let Storage::CPU(lhs_data) = lhs.0.data.as_ref() {
            let strides = ndarray::compact_strides(&shape);
            let len = shape[0] * strides[0];
            let mut last_offset = lhs.len();
            let reduce_lens = last_offset / len;
            let mut data = vec![init; len];
            let mut idx = Idx::new(lhs, dims);
            let mut temp = vec![T::zero(); len];
            for _ in 0..reduce_lens {
                let offset = idx.get(lhs.0.offset);
                if offset != last_offset {
                    compact(
                        &mut idx.idx,
                        &lhs.0.shape,
                        &lhs.0.strides,
                        &mut temp,
                        &lhs_data[offset..],
                        dims,
                        &mut 0,
                        0,
                    );
                    last_offset = offset;
                }
                for i in 0..len {
                    data[i] = op(data[i], temp[i]);
                }
                idx.next();
            }
            NDArray::make(Storage::CPU(data), shape, strides, 0, Self)
        } else {
            panic!("Tensor Storage mismatched with Device.")
        }
    }
}

impl<T: CPUType> Device<T> for CPU {
    fn new(data: *mut T, shape: &[usize]) -> NDArray<T, Self> {
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

    fn ones(shape: &[usize]) -> NDArray<T, Self> {
        let strides = ndarray::compact_strides(shape);
        NDArray::make(
            Storage::CPU(vec![T::one(); shape[0] * strides[0]]),
            shape.to_vec(),
            strides.to_vec(),
            0,
            Self,
        )
    }

    fn zeros(shape: &[usize]) -> NDArray<T, Self> {
        let strides = ndarray::compact_strides(shape);
        NDArray::make(
            Storage::CPU(vec![T::zero(); shape[0] * strides[0]]),
            shape.to_vec(),
            strides.to_vec(),
            0,
            Self,
        )
    }

    fn one_hot<U: Unsigned, D: Device<U>>(
        indices: &NDArray<U, D>,
        num_classes: usize,
    ) -> NDArray<T, Self> {
        let len = indices.len();
        let mut data = vec![T::zero(); len * num_classes];
        let mut idx = Idx::new(indices, indices.ndim());
        for i in 0..len {
            data[i * num_classes + indices[&idx].to_usize().unwrap()] = T::one();
            idx.next();
        }
        let mut shape = idx.shape.to_vec();
        shape.push(num_classes);
        let strides = ndarray::compact_strides(&shape);
        NDArray::make(Storage::CPU(data), shape, strides, 0, Self)
    }

    fn rand(shape: &[usize], low: T, high: T) -> NDArray<T, Self> {
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

    fn randn(shape: &[usize], mean: T, std: T) -> NDArray<T, Self> where T: Float, StandardNormal: Distribution<T> {
        let strides = ndarray::compact_strides(shape);
        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean, std).unwrap();
        let mut data = Vec::with_capacity(shape[0] * strides[0]);
        for _ in 0..shape[0] * strides[0] {
            data.push(normal.sample(&mut rng));
        }
        NDArray::make(
            Storage::CPU(data),
            shape.to_vec(),
            strides.to_vec(),
            0,
            Self,
        )
    }

    fn add(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        self.ewise_op(lhs, rhs, |x, y| x + y)
    }

    fn sub(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        self.ewise_op(lhs, rhs, |x, y| x - y)
    }

    fn add_scalar(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| x + y)
    }

    fn scalar_sub(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| y - x)
    }

    fn mul(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        self.ewise_op(lhs, rhs, |x, y| x * y)
    }

    fn mul_scalar(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| x * y)
    }

    fn eq(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> bool {
        if !lhs.is_contiguous() {
            return self.eq(&lhs.contiguous(), rhs);
        }
        if !rhs.is_contiguous() {
            return self.eq(lhs, &rhs.contiguous());
        }
        if let (Storage::CPU(lhs_data), Storage::CPU(rhs_data)) =
            (lhs.0.data.as_ref(), rhs.0.data.as_ref())
        {
            let len = lhs.len();
            lhs_data[lhs.0.offset..lhs.0.offset + len]
                .iter()
                .zip(&rhs_data[rhs.0.offset..rhs.0.offset + len])
                .all(|(&x, &y)| (x - y).abs() < T::atol())
        } else {
            panic!("Tensor Storage mismatched with Device.")
        }
    }

    fn pow(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self>
    where
        T: Pow<T, Output = T>,
    {
        self.ewise_op(lhs, rhs, |x, y| x.pow(y))
    }

    fn pow_scalar<U: Type>(&self, lhs: &NDArray<T, Self>, rhs: U) -> NDArray<T, Self>
    where
        T: Pow<U, Output = T>,
    {
        self.scalar_op(lhs, rhs, |x, y| x.pow(y))
    }

    fn scalar_pow(&self, lhs: T, rhs: &NDArray<T, Self>) -> NDArray<T, Self>
    where
        T: Pow<T, Output = T>,
    {
        self.scalar_op(rhs, lhs, |x, y| y.pow(x))
    }

    fn div(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        self.ewise_op(lhs, rhs, |x, y| x / y)
    }

    fn div_scalar(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| x / y)
    }

    fn scalar_div(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| y / x)
    }

    fn ln(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>
    where
        T: Float,
    {
        self.scalar_op(lhs, T::zero(), |x, _| x.ln())
    }

    fn sqrt(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self>
    where
        T: Float,
    {
        self.scalar_op(lhs, T::zero(), |x, _| x.sqrt())
    }

    fn maximum_scalar(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| if x > y { x } else { y })
    }

    fn gt_scalar(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| if x > y { T::one() } else { T::zero() })
    }

    fn matmul(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        if let (Storage::CPU(lhs_data), Storage::CPU(rhs_data)) =
            (lhs.0.data.as_ref(), rhs.0.data.as_ref())
        {
            let ndim = lhs.ndim();
            let mut shape = lhs.0.shape.clone();
            let dims = [shape[ndim - 2], shape[ndim - 1], rhs.0.shape[ndim - 1]];
            shape[ndim - 1] = dims[2];
            let strides = ndarray::compact_strides(&shape);
            let lhs_idx = Idx::new(lhs, ndim - 1);
            let rhs_idx = Idx::new(rhs, ndim - 1);
            let mut data = vec![T::zero(); shape[0] * strides[0]];
            T::matmul(
                &lhs_data[lhs.0.offset..],
                &rhs_data[rhs.0.offset..],
                &mut data,
                lhs_idx,
                rhs_idx,
                dims,
                [lhs.0.strides[ndim - 1], rhs.0.strides[ndim - 1]],
            );
            NDArray::make(Storage::CPU(data), shape, strides, 0, CPU)
        } else {
            panic!("Tensor Storage mismatched with Device.")
        }
    }

    fn sum(&self, lhs: &NDArray<T, Self>, shape: Vec<usize>, dims: usize) -> NDArray<T, Self> {
        self.reduce_op(lhs, shape, dims, T::zero(), |x, y| x + y)
    }

    fn max(&self, lhs: &NDArray<T, Self>, shape: Vec<usize>, dims: usize) -> NDArray<T, Self> {
        self.reduce_op(
            lhs,
            shape,
            dims,
            T::min_value(),
            |x, y| if x > y { x } else { y },
        )
    }

    fn equal(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        self.ewise_op(lhs, rhs, |x, y| if x == y { T::one() } else { T::zero() })
    }

    fn contiguous(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        if let Storage::CPU(lhs_data) = lhs.0.data.as_ref() {
            let mut data = vec![T::zero(); lhs.len()];
            let shape = lhs.0.shape.clone();
            let strides = ndarray::compact_strides(&shape);
            let mut idx = vec![];
            compact(
                &mut idx,
                &shape,
                &lhs.0.strides,
                &mut data,
                &lhs_data[lhs.0.offset..],
                0,
                &mut 0,
                0,
            );
            NDArray::make(Storage::CPU(data), shape, strides, 0, Self)
        } else {
            panic!("Tensor Storage mismatched with Device.")
        }
    }

    fn index<U: Unsigned, F: Device<U>>(
        &self,
        lhs: &NDArray<T, Self>,
        index: NDArray<U, F>,
    ) -> NDArray<T, Self> {
        if !lhs.is_contiguous() {
            return self.index(&CPU.contiguous(lhs), index);
        }
        if let Storage::CPU(lhs_data) = lhs.0.data.as_ref() {
            let mut shape = index.shape().to_vec();
            let len = lhs.len() / lhs.shape()[0];
            let index_len = index.len();
            let mut data = Vec::with_capacity(index_len * len);
            let mut idx = Idx::new(&index, index.ndim());
            let offset = lhs.0.offset;
            for _ in 0..index_len {
                let i = index[&idx].to_usize().unwrap();
                let l = offset + i * len;
                data.extend(&lhs_data[l..l + len]);
                idx.next();
            }
            shape.extend(&lhs.shape()[1..]);
            let strides = ndarray::compact_strides(&shape);
            NDArray::make(Storage::CPU(data), shape, strides, 0, Self)
        } else {
            panic!("Tensor Storage mismatched with Device.")
        }
    }

    fn index_rev<U: Unsigned, F: Device<U>>(
        &self,
        lhs: &NDArray<T, Self>,
        index: NDArray<U, F>,
        dim: usize,
    ) -> NDArray<T, Self> {
        if !lhs.is_contiguous() {
            return self.index_rev(&CPU.contiguous(lhs), index, dim);
        }
        if let Storage::CPU(lhs_data) = lhs.0.data.as_ref() {
            let mut shape = vec![dim];
            shape.extend(&lhs.shape()[index.ndim()..]);
            let strides = ndarray::compact_strides(&shape);
            let index_len = index.len();
            let len = lhs.len() / index_len;
            let mut data = vec![T::zero(); shape[0] * strides[0]];
            let mut idx = Idx::new(&index, index.ndim());
            let offset = lhs.0.offset;
            for i in 0..index_len {
                let id = index[&idx].to_usize().unwrap();
                let l = offset + i * len;
                data[id * len..(id + 1) * len].iter_mut().zip(
                    &lhs_data[l..l + len],
                ).for_each(|(x, &y)| *x += y);
                idx.next();
            }
            NDArray::make(Storage::CPU(data), shape, strides, 0, Self)
        } else {
            panic!("Tensor Storage mismatched with Device.")
        }
    }

    fn data(lhs: &NDArray<T, Self>) -> Vec<T> {
        if let Storage::CPU(data) = lhs.0.data.as_ref() {
            data.clone()
        } else {
            panic!("Tensor Storage mismatched with Device")
        }
    }

    fn encode<E: Encoder>(encoder: &mut E, lhs: &NDArray<T, Self>) -> Result<(), EncodeError> {
        if let Storage::CPU(data) = lhs.0.data.as_ref() {
            Encode::encode(data, encoder)?;
            Encode::encode(&lhs.0.shape, encoder)?;
            Ok(())
        } else {
            Err(EncodeError::Other(
                "Storage type does not compatible with CPU.",
            ))
        }
    }

    fn decode<D: Decoder>(decoder: &mut D) -> Result<NDArray<T, Self>, DecodeError> {
        let data = Decode::decode(decoder)?;
        let shape: Vec<usize> = Decode::decode(decoder)?;
        let strides = ndarray::compact_strides(&shape);
        Ok(NDArray::make(Storage::CPU(data), shape, strides, 0, Self))
    }
}

impl<T: CPUType> Display for Tensor<T, CPU> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "CPU(")?;
        let data = &self.realize_cached_data();
        let mut idx = Idx::new(data, data.ndim());
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

fn compact<T: CPUType>(
    idx: &mut Vec<usize>,
    shape: &[usize],
    strides: &[usize],
    data: &mut Vec<T>,
    src: &[T],
    dim: usize,
    pos: &mut usize,
    offset: usize,
) {
    if dim == shape.len() - 1 {
        if strides[dim] == 0 {
            data[*pos..*pos + shape[dim]]
                .iter_mut()
                .for_each(|x| *x = src[offset]);
            *pos += shape[dim];
        } else {
            T::copy(
                &src[offset..],
                &mut data[*pos..*pos + shape[dim]],
                strides[dim],
            );
            *pos += shape[dim];
        }
    } else {
        if strides[dim] == 0 {
            idx.push(0);
            let len = *pos;
            compact(idx, shape, strides, data, src, dim + 1, pos, offset);
            idx.pop();
            let copied = data[len..*pos].to_vec();
            let t = *pos - len;
            for _ in 1..shape[dim] {
                data[*pos..*pos + t].copy_from_slice(&copied);
                *pos += t;
            }
        } else {
            for i in 0..shape[dim] {
                idx.push(i);
                compact(
                    idx,
                    shape,
                    strides,
                    data,
                    src,
                    dim + 1,
                    pos,
                    offset + i * strides[dim],
                );
                idx.pop();
            }
        }
    }
}
