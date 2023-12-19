use crate::device::Device;
use crate::ndarray::{self, Idx, NDArray, Storage};
use crate::tensor::Tensor;
use crate::type_trait::{Float, Type, Unsigned};
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode};
use num_traits::Pow;
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

    fn ewise_op<T: Type>(
        &self,
        lhs: &NDArray<T, Self>,
        rhs: &NDArray<T, Self>,
        op: impl Fn(T, T) -> T,
    ) -> NDArray<T, Self> {
        if !lhs.is_contiguous() {
            return self.ewise_op(&lhs.contiguous(), rhs, op);
        }
        if !rhs.is_contiguous() {
            return self.ewise_op(lhs, &rhs.contiguous(), op);
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

    fn reduce_op<T: Type>(
        &self,
        lhs: &NDArray<T, Self>,
        shape: Vec<usize>,
        reduce_dims: usize,
        init: T,
        op: impl Fn(T, T) -> T,
    ) -> NDArray<T, Self> {
        let strides = ndarray::compact_strides(&shape);
        let len = shape[0] * strides[0];
        let mut idx = Idx::new(&lhs.0.shape);
        let reduce_lens = idx.shape[reduce_dims..lhs.ndim()].iter().product::<usize>();
        let data = (0..len)
            .into_iter()
            .map(|_| {
                let mut acc = init;
                for _ in 0..reduce_lens {
                    acc = op(acc, lhs[&idx]);
                    idx.next();
                }
                acc
            })
            .collect::<Vec<_>>();
        NDArray::make(Storage::CPU(data), shape, strides, 0, Self)
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
        let len = indices.len();
        let mut data = vec![T::zero(); len * num_classes];
        let mut idx = Idx::new(&indices.0.shape);
        for i in 0..len {
            data[i * num_classes + indices[&idx].to_usize().unwrap()] = T::one();
            idx.next();
        }
        let mut shape = idx.shape;
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
        self.ewise_op(lhs, rhs, |x, y| x + y)
    }

    fn sub<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        self.ewise_op(lhs, rhs, |x, y| x - y)
    }

    fn add_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| x + y)
    }

    fn scalar_sub<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| y - x)
    }

    fn mul<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        self.ewise_op(lhs, rhs, |x, y| x * y)
    }

    fn mul_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| x * y)
    }

    fn eq<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> bool {
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

    fn pow<T: Type + Pow<T, Output = T>>(
        &self,
        lhs: &NDArray<T, Self>,
        rhs: &NDArray<T, Self>,
    ) -> NDArray<T, Self> {
        self.ewise_op(lhs, rhs, |x, y| x.pow(y))
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
        self.ewise_op(lhs, rhs, |x, y| x / y)
    }

    fn div_scalar<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| x / y)
    }

    fn scalar_div<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: T) -> NDArray<T, Self> {
        self.scalar_op(lhs, rhs, |x, y| y / x)
    }

    fn ln<T: Float>(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        self.scalar_op(lhs, T::zero(), |x, _| x.ln())
    }

    fn sqrt<T: Float>(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        self.scalar_op(lhs, T::zero(), |x, _| x.sqrt())
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
                let m_offset = m * dims[2];
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
                    let inner_offset = outer_offset + m_offset + n;
                    for v in (0..dims[1]).step_by(tile) {
                        let t1 = v * tile;
                        let t2 = (v + tile) * tile;
                        let offset = n * tiled_dims[1];
                        let tile_lhs = &temp_lhs[t1..t2];
                        let tile_rhs = &temp_rhs[t1 + offset..t2 + offset];
                        let temp_ans = T::tiled_matmul(tile_lhs, tile_rhs);
                        for i in 0..r {
                            let i_offset = i * tile;
                            let x_inner = inner_offset + i * dims[2];
                            for j in 0..c {
                                data[x_inner + j] += temp_ans[i_offset + j];
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
        self.reduce_op(lhs, shape, reduce_dims, T::zero(), |x, y| x + y)
    }

    fn max<T: Type>(
        &self,
        lhs: &NDArray<T, Self>,
        shape: Vec<usize>,
        reduce_dims: usize,
    ) -> NDArray<T, Self> {
        self.reduce_op(lhs, shape, reduce_dims, T::min_value(), |x, y| {
            if x > y {
                x
            } else {
                y
            }
        })
    }

    fn equal<T: Type>(&self, lhs: &NDArray<T, Self>, rhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        self.ewise_op(lhs, rhs, |x, y| if x == y { T::one() } else { T::zero() })
    }

    fn contiguous<T: Type>(&self, lhs: &NDArray<T, Self>) -> NDArray<T, Self> {
        if let Storage::CPU(lhs_data) = lhs.0.data.as_ref() {
            let mut data = Vec::with_capacity(lhs.len());
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
            );
            NDArray::make(Storage::CPU(data), shape, strides, 0, Self)
        } else {
            panic!("Tensor Storage mismatched with Device.")
        }
    }

    fn data<T: Type>(lhs: &NDArray<T, Self>) -> Vec<T> {
        if let Storage::CPU(data) = lhs.0.data.as_ref() {
            data.clone()
        } else {
            panic!("Tensor Storage mismatched with Device")
        }
    }

    fn encode<T: Type, E: Encoder>(
        encoder: &mut E,
        lhs: &NDArray<T, Self>,
    ) -> Result<(), EncodeError> {
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
        if let Storage::CPU(data) = self.0.data.as_ref() {
            &data[index
                .idx
                .iter()
                .zip(&self.0.strides)
                .fold(self.0.offset, |acc, (idx, dim)| acc + idx * dim)]
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

fn compact<T: Type>(
    idx: &mut Vec<usize>,
    shape: &[usize],
    strides: &[usize],
    data: &mut Vec<T>,
    src: &[T],
    dim: usize,
) {
    if dim == shape.len() - 1 {
        for i in 0..shape[dim] {
            idx.push(i);
            let offset = idx
                .iter()
                .zip(strides)
                .fold(0, |acc, (&idx, &stride)| acc + idx * stride);
            data.push(src[offset]);
            idx.pop();
        }
    } else {
        if strides[dim] == 0 {
            idx.push(0);
            compact(idx, shape, strides, data, src, dim + 1);
            idx.pop();
            let copied = data.clone();
            for _ in 1..shape[dim] {
                data.extend(&copied)
            }
        } else {
            for i in 0..shape[dim] {
                idx.push(i);
                compact(idx, shape, strides, data, src, dim + 1);
                idx.pop();
            }
        }
    }
}
