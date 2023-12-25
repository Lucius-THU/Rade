use crate::cpu::CPUIdx;
use crate::type_trait::Type;
use num_traits::Zero;
use std::cmp::min;
use std::simd::Simd;

extern crate blas_src;

extern "C" {
    pub fn cblas_sgemm(
        layout: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );

    pub fn cblas_scopy(n: i32, x: *const f32, inc_x: i32, y: *mut f32, inc_y: i32);
}

pub trait CPUType: Type + CPUCopy + CPUMatmul {}

pub trait CPUTile: Sized {
    const LANES: usize;

    fn tiled_matmul(
        lhs: &[Self],
        rhs: &[Self],
        out: &mut [Self],
        m: usize,
        k: usize,
        r: usize,
        c: usize,
    );
}

pub trait CPUMatmul: Sized {
    fn matmul(
        lhs: &[Self],
        rhs: &[Self],
        out: &mut [Self],
        lhs_idx: CPUIdx,
        rhs_idx: CPUIdx,
        dims: [usize; 3],
        strides: [usize; 2],
    );
}

pub trait CPUCopy: Sized {
    fn copy(lhs: &[Self], rhs: &mut [Self], stride: usize);
}

impl CPUCopy for f32 {
    fn copy(lhs: &[f32], rhs: &mut [f32], stride: usize) {
        unsafe {
            cblas_scopy(
                rhs.len() as i32,
                lhs.as_ptr(),
                stride as i32,
                rhs.as_mut_ptr(),
                1,
            );
        }
    }
}

impl CPUMatmul for f32 {
    fn matmul(
        lhs: &[f32],
        rhs: &[f32],
        out: &mut [f32],
        mut lhs_idx: CPUIdx,
        mut rhs_idx: CPUIdx,
        dims: [usize; 3],
        strides: [usize; 2],
    ) {
        let total = out.len();
        let inner_size = dims[0] * dims[2];
        let mut temp_lhs = vec![0.; dims[0] * dims[1]];
        let mut temp_rhs = vec![0.; dims[1] * dims[2]];
        for o in 0..total / inner_size {
            let outer_offset = o * inner_size;
            for i in 0..dims[0] {
                let offset = lhs_idx.get();
                f32::copy(
                    &lhs[offset..],
                    &mut temp_lhs[i * dims[1]..(i + 1) * dims[1]],
                    strides[0],
                );
                lhs_idx.next();
            }
            for i in 0..dims[1] {
                let offset = rhs_idx.get();
                f32::copy(
                    &rhs[offset..],
                    &mut temp_rhs[i * dims[2]..(i + 1) * dims[2]],
                    strides[1],
                );
                rhs_idx.next();
            }
            unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
                    dims[0] as i32,
                    dims[2] as i32,
                    dims[1] as i32,
                    1.0,
                    temp_lhs.as_ptr(),
                    dims[1] as i32,
                    temp_rhs.as_ptr(),
                    dims[2] as i32,
                    0.0,
                    out[outer_offset..].as_mut_ptr(),
                    dims[2] as i32,
                );
            }
        }
    }
}

macro_rules! impl_copy {
    ($($t:ty),*) => {
        $(
            impl CPUCopy for $t {
                fn copy(lhs: &[$t], rhs: &mut [$t], stride: usize) {
                    for i in 0..rhs.len() {
                        rhs[i] = lhs[i * stride];
                    }
                }
            }
        )*
    }
}

macro_rules! impl_tile {
    ($($t:ty),*) => {
        $(
            impl CPUTile for $t {
                const LANES: usize = 8;

                fn tiled_matmul(lhs: &[$t], rhs: &[$t], out: &mut [$t], m: usize, k: usize, r: usize, c: usize) {
                    let tile = Self::LANES;
                    let mut temp = vec![Simd::<$t, { Self::LANES }>::splat(<$t as Zero>::zero()); tile];
                    for k in (0..m).step_by(tile) {
                        let t1 = k * tile;
                        let t2 = (k + tile) * tile;
                        let tile_lhs = &lhs[t1..t2];
                        let tile_rhs = &rhs[t1..t2];
                        for j in 0..tile {
                            let j_offset = j * tile;
                            let rhs_simd = Simd::<$t, { Self::LANES }>::from_slice(&tile_rhs[j_offset..]);
                            for i in 0..tile {
                                let i_offset = i * tile;
                                let lhs_simd = Simd::<$t, { Self::LANES }>::splat(tile_lhs[i_offset + j]);
                                temp[i] += lhs_simd * rhs_simd;
                            }
                        }
                    }
                    for i in 0..r {
                        out[i * k..i * k + c].copy_from_slice(&temp[i][..c]);
                    }
                }
            }
        )*
    };
}

macro_rules! impl_matmul {
    ($($t:ty),*) => {
        $(
            impl CPUMatmul for $t {
                fn matmul(lhs: &[$t], rhs: &[$t], out: &mut [$t], mut lhs_idx: CPUIdx, mut rhs_idx: CPUIdx, dims: [usize; 3], strides: [usize; 2]) {
                    let total = out.len();
                    let inner_size = dims[0] * dims[2];
                    let tile = <$t>::LANES;
                    let tiled_dims = dims
                        .iter()
                        .map(|x| (x + tile - 1) / tile * tile)
                        .collect::<Vec<_>>();
                    for o in 0..total / inner_size {
                        let outer_offset = o * inner_size;
                        let mut temp_rhs = vec![<$t>::zero(); tiled_dims[1] * tiled_dims[2]];
                        let mut uncontiguous = vec![<$t>::zero(); dims[2]];
                        for i in 0..dims[1] {
                            let offset = rhs_idx.get();
                            <$t as CPUCopy>::copy(
                                &rhs[offset..],
                                &mut uncontiguous,
                                strides[1],
                            );
                            for j in 0..dims[2] {
                                let col_tiled = j / tile * tile;
                                let col_rem = j - col_tiled;
                                temp_rhs[col_tiled * tiled_dims[1] + i * tile + col_rem] = uncontiguous[j];
                            }
                            rhs_idx.next();
                        }
                        for m in (0..dims[0]).step_by(tile) {
                            let mut temp_lhs = vec![<$t>::zero(); tile * tiled_dims[1]];
                            let mut uncontiguous = vec![<$t>::zero(); dims[1]];
                            let r = min(tile, dims[0] - m);
                            let m_offset = m * dims[2];
                            for i in 0..r {
                                let offset = lhs_idx.get();
                                <$t as CPUCopy>::copy(
                                    &lhs[offset..],
                                    &mut uncontiguous,
                                    strides[0],
                                );
                                for j in 0..dims[1] {
                                    let col_tiled = j / tile * tile;
                                    let col_rem = j - col_tiled;
                                    temp_lhs[(col_tiled + i) * tile + col_rem] = uncontiguous[j];
                                }
                                lhs_idx.next();
                            }
                            for n in (0..dims[2]).step_by(tile) {
                                <$t>::tiled_matmul(
                                    &temp_lhs,
                                    &temp_rhs[n * tiled_dims[1]..],
                                    &mut out[outer_offset + m_offset + n..],
                                    dims[1],
                                    dims[2],
                                    r,
                                    min(tile, dims[2] - n),
                                );
                            }
                        }
                    }
                }
            }
        )*
    };
}

impl_copy!(f64, isize, i8, i16, i32, i64, usize, u8, u16, u32, u64);

impl_tile!(f64, isize, i8, i16, i32, i64, usize, u8, u16, u32, u64);

impl_matmul!(f64, isize, i8, i16, i32, i64, usize, u8, u16, u32, u64);

impl<T: CPUCopy + CPUMatmul + Type> CPUType for T {}
