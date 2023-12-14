use std::simd::Simd;
use num_traits::Zero;

pub trait Ops: Sized {
    const LANES: usize;

    fn tiled_matmul(lhs: &[Self], rhs: &[Self]) -> Vec<Self>;
}

macro_rules! impl_ops_simd {
    ($($t:ty),*) => {
        $(
            impl Ops for $t {
                const LANES: usize = 64;

                fn tiled_matmul(lhs: &[$t], rhs: &[$t]) -> Vec<$t> {
                    let tile = Self::LANES;
                    let mut out = vec![Simd::<$t, { Self::LANES }>::splat(<$t as Zero>::zero()); tile];
                    for j in 0..tile {
                        let j_offset = j * tile;
                        let rhs_simd = Simd::<$t, { Self::LANES }>::from_slice(&rhs[j_offset..]);
                        for i in 0..tile {
                            let i_offset = i * tile;
                            let lhs_simd = Simd::<$t, { Self::LANES }>::splat(lhs[i_offset + j]);
                            out[i] += lhs_simd * rhs_simd;
                        }
                    }
                    let mut ans = Vec::with_capacity(tile * tile);
                    for i in 0..tile {
                        ans.extend_from_slice(out[i].as_array());
                    }
                    ans
                }
            }
        )*
    };
}

macro_rules! impl_ops_plain {
    ($($t:ty),*) => {
        $(
            impl Ops for $t {
                const LANES: usize = 64;

                fn tiled_matmul(lhs: &[$t], rhs: &[$t]) -> Vec<$t> {
                    let tile = Self::LANES;
                    let mut ans = vec![<$t as Zero>::zero(); tile * tile];
                    for j in 0..tile {
                        let j_offset = j * tile;
                        for i in 0..tile {
                            let i_offset = i * tile;
                            for k in 0..tile {
                                ans[i_offset + k] += lhs[i_offset + j] * rhs[j_offset + k];
                            }
                        }
                    }
                    ans
                }
            }
        )*
    };
}

impl_ops_simd!(f32, f64, isize, i8, i16, i32, i64, usize, u8, u16, u32, u64);
impl_ops_plain!(i128, u128);