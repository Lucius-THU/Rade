use num_traits::Zero;
use std::simd::Simd;

pub trait Tile: Sized {
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

macro_rules! impl_tile {
    ($($t:ty),*) => {
        $(
            impl Tile for $t {
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

impl_tile!(f32, f64, isize, i8, i16, i32, i64, usize, u8, u16, u32, u64);
