//! All architecture-specific unsafe operations are confined here. Runtime AVX2
//! detection guards each entry. Loads/stores are unaligned and always 16 i16s
//! within validated slices whose lengths are multiples of 16. Model bounds keep FT sums
//! within +/-20480; output pairwise i32 sums and horizontal sums cannot overflow.
use super::{Backend, QA};

/// All removals precede additions, preserving the accumulator's signed bounds.
pub(super) fn update_delta<const R: usize, const A: usize>(
    backend: Backend,
    sums: &mut [i16; 512],
    removed: [&[i16]; R],
    added: [&[i16]; A],
) {
    assert!(R <= 2 && A <= 2);
    assert!(removed.iter().chain(&added).all(|row| row.len() == 512));
    #[cfg(target_arch = "x86_64")]
    if backend == Backend::Avx2 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: CPU and all 512-lane row bounds checked above.
        unsafe { avx2_delta(sums, removed, added) };
        return;
    }
    let _ = backend;
    for (i, sum) in sums.iter_mut().enumerate() {
        for row in removed {
            *sum -= row[i];
        }
        for row in added {
            *sum += row[i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_delta<const R: usize, const A: usize>(
    sums: &mut [i16; 512],
    removed: [&[i16]; R],
    added: [&[i16]; A],
) {
    use std::arch::x86_64::*;
    for i in (0..512).step_by(16) {
        // SAFETY: i+16<=512 for the accumulator and every validated row.
        unsafe {
            let mut sum = _mm256_loadu_si256(sums.as_ptr().add(i).cast());
            for row in removed {
                sum = _mm256_sub_epi16(sum, _mm256_loadu_si256(row.as_ptr().add(i).cast()));
            }
            for row in added {
                sum = _mm256_add_epi16(sum, _mm256_loadu_si256(row.as_ptr().add(i).cast()));
            }
            _mm256_storeu_si256(sums.as_mut_ptr().add(i).cast(), sum);
        }
    }
}

/// 16 output neurons per tile, then pairs of inputs, then neuron / input lane.
/// This feeds two input values to all 16 output lanes without horizontal sums.
pub(super) fn pack_affine<const I: usize, const O: usize>(weights: &[i16]) -> Vec<i16> {
    assert!(I > 0 && I <= 512 && I.is_multiple_of(16) && O > 0 && O.is_multiple_of(16));
    assert_eq!(weights.len(), I * O);
    let mut packed = Vec::with_capacity(I * O);
    for tile in (0..O).step_by(16) {
        for input in (0..I).step_by(2) {
            for output in tile..tile + 16 {
                packed.extend_from_slice(&weights[output * I + input..output * I + input + 2]);
            }
        }
    }
    packed
}

/// Inputs are clipped activations in 0..255; weights/bias are validated v3
/// parameters. Every i32 partial sum is bounded by 512*255*127+65536.
pub(super) fn affine<const I: usize, const O: usize>(
    backend: Backend,
    input: &[i16; I],
    packed: &[i16],
    bias: &[i32; O],
    output: &mut [i16; O],
) {
    assert!(I > 0 && I <= 512 && I.is_multiple_of(16) && O > 0 && O.is_multiple_of(16));
    assert_eq!(packed.len(), I * O);
    debug_assert!(input.iter().all(|&v| (0..=255).contains(&v)));
    #[cfg(target_arch = "x86_64")]
    if backend == Backend::Avx2 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: CPU checked; dimensions and packed buffer sizes validated.
        unsafe { avx2_affine(input, packed, bias, output) };
        return;
    }
    let _ = backend;
    for (o, value) in output.iter_mut().enumerate() {
        let mut sum = bias[o];
        for i in (0..I).step_by(2) {
            let offset = ((o / 16) * (I / 2) + i / 2) * 32 + (o % 16) * 2;
            sum += i32::from(input[i]) * i32::from(packed[offset])
                + i32::from(input[i + 1]) * i32::from(packed[offset + 1]);
        }
        *value = (sum / 64).clamp(0, 255) as i16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_affine<const I: usize, const O: usize>(
    input: &[i16; I],
    packed: &[i16],
    bias: &[i32; O],
    output: &mut [i16; O],
) {
    use std::arch::x86_64::*;
    for tile in (0..O).step_by(16) {
        // SAFETY: O is a multiple of 16; bias loads cover tile..tile+16.
        let (mut lo, mut hi) = unsafe {
            (
                _mm256_loadu_si256(bias.as_ptr().add(tile).cast()),
                _mm256_loadu_si256(bias.as_ptr().add(tile + 8).cast()),
            )
        };
        for block in (0..I).step_by(16) {
            // SAFETY: I is a multiple of 16; this loads 8 complete input pairs.
            let values = unsafe { _mm256_loadu_si256(input.as_ptr().add(block).cast()) };
            let zeros = _mm256_cmpeq_epi32(values, _mm256_setzero_si256());
            let mut mask = (!_mm256_movemask_ps(_mm256_castsi256_ps(zeros)) as u32) & 255;
            while mask != 0 {
                let i = block + 2 * mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let pair = i32::from(input[i]) | (i32::from(input[i + 1]) << 16);
                let x = _mm256_set1_epi32(pair);
                let offset = tile * I + i * 16;
                // SAFETY: offset+32 <= I*O, each input pair occupies 32 i16s.
                unsafe {
                    let w0 = _mm256_loadu_si256(packed.as_ptr().add(offset).cast());
                    let w1 = _mm256_loadu_si256(packed.as_ptr().add(offset + 16).cast());
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(x, w0));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(x, w1));
                }
            }
        }
        // For negative values floor/trunc both clamp to zero. Positive shift
        // is exact /64. Pack's lane interleave is restored before storing.
        lo = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_srai_epi32::<6>(lo), _mm256_setzero_si256()),
            _mm256_set1_epi32(255),
        );
        hi = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_srai_epi32::<6>(hi), _mm256_setzero_si256()),
            _mm256_set1_epi32(255),
        );
        let result = _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi32(lo, hi));
        // SAFETY: tile+16<=O; store is unaligned and spans 16 output i16s.
        unsafe { _mm256_storeu_si256(output.as_mut_ptr().add(tile).cast(), result) };
    }
}

/// Pair matching halves into 256 unsigned clipped products divided by 255.
pub(super) fn pair(backend: Backend, sums: &[i16; 512], output: &mut [i16; 256]) {
    #[cfg(target_arch = "x86_64")]
    if backend == Backend::Avx2 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: CPU support checked; all fixed-size loads/stores are in bounds.
        unsafe { avx2_pair(sums, output) };
        return;
    }
    let _ = backend;
    for (i, value) in output.iter_mut().enumerate() {
        *value =
            (i32::from(sums[i]).clamp(0, QA) * i32::from(sums[i + 256]).clamp(0, QA) / QA) as i16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_pair(sums: &[i16; 512], output: &mut [i16; 256]) {
    use std::arch::x86_64::*;
    for i in (0..256).step_by(16) {
        // SAFETY: i+16<=256 and i+256+16<=512; unaligned loads/stores are supported.
        unsafe {
            let a = _mm256_loadu_si256(sums.as_ptr().add(i).cast());
            let b = _mm256_loadu_si256(sums.as_ptr().add(i + 256).cast());
            let zero = _mm256_setzero_si256();
            let max = _mm256_set1_epi16(255);
            let a = _mm256_min_epi16(_mm256_max_epi16(a, zero), max);
            let b = _mm256_min_epi16(_mm256_max_epi16(b, zero), max);
            let product = _mm256_mullo_epi16(a, b);
            // Exact unsigned floor(x/255) for 0<=x<=65025; sum<=65280 fits u16.
            let adjusted = _mm256_add_epi16(
                _mm256_add_epi16(product, _mm256_set1_epi16(1)),
                _mm256_srli_epi16::<8>(product),
            );
            _mm256_storeu_si256(
                output.as_mut_ptr().add(i).cast(),
                _mm256_srli_epi16::<8>(adjusted),
            );
        }
    }
}

pub(super) fn update(backend: Backend, sums: &mut [i16], weights: &[i16], adding: bool) {
    assert_eq!(sums.len(), weights.len());
    assert!([256, 512].contains(&sums.len()));
    #[cfg(target_arch = "x86_64")]
    if backend == Backend::Avx2 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: CPU/OS support checked above; equal validated lengths are multiples of 16.
        unsafe { avx2_update(sums, weights, adding) };
        return;
    }
    let _ = backend;
    for (sum, &weight) in sums.iter_mut().zip(weights) {
        *sum = if adding { *sum + weight } else { *sum - weight };
    }
}
pub(super) fn output(backend: Backend, sums: &[i16], weights: &[i16]) -> i64 {
    assert_eq!(sums.len(), weights.len());
    assert!([16, 32, 256, 512].contains(&sums.len()));
    #[cfg(target_arch = "x86_64")]
    if backend == Backend::Avx2 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: CPU/OS support checked above; equal validated lengths are multiples of 16.
        return unsafe { avx2_output(sums, weights) };
    }
    let _ = backend;
    sums.iter()
        .zip(weights)
        .map(|(&sum, &weight)| i64::from(sum.clamp(0, QA as i16)) * i64::from(weight))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_update(sums: &mut [i16], weights: &[i16], adding: bool) {
    use std::arch::x86_64::*;
    for index in (0..sums.len()).step_by(16) {
        // SAFETY: index + 16 <= len; both 32-byte ranges lie inside the slices.
        unsafe {
            let a = _mm256_loadu_si256(sums.as_ptr().add(index).cast());
            let b = _mm256_loadu_si256(weights.as_ptr().add(index).cast());
            let value = if adding {
                _mm256_add_epi16(a, b)
            } else {
                _mm256_sub_epi16(a, b)
            };
            _mm256_storeu_si256(sums.as_mut_ptr().add(index).cast(), value);
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_output(sums: &[i16], weights: &[i16]) -> i64 {
    use std::arch::x86_64::*;
    let mut total = _mm256_setzero_si256();
    for index in (0..sums.len()).step_by(16) {
        // SAFETY: index + 16 <= len; both 32-byte loads lie inside the slices.
        unsafe {
            let input = _mm256_loadu_si256(sums.as_ptr().add(index).cast());
            let clipped = _mm256_min_epi16(
                _mm256_max_epi16(input, _mm256_setzero_si256()),
                _mm256_set1_epi16(QA as i16),
            );
            let weight = _mm256_loadu_si256(weights.as_ptr().add(index).cast());
            total = _mm256_add_epi32(total, _mm256_madd_epi16(clipped, weight));
        }
    }
    let mut lanes = [0i32; 8];
    // SAFETY: lanes provides exactly 32 writable bytes; no alignment required.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), total) };
    lanes.into_iter().map(i64::from).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_affine<const I: usize, const O: usize>() {
        let mut seed = 19u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for case in 0..96 {
            let mut input = [0i16; I];
            for value in &mut input {
                let r = next();
                *value = if case % 3 == 0 || r % 3 == 0 {
                    0
                } else {
                    (r % 256) as i16
                };
            }
            let mut weights = vec![0i16; I * O];
            for w in &mut weights {
                *w = (next() % 255) as i16 - 127;
            }
            let mut bias = [0i32; O];
            for b in &mut bias {
                *b = (next() % 131073) as i32 - 65536;
            }
            if case == 0 {
                input.fill(255);
                weights.fill(127);
                bias.fill(65536);
            }
            if case == 1 {
                input.fill(255);
                weights.fill(-127);
                bias.fill(-65536);
            }
            if case == 2 {
                input[I - 1] = 255;
            }
            if case == 3 {
                input.fill(0);
                let boundaries = [
                    -65536, -65, -64, -63, -1, 0, 1, 63, 64, 65, 16319, 16320, 16321, 65535, 65536,
                ];
                for (i, b) in bias.iter_mut().enumerate() {
                    *b = boundaries[i % boundaries.len()];
                }
            }
            let reference: Vec<i16> = (0..O)
                .map(|o| {
                    let raw = i64::from(bias[o])
                        + input
                            .iter()
                            .zip(&weights[o * I..(o + 1) * I])
                            .map(|(&a, &w)| i64::from(a) * i64::from(w))
                            .sum::<i64>();
                    (raw / 64).clamp(0, 255) as i16
                })
                .collect();
            let packed = pack_affine::<I, O>(&weights);
            for backend in [Backend::Scalar, Backend::Avx2] {
                if !backend.available() {
                    continue;
                }
                let mut output = [0i16; O];
                affine(backend, &input, &packed, &bias, &mut output);
                assert_eq!(
                    output.as_slice(),
                    reference,
                    "{I}x{O} case {case} {backend:?}"
                );
            }
        }
    }

    #[test]
    fn tiled_affine_matches_independent_output_major_dot_at_bounds_and_sparse_inputs() {
        check_affine::<512, 16>();
        check_affine::<16, 32>();
    }
    #[test]
    fn pair_products_are_exact_for_every_clipped_input_combination() {
        let mut input = [0i16; 512];
        let mut output = [0i16; 256];
        for a in 0..=255 {
            input[..256].fill(a);
            for (b, v) in input[256..].iter_mut().enumerate() {
                *v = b as i16;
            }
            for backend in [Backend::Scalar, Backend::Avx2] {
                if !backend.available() {
                    continue;
                }
                pair(backend, &input, &mut output);
                for (b, &v) in output.iter().enumerate() {
                    assert_eq!(i32::from(v), i32::from(a) * b as i32 / 255);
                }
            }
        }
        input[..256].fill(i16::MIN);
        input[256..].fill(i16::MAX);
        pair(Backend::Scalar, &input, &mut output);
        assert_eq!(output, [0; 256]);
        if Backend::Avx2.available() {
            pair(Backend::Avx2, &input, &mut output);
            assert_eq!(output, [0; 256]);
        }
    }
}
