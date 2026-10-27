//! All architecture-specific unsafe operations are confined here. Runtime AVX2
//! detection guards each entry. Loads/stores are unaligned and always 16 i16s
//! within validated slices whose lengths are multiples of 16. Model bounds keep FT sums
//! within +/-20480; output pairwise i32 sums and horizontal sums cannot overflow.
use super::{Backend, QA};

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
