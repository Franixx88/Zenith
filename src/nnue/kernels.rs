//! All architecture-specific unsafe operations are confined here. Runtime AVX2
//! detection guards each entry. Loads/stores are unaligned and always 16 i16s
//! within fixed 256-lane arrays. Validated model bounds keep intermediate sums
//! within +/-20480; output pairwise i32 sums and horizontal sums cannot overflow.
use super::{Backend, HIDDEN, QA};

pub(super) fn update(backend: Backend, sums: &mut [i16; HIDDEN], weights: &[i16], adding: bool) {
    let weights: &[i16; HIDDEN] = weights.try_into().expect("256 weights");
    #[cfg(target_arch = "x86_64")]
    if backend == Backend::Avx2 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: CPU/OS support checked above, lengths guaranteed by array types.
        unsafe { avx2_update(sums, weights, adding) };
        return;
    }
    let _ = backend;
    for (sum, &weight) in sums.iter_mut().zip(weights) {
        *sum = if adding { *sum + weight } else { *sum - weight };
    }
}
pub(super) fn output(backend: Backend, sums: &[i16; HIDDEN], weights: &[i16]) -> i64 {
    let weights: &[i16; HIDDEN] = weights.try_into().expect("256 output weights");
    #[cfg(target_arch = "x86_64")]
    if backend == Backend::Avx2 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: CPU/OS support checked above, exactly 256 lanes in both arrays.
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
unsafe fn avx2_update(sums: &mut [i16; HIDDEN], weights: &[i16; HIDDEN], adding: bool) {
    use std::arch::x86_64::*;
    for index in (0..HIDDEN).step_by(16) {
        // SAFETY: index<=240; both 32-byte ranges lie inside their arrays.
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
unsafe fn avx2_output(sums: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i64 {
    use std::arch::x86_64::*;
    let mut total = _mm256_setzero_si256();
    for index in (0..HIDDEN).step_by(16) {
        // SAFETY: index<=240; both 32-byte loads lie inside their arrays.
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
