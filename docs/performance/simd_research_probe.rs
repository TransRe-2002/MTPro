fn detect_step_core(diff_abs: &[f64], min_offset: f64, max_steps: usize) -> Vec<i64> {
    if diff_abs.is_empty() || max_steps == 0 {
        return Vec::new();
    }
    let mut working = diff_abs.to_vec();
    let limit = max_steps.min(working.len());
    let mut indices: Vec<i64> = Vec::with_capacity(limit);

    for _ in 0..limit {
        let (best_i, best_v) =
            working
                .iter()
                .enumerate()
                .fold(
                    (0usize, f64::MIN),
                    |(bi, bv), (i, &v)| {
                        if v > bv { (i, v) } else { (bi, bv) }
                    },
                );
        if best_v <= min_offset {
            break;
        }
        indices.push(best_i as i64);
        working[best_i] = f64::MIN / 2.0;
    }
    indices.sort_unstable();
    indices
}

use std::arch::x86_64::*;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::hint::black_box;
use std::time::Instant;
#[target_feature(enable = "avx")]
fn vector_argmax(y: &[f64]) -> usize {
    let mut best = _mm256_set1_pd(f64::MIN);
    let mut chunks = y.chunks_exact(4);
    for c in &mut chunks {
        let v = _mm256_setr_pd(c[0], c[1], c[2], c[3]);
        let mask = _mm256_cmp_pd::<_CMP_GT_OQ>(v, best);
        best = _mm256_blendv_pd(best, v, mask);
    }
    let lo = _mm256_castpd256_pd128(best);
    let hi = _mm256_extractf128_pd::<1>(best);
    let lanes = [
        _mm_cvtsd_f64(lo),
        _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo)),
        _mm_cvtsd_f64(hi),
        _mm_cvtsd_f64(_mm_unpackhi_pd(hi, hi)),
    ];
    let mut maximum = f64::MIN;
    for v in lanes.into_iter().chain(chunks.remainder().iter().copied()) {
        if v > maximum {
            maximum = v;
        }
    }
    y.iter().position(|v| *v == maximum).unwrap_or(0)
}
fn simd_steps(y: &[f64], threshold: f64, k: usize) -> Vec<i64> {
    let mut work = y.to_vec();
    let mut out = Vec::new();
    for _ in 0..k.min(y.len()) {
        // Only entered after the runtime CPU check in main.
        let i = unsafe { vector_argmax(&work) };
        if work[i] <= threshold || work[i].is_nan() {
            break;
        }
        out.push(i as i64);
        work[i] = f64::MIN / 2.;
    }
    out.sort_unstable();
    out
}
#[derive(Clone, Copy)]
struct Candidate {
    v: f64,
    i: usize,
}
impl PartialEq for Candidate {
    fn eq(&self, o: &Self) -> bool {
        self.v == o.v && self.i == o.i
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Candidate {
    fn cmp(&self, o: &Self) -> Ordering {
        self.v.total_cmp(&o.v).then_with(|| o.i.cmp(&self.i))
    }
}
fn topk(y: &[f64], threshold: f64, k: usize) -> Vec<i64> {
    let mut heap: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
    if k == 0 {
        return vec![];
    }
    for (i, &v) in y.iter().enumerate() {
        if !(v > threshold) {
            continue;
        }
        let c = Candidate { v, i };
        if heap.len() < k {
            heap.push(Reverse(c));
        } else if c > heap.peek().unwrap().0 {
            heap.pop();
            heap.push(Reverse(c));
        }
    }
    let mut out: Vec<i64> = heap.into_iter().map(|c| c.0.i as i64).collect();
    out.sort_unstable();
    out
}
fn time(name: &str, f: impl Fn() -> Vec<i64>) {
    black_box(f());
    let mut times = Vec::new();
    for _ in 0..7 {
        let t = Instant::now();
        black_box(f());
        times.push(t.elapsed().as_secs_f64() * 1000.);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    println!("{name}: {:.3} ms", times[3]);
}
fn step_main() {
    let n = 1_000_000;
    let mut y: Vec<f64> = (0..n)
        .map(|i| {
            let a = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (a >> 11) as f64 / ((1u64 << 53) as f64)
        })
        .collect();
    y[100] = 2.;
    y[900000] = 2.;
    for i in (0..n).step_by(997) {
        y[i] = f64::NAN;
    }
    let reference = detect_step_core(&y, 0., 20);
    assert_eq!(topk(&y, 0., 20), reference);
    time("original repeated argmax K20", || {
        detect_step_core(black_box(&y), 0., 20)
    });
    time("one-pass topK K20", || topk(black_box(&y), 0., 20));
    if is_x86_feature_detected!("avx") {
        assert_eq!(simd_steps(&y, 0., 20), reference);
        time("AVX repeated argmax K20", || {
            simd_steps(black_box(&y), 0., 20)
        });
    }
}

#[target_feature(enable = "avx")]
fn lanes(v: __m256d) -> [f64; 4] {
    let lo = _mm256_castpd256_pd128(v);
    let hi = _mm256_extractf128_pd::<1>(v);
    [
        _mm_cvtsd_f64(lo),
        _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo)),
        _mm_cvtsd_f64(hi),
        _mm_cvtsd_f64(_mm_unpackhi_pd(hi, hi)),
    ]
}
fn sift_scalar(c: &[f64], u: &[f64], l: &[f64]) -> (Vec<f64>, f64, f64) {
    let mut out = vec![0.; c.len()];
    let mut dn = 0.;
    let mut cn = 0.;
    for i in 0..c.len() {
        let mean = 0.5 * (u[i] + l[i]);
        out[i] = c[i] - mean;
        let d = c[i] - out[i];
        dn += d * d;
        cn += c[i] * c[i];
    }
    (out, dn, cn)
}
#[target_feature(enable = "avx")]
fn sift_simd(c: &[f64], u: &[f64], l: &[f64]) -> (Vec<f64>, f64, f64) {
    let mut out = vec![0.; c.len()];
    let mut dn = _mm256_setzero_pd();
    let mut cn = _mm256_setzero_pd();
    let half = _mm256_set1_pd(0.5);
    let n = c.len() / 4 * 4;
    for i in (0..n).step_by(4) {
        let a = _mm256_setr_pd(c[i], c[i + 1], c[i + 2], c[i + 3]);
        let b = _mm256_setr_pd(u[i], u[i + 1], u[i + 2], u[i + 3]);
        let z = _mm256_setr_pd(l[i], l[i + 1], l[i + 2], l[i + 3]);
        let mean = _mm256_mul_pd(half, _mm256_add_pd(b, z));
        let next = _mm256_sub_pd(a, mean);
        out[i..i + 4].copy_from_slice(&lanes(next));
        let d = _mm256_sub_pd(a, next);
        dn = _mm256_add_pd(dn, _mm256_mul_pd(d, d));
        cn = _mm256_add_pd(cn, _mm256_mul_pd(a, a));
    }
    let mut dsum: f64 = lanes(dn).into_iter().sum();
    let mut csum: f64 = lanes(cn).into_iter().sum();
    for i in n..c.len() {
        let mean = 0.5 * (u[i] + l[i]);
        out[i] = c[i] - mean;
        let d = c[i] - out[i];
        dsum += d * d;
        csum += c[i] * c[i];
    }
    (out, dsum, csum)
}
fn sift_time(name: &str, f: impl Fn() -> (Vec<f64>, f64, f64)) {
    black_box(f());
    let mut ts = vec![];
    for _ in 0..11 {
        let t = Instant::now();
        black_box(f());
        ts.push(t.elapsed().as_secs_f64() * 1000.);
    }
    ts.sort_by(|a, b| a.total_cmp(b));
    println!("{name}: {:.3} ms", ts[5]);
}
fn main() {
    step_main();
    let n = 1_000_000;
    let c: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
    let u: Vec<f64> = c.iter().map(|a| a * 0.3 + 1.).collect();
    let l: Vec<f64> = c.iter().map(|a| a * 0.2 - 1.).collect();
    sift_time("EMD sift scalar 1M", || sift_scalar(black_box(&c), &u, &l));
    if is_x86_feature_detected!("avx") {
        let a = sift_scalar(&c, &u, &l);
        let b = unsafe { sift_simd(&c, &u, &l) };
        assert_eq!(a.0, b.0);
        let err = ((a.1 - b.1) / a.1).abs().max(((a.2 - b.2) / a.2).abs());
        assert!(err < 1e-10);
        println!("EMD reduction max relative error: {err:.3e}");
        sift_time("EMD sift AVX 1M", || unsafe {
            sift_simd(black_box(&c), &u, &l)
        });
    }
}
