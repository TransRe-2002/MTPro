//! Diurnal / periodic signal subtraction (Rust acceleration for core/algorithms/diurnal.py).
//!
//! Provides `subtract_periodic_signal_series_rs` used by the Python layer.
//!
//! Algorithm:
//!   1. Pre-build design matrix (trend + harmonic cos/sin columns) — same for every window.
//!   2. For each window: solve least-squares `design[valid] @ coeffs = segment[valid]`,
//!      extract harmonic coefficients, compute `harmonic_signal = harmonic_matrix @ coeffs`,
//!      scatter-add weighted into `noise_sum` and `weight_sum`.
//!   3. Final: `estimated_noise = noise_sum / weight_sum`.

use numpy::{
    array::PyArray1,
    prelude::*,
    PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::prelude::*;
use pyo3::Bound;

/// Solve `A x = b` via QR decomposition (Householder) for small dense systems.
/// A is (m, n), b is (m,). Returns x of length n.
/// This is a simple implementation sufficient for the small system sizes
/// encountered here (n ≤ ~10, m up to window_size ~51840 for 3-day 5s data).
fn lstsq_qr(a: &[f64], b: &[f64], m: usize, n: usize) -> Vec<f64> {
    // Augmented matrix [A | b], work in column-major for QR.
    // Actually we'll do row-major Householder QR.

    // Make a working copy of A (m×n) and b (m)
    let mut r = a.to_vec(); // m*n row-major, will become R
    let mut bb = b.to_vec(); // m, will become Q^T b

    // Householder QR: for each column k in 0..min(m,n), reflect to zero below diagonal.
    let steps = n.min(m);
    // Apply Householder transformations to bb as we go, accumulating into qtb.
    // We'll store Householder vectors in the lower triangle of r.
    for k in 0..steps {
        // Compute norm of column k below (and including) row k.
        let mut sigma: f64 = 0.0;
        for i in k..m {
            sigma += r[i * n + k] * r[i * n + k];
        }
        sigma = sigma.sqrt();
        if sigma == 0.0 {
            continue;
        }
        // Choose sign to avoid cancellation.
        let diag = r[k * n + k];
        let sign = if diag >= 0.0 { 1.0 } else { -1.0 };
        let mut v = vec![0.0_f64; m - k];
        v[0] = diag + sign * sigma;
        for i in 1..(m - k) {
            v[i] = r[(k + i) * n + k];
        }
        let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
        if v_norm_sq == 0.0 {
            continue;
        }
        // Apply reflection to remaining columns of r (columns k..n)
        for j in k..n {
            let mut dot = 0.0;
            for i in 0..(m - k) {
                dot += v[i] * r[(k + i) * n + j];
            }
            let factor = 2.0 * dot / v_norm_sq;
            for i in 0..(m - k) {
                r[(k + i) * n + j] -= factor * v[i];
            }
        }
        // Apply reflection to bb
        let mut dot = 0.0;
        for i in 0..(m - k) {
            dot += v[i] * bb[k + i];
        }
        let factor = 2.0 * dot / v_norm_sq;
        for i in 0..(m - k) {
            bb[k + i] -= factor * v[i];
        }
    }

    // Back-substitute: R (upper triangular) is in r[0..n, 0..n] row-major (r[i*n+j], i<=j).
    // The transformed rhs Q^T b is in bb[0..n].
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = bb[i];
        for j in (i + 1)..n {
            s -= r[i * n + j] * x[j];
        }
        let diag = r[i * n + i];
        if diag.abs() > 1e-30 {
            x[i] = s / diag;
        } else {
            x[i] = 0.0;
        }
    }
    x
}

/// Python-callable: subtract_periodic_signal_series_rs(values, valid_mask_u8, starts, weights,
///     design_matrix, harmonic_matrix, harmonic_offset, window_size, min_valid_samples,
///     min_valid_fraction, base_period_seconds, harmonics, sample_interval_seconds, trend_degree)
///
/// Returns (noise_sum: PyArray1<f64>, weight_sum: PyArray1<f64>).
/// Python wrapper handles normalization, NaN handling and DataFrame construction.
#[pyfunction]
#[pyo3(name = "subtract_periodic_signal_series_rs")]
#[allow(clippy::too_many_arguments)]
pub fn subtract_periodic_signal_series_rs<'py>(
    py: Python<'py>,
    values: numpy::PyReadonlyArray1<'py, f64>,
    valid_mask: numpy::PyReadonlyArray1<'py, u8>,
    starts: numpy::PyReadonlyArray1<'py, i64>,
    weights: numpy::PyReadonlyArray1<'py, f64>,
    design_matrix: numpy::PyReadonlyArray2<'py, f64>,
    harmonic_matrix: numpy::PyReadonlyArray2<'py, f64>,
    harmonic_offset: usize,
    window_size: usize,
    min_valid_samples: usize,
    min_valid_fraction: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let values = values.as_slice()?;
    let valid_mask = valid_mask.as_slice()?;
    let starts = starts.as_slice()?;
    let weights = weights.as_slice()?;
    let design = design_matrix.as_slice()?;
    let harmonic = harmonic_matrix.as_slice()?;
    let design_dims = design_matrix.dims();
    let n_rows = design_dims[0];
    let n_params = design_dims[1];
    let harm_dims = harmonic_matrix.dims();
    let _n_rows_h = harm_dims[0];
    let n_harm = harm_dims[1];
    if n_rows != window_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("design_matrix rows {} != window_size {}", n_rows, window_size),
        ));
    }
    let _ = n_harm;

    let n = values.len();
    let mut noise_sum = vec![0.0_f64; n];
    let mut weight_sum = vec![0.0_f64; n];

    // Pre-allocate reusable buffers
    let mut a_valid = Vec::with_capacity(n_rows * n_params);
    let mut b_valid = Vec::with_capacity(n_rows);
    let mut harm_signal = vec![0.0_f64; n_rows];

    for &start_i in starts {
        let start = start_i as usize;
        let stop = (start + window_size).min(n);
        let cur_window = stop - start;
        if cur_window == 0 {
            continue;
        }

        // Count valid samples in window
        let mut valid_count = 0usize;
        for i in 0..cur_window {
            if valid_mask[start + i] != 0 {
                valid_count += 1;
            }
        }

        let valid_fraction = valid_count as f64 / window_size as f64;

        // Skip if insufficient valid samples (Python wrapper will mark diagnostics)
        if valid_count < min_valid_samples || valid_fraction < min_valid_fraction {
            continue;
        }

        // Build valid subset of design matrix and rhs
        a_valid.clear();
        b_valid.clear();
        a_valid.reserve(valid_count * n_params);
        b_valid.reserve(valid_count);
        for i in 0..cur_window {
            if valid_mask[start + i] != 0 {
                for j in 0..n_params {
                    a_valid.push(design[i * n_params + j]);
                }
                b_valid.push(values[start + i]);
            }
        }

        // Solve least squares: a_valid @ coeffs = b_valid
        let coeffs = lstsq_qr(&a_valid, &b_valid, valid_count, n_params);

        // Compute harmonic signal = harmonic_matrix @ harmonic_coeffs
        let harm_coeffs = &coeffs[harmonic_offset..];
        for i in 0..cur_window {
            let mut s = 0.0;
            for j in 0..n_harm {
                s += harmonic[i * n_harm + j] * harm_coeffs[j];
            }
            harm_signal[i] = s;
        }

        // Scatter-add weighted
        for i in 0..cur_window {
            noise_sum[start + i] += harm_signal[i] * weights[i];
            weight_sum[start + i] += weights[i];
        }
    }

    Ok((
        PyArray1::from_vec(py, noise_sum),
        PyArray1::from_vec(py, weight_sum),
    ))
}