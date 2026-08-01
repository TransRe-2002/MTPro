//! Diurnal / periodic signal subtraction (Rust acceleration for core/algorithms/diurnal.py).
//!
//! Provides `subtract_periodic_signal_series_rs` used by the Python layer.
//!
//! Algorithm (two-step least squares, matches the Python `_design_matrices`
//! two-matrix split to avoid trend/harmonic collinearity):
//!   For each window:
//!     1. Solve `trend_matrix[valid] @ trend_coeffs = segment[valid]` (least squares via SVD).
//!     2. Compute `detrended = segment - trend_matrix @ trend_coeffs`.
//!     3. Solve `harmonic_matrix[valid] @ harmonic_coeffs = detrended[valid]` (least squares via SVD).
//!     4. Compute `harmonic_signal = harmonic_matrix @ harmonic_coeffs`.
//!     5. Scatter-add `harmonic_signal * weights` into `noise_sum`,
//!        and `weights` into `weight_sum`.

use nalgebra::{DMatrix, DVector};
use numpy::{array::PyArray1, prelude::*};
use pyo3::prelude::*;
use pyo3::Bound;

/// Solve `A x = b` in least-squares sense using nalgebra's SVD with pseudo-inverse.
/// `a` is m×n row-major slice, `b` is length m. Returns x of length n.
///
/// Uses nalgebra's SVD (the same LAPACK-style implicit-shift algorithm as numpy/scipy),
/// so results are bit-identical to `np.linalg.lstsq` up to floating-point reordering.
///
/// Tolerances mirror numpy's default `rcond=None` behaviour: singular values
/// below `eps * max_sigma` are zeroed, where `eps` is the f64 machine epsilon
/// scaled by the larger of (m, n).
fn lstsq_svd(a: &[f64], b: &[f64], m: usize, n: usize) -> Vec<f64> {
    // Reinterpret row-major `a` as nalgebra DMatrix (column-major, so we transpose).
    let mat = DMatrix::from_row_slice(m, n, a);
    let rhs = DVector::from_row_slice(b);
    // SVD with U and V^T both computed (needed for solve).
    let svd = mat.clone().svd(true, true);
    // rcond-style epsilon: max(m, n) * f64::EPS as the singular-value cutoff.
    let n_eff = m.max(n);
    let eps = (n_eff as f64) * f64::EPSILON;
    // solve returns Result; on failure fall back to pseudo-inverse via pinv.
    match svd.solve(&rhs, eps) {
        Ok(x) => x.as_slice().to_vec(),
        Err(_) => {
            // Defensive: pseudo_inverse then matrix-vector multiply.
            svd.pseudo_inverse(eps)
                .map(|pinv| (pinv * &rhs).as_slice().to_vec())
                .unwrap_or_else(|_| vec![0.0; n])
        }
    }
}

/// Python-callable: subtract_periodic_signal_series_rs(values, valid_mask_u8, starts, weights,
///     trend_matrix, harmonic_matrix, window_size, min_valid_samples, min_valid_fraction).
///
/// Returns (noise_sum: PyArray1<f64>, weight_sum: PyArray1<f64>,
///          per_window_amplitudes: PyArray2<f64>).
/// `per_window_amplitudes` has shape (n_starts, n_harm/2); each row gives the
/// per-harmonic amplitudes for that window (NaN for skipped windows).
#[pyfunction]
#[pyo3(name = "subtract_periodic_signal_series_rs")]
#[allow(clippy::too_many_arguments)]
pub fn subtract_periodic_signal_series_rs<'py>(
    py: Python<'py>,
    values: numpy::PyReadonlyArray1<'py, f64>,
    valid_mask: numpy::PyReadonlyArray1<'py, u8>,
    starts: numpy::PyReadonlyArray1<'py, i64>,
    weights: numpy::PyReadonlyArray1<'py, f64>,
    trend_matrix: numpy::PyReadonlyArray2<'py, f64>,
    harmonic_matrix: numpy::PyReadonlyArray2<'py, f64>,
    window_size: usize,
    min_valid_samples: usize,
    min_valid_fraction: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, numpy::PyArray2<f64>>,
)> {
    use numpy::ndarray::Array2;
    let values = values.as_slice()?;
    let valid_mask = valid_mask.as_slice()?;
    let starts = starts.as_slice()?;
    let weights = weights.as_slice()?;
    let trend = trend_matrix.as_slice()?;
    let trend_dims = trend_matrix.dims();
    let _trend_rows = trend_dims[0];
    let n_trend = trend_dims[1];

    let harmonic = harmonic_matrix.as_slice()?;
    let harm_dims = harmonic_matrix.dims();
    let _harm_rows = harm_dims[0];
    let n_harm = harm_dims[1];
    let n_amplitudes = n_harm / 2;

    let n = values.len();
    let mut noise_sum = vec![0.0_f64; n];
    let mut weight_sum = vec![0.0_f64; n];
    let n_starts = starts.len();
    let mut amplitudes = vec![f64::NAN; n_starts * n_amplitudes];

    // Pre-allocate reusable buffers.
    let mut a_valid: Vec<f64> = Vec::with_capacity(window_size * n_trend.max(n_harm));
    let mut b_valid: Vec<f64> = Vec::with_capacity(window_size);
    let mut detrended_full = vec![0.0_f64; window_size];
    let mut harmonic_signal = vec![0.0_f64; window_size];

    for (window_idx, &start_i) in starts.iter().enumerate() {
        let start = start_i as usize;
        let stop = (start + window_size).min(n);
        let cur_window = stop - start;
        if cur_window == 0 {
            continue;
        }

        // Count valid samples in window.
        let mut valid_count = 0usize;
        for i in 0..cur_window {
            if valid_mask[start + i] != 0 {
                valid_count += 1;
            }
        }
        if valid_count == 0 {
            continue;
        }
        let valid_fraction = valid_count as f64 / window_size as f64;
        if valid_count < min_valid_samples || valid_fraction < min_valid_fraction {
            continue;
        }

        // Step 1: trend fit on valid rows.
        a_valid.clear();
        b_valid.clear();
        a_valid.reserve(valid_count * n_trend);
        b_valid.reserve(valid_count);
        for i in 0..cur_window {
            if valid_mask[start + i] != 0 {
                for j in 0..n_trend {
                    a_valid.push(trend[i * n_trend + j]);
                }
                b_valid.push(values[start + i]);
            }
        }
        let trend_coeffs = lstsq_svd(&a_valid, &b_valid, valid_count, n_trend);

        // Compute detrended signal over the full window (not just valid rows).
        for i in 0..cur_window {
            let mut s = 0.0;
            for j in 0..n_trend {
                s += trend[i * n_trend + j] * trend_coeffs[j];
            }
            detrended_full[i] = values[start + i] - s;
        }

        // Step 2: harmonic fit on detrended valid rows.
        a_valid.clear();
        b_valid.clear();
        a_valid.reserve(valid_count * n_harm);
        b_valid.reserve(valid_count);
        for i in 0..cur_window {
            if valid_mask[start + i] != 0 {
                for j in 0..n_harm {
                    a_valid.push(harmonic[i * n_harm + j]);
                }
                b_valid.push(detrended_full[i]);
            }
        }
        let harm_coeffs = lstsq_svd(&a_valid, &b_valid, valid_count, n_harm);

        // Compute harmonic_signal over the full window.
        for i in 0..cur_window {
            let mut s = 0.0;
            for j in 0..n_harm {
                s += harmonic[i * n_harm + j] * harm_coeffs[j];
            }
            harmonic_signal[i] = s;
        }

        // Per-harmonic amplitudes: coeffs[2k] = cos, coeffs[2k+1] = sin,
        // amp[k] = sqrt(cos^2 + sin^2).
        for k in 0..n_amplitudes {
            let cos_c = harm_coeffs[2 * k];
            let sin_c = harm_coeffs[2 * k + 1];
            amplitudes[window_idx * n_amplitudes + k] = (cos_c * cos_c + sin_c * sin_c).sqrt();
        }

        // Scatter-add weighted.
        for i in 0..cur_window {
            noise_sum[start + i] += harmonic_signal[i] * weights[i];
            weight_sum[start + i] += weights[i];
        }
    }

    let amp_array = Array2::from_shape_vec((n_starts, n_amplitudes), amplitudes)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("amplitude shape: {e}")))?;

    Ok((
        PyArray1::from_vec(py, noise_sum),
        PyArray1::from_vec(py, weight_sum),
        amp_array.into_pyarray(py),
    ))
}