//! Step-removal algorithms (Rust acceleration for core/algorithms/step.py).
//!
//! Provides `detect_step_indices_rs` and `windowed_mean_destep_rs` used by
//! the Python layer.

use numpy::array::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Bound;

/// Diagnostic info for one candidate step in windowed-mean destep.
/// Mirrors the Python `WindowedStepCandidate` dataclass fields.
#[pyclass(frozen, name = "WindowedStepCandidateRs")]
pub struct WindowedStepCandidateRs {
    #[pyo3(get)]
    index: i64,
    #[pyo3(get)]
    raw_diff: f64,
    /// `None` when step value couldn't be estimated (empty window);
    /// `Some(v)` otherwise (including when below threshold).
    #[pyo3(get)]
    step_value: Option<f64>,
    #[pyo3(get)]
    before_count: i64,
    #[pyo3(get)]
    after_count: i64,
    #[pyo3(get)]
    applied: bool,
}

impl WindowedStepCandidateRs {
    fn new(
        index: usize,
        raw_diff: f64,
        step_value: Option<f64>,
        before_count: usize,
        after_count: usize,
        applied: bool,
    ) -> Self {
        Self {
            index: index as i64,
            raw_diff,
            step_value,
            before_count: before_count as i64,
            after_count: after_count as i64,
            applied,
        }
    }
}

/// Core: find the `max_steps` largest peaks in `diff_abs` exceeding `min_offset`.
///
/// Iteratively argmax + neutralize. K is small (default 20) so this O(K·N)
/// pass beats any priority-queue while keeping the implementation trivial.
fn detect_step_core(diff_abs: &[f64], min_offset: f64, max_steps: usize) -> Vec<i64> {
    if diff_abs.is_empty() || max_steps == 0 {
        return Vec::new();
    }
    let mut working = diff_abs.to_vec();
    let limit = max_steps.min(working.len());
    let mut indices: Vec<i64> = Vec::with_capacity(limit);

    for _ in 0..limit {
        let (best_i, best_v) = working
            .iter()
            .enumerate()
            .fold((0usize, f64::MIN), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            });
        if best_v <= min_offset {
            break;
        }
        indices.push(best_i as i64);
        working[best_i] = f64::MIN / 2.0;
    }
    indices.sort_unstable();
    indices
}

/// Python-callable: detect_step_indices_rs(diff_abs, min_offset, max_steps).
///
/// Returns sorted ascending indices (as PyArray1<i64>) of the largest peaks
/// in `diff_abs` that exceed `min_offset`, up to `max_steps` entries.
#[pyfunction]
#[pyo3(name = "detect_step_indices_rs")]
pub fn detect_step_indices_rs<'py>(
    py: Python<'py>,
    diff_abs: numpy::PyReadonlyArray1<'py, f64>,
    min_offset: f64,
    max_steps: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let slice = diff_abs.as_slice()?;
    let indices = detect_step_core(slice, min_offset, max_steps);
    Ok(PyArray1::from_vec(py, indices))
}

/// Core: windowed-mean destep. Computes per-step before/after window means,
/// builds the cumulative `shift` array in O(N + K) — scatters per-step deltas
/// as a single marker per step, then a single prefix-sum pass — instead of
/// O(N·K) repeated tail writes.
///
/// Args:
///   - compact: finite-only input series (length N)
///   - diff_abs: abs-diff, length N-1 (used for raw_diff diagnostic only)
///   - step_indices: candidate indices (already sorted)
///   - start_indices, end_indices: per-step window boundary arrays
///   - window: (unused, kept for API symmetry with the Python implementation)
///   - min_offset: rejection threshold for step_value
///
/// Returns (shift: Vec<f64>, applied_steps: Vec<(i64, f64)>, candidates).
fn windowed_mean_destep_core(
    compact: &[f64],
    diff_abs: &[f64],
    step_indices: &[i64],
    start_indices: &[i64],
    end_indices: &[i64],
    min_offset: f64,
) -> (Vec<f64>, Vec<(i64, f64)>, Vec<WindowedStepCandidateRs>) {
    let n = compact.len();
    let mut shift = vec![0.0_f64; n];
    let mut applied_steps: Vec<(i64, f64)> = Vec::new();
    let mut candidates: Vec<WindowedStepCandidateRs> = Vec::with_capacity(step_indices.len());

    for (k, &step_idx_i) in step_indices.iter().enumerate() {
        let step_idx = step_idx_i as usize;
        let win_start = start_indices[k] as usize;
        let win_end = end_indices[k] as usize;

        let before = &compact[win_start..step_idx.saturating_add(1).min(n)];
        let after_start = (step_idx + 1).min(n);
        let after = &compact[after_start..win_end.min(n)];

        let raw_diff = if step_idx < diff_abs.len() { diff_abs[step_idx] } else { 0.0 };

        if before.is_empty() || after.is_empty() {
            candidates.push(WindowedStepCandidateRs::new(
                step_idx, raw_diff, None, before.len(), after.len(), false,
            ));
            continue;
        }

        let before_mean: f64 = before.iter().copied().sum::<f64>() / before.len() as f64;
        let after_mean: f64 = after.iter().copied().sum::<f64>() / after.len() as f64;
        let step_value = after_mean - before_mean;

        if step_value.abs() < min_offset {
            candidates.push(WindowedStepCandidateRs::new(
                step_idx, raw_diff, Some(step_value), before.len(), after.len(), false,
            ));
            continue;
        }

        // Scatter delta marker at step_idx+1; prefix-sum pass completes shift construction.
        let apply_at = (step_idx + 1).min(n);
        shift[apply_at] -= step_value;
        applied_steps.push((step_idx as i64, step_value));
        candidates.push(WindowedStepCandidateRs::new(
            step_idx, raw_diff, Some(step_value), before.len(), after.len(), true,
        ));
    }

    // Single prefix-sum pass — O(N), regardless of K.
    let mut running = 0.0_f64;
    for v in shift.iter_mut() {
        running += *v;
        *v = running;
    }
    (shift, applied_steps, candidates)
}

/// Python-callable: windowed_mean_destep_rs(compact, diff_abs, step_indices,
/// start_indices, end_indices, min_offset).
///
/// Returns (shift: PyArray1<f64>, applied_steps: Vec<(i64, f64)>,
///          candidates: PyList<WindowedStepCandidateRs>).
#[pyfunction]
#[pyo3(name = "windowed_mean_destep_rs")]
#[allow(clippy::type_complexity)]
pub fn windowed_mean_destep_rs<'py>(
    py: Python<'py>,
    compact: numpy::PyReadonlyArray1<'py, f64>,
    diff_abs: numpy::PyReadonlyArray1<'py, f64>,
    step_indices: numpy::PyReadonlyArray1<'py, i64>,
    start_indices: numpy::PyReadonlyArray1<'py, i64>,
    end_indices: numpy::PyReadonlyArray1<'py, i64>,
    min_offset: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Vec<(i64, f64)>,
    Bound<'py, PyList>,
)> {
    let compact = compact.as_slice()?;
    let diff_abs = diff_abs.as_slice()?;
    let step_indices = step_indices.as_slice()?;
    let start_indices = start_indices.as_slice()?;
    let end_indices = end_indices.as_slice()?;

    let (shift, applied_steps, candidates) =
        windowed_mean_destep_core(compact, diff_abs, step_indices, start_indices, end_indices, min_offset);

    let candidates_py = PyList::new(py, candidates)?;

    Ok((PyArray1::from_vec(py, shift), applied_steps, candidates_py))
}