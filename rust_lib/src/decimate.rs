use numpy::array::PyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

const PARALLEL_BOXCAR_MIN_POINTS: usize = 65_536;

/// NaN-aware boxcar decimation used by the destructive data preprocessor.
///
/// Each non-overlapping chunk produces one output sample. Finite values are
/// averaged in input order; a chunk is rejected when its missing fraction is
/// above `max_missing_fraction`. `par_chunks` preserves chunk order when
/// collected into a `Vec`, so the result is deterministic.
fn nan_boxcar_decimate_core(y: &[f64], factor: usize, max_missing_fraction: f64) -> Vec<f64> {
    let reduce_window = |window: &[f64]| {
        let mut sum = 0.0_f64;
        let mut good_count = 0usize;
        for &value in window {
            if value.is_finite() {
                sum += value;
                good_count += 1;
            }
        }
        if good_count == 0 {
            return f64::NAN;
        }
        let missing_fraction = 1.0 - good_count as f64 / window.len() as f64;
        if missing_fraction <= max_missing_fraction {
            sum / good_count as f64
        } else {
            f64::NAN
        }
    };

    if y.len() < PARALLEL_BOXCAR_MIN_POINTS {
        y.chunks(factor).map(reduce_window).collect()
    } else {
        y.par_chunks(factor).map(reduce_window).collect()
    }
}

/// Python-callable NaN-aware boxcar decimation (zero-copy input, Rayon output).
///
/// The NumPy array stays read-borrowed while `py.detach` releases the GIL, so
/// other Python threads can run while Rayon processes independent windows.
#[pyfunction]
#[pyo3(name = "nan_boxcar_decimate_rs")]
#[pyo3(signature = (y, factor, max_missing_fraction=1.0))]
pub fn nan_boxcar_decimate_rs<'py>(
    py: Python<'py>,
    y: numpy::PyReadonlyArray1<'py, f64>,
    factor: usize,
    max_missing_fraction: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if factor < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "factor must be >= 2, got {factor}"
        )));
    }
    if !max_missing_fraction.is_finite() || !(0.0..=1.0).contains(&max_missing_fraction) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "max_missing_fraction must be between 0 and 1, got {max_missing_fraction}"
        )));
    }

    let input = y.as_slice()?;
    let output = py.detach(|| nan_boxcar_decimate_core(input, factor, max_missing_fraction));
    Ok(PyArray1::from_vec(py, output).unbind())
}

/// Min/max envelope decimation (Rust core, parallel).
fn decimate_core(x: &[f64], y: &[f64], target_bins: usize) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    if n <= target_bins * 2 {
        return (x.to_vec(), y.to_vec());
    }
    let bins = target_bins.max(1).min(n / 2);
    let chunk_size = n / bins;

    let chunks: Vec<(usize, usize)> = (0..bins)
        .map(|i| {
            let start = i * chunk_size;
            let end = if i == bins - 1 { n } else { start + chunk_size };
            (start, end)
        })
        .collect();

    let results: Vec<(Vec<f64>, Vec<f64>)> = chunks
        .par_iter()
        .map(|&(start, end)| {
            let yc = &y[start..end];
            if yc.len() <= 1 {
                return (vec![x[start]], vec![y[start]]);
            }
            let mut imin = 0usize;
            let mut imax = 0usize;
            let mut minv = f64::MAX;
            let mut maxv = f64::MIN;
            for (j, &v) in yc.iter().enumerate() {
                if v < minv {
                    minv = v;
                    imin = j;
                }
                if v > maxv {
                    maxv = v;
                    imax = j;
                }
            }
            let a = start + imin;
            let b = start + imax;
            if a < b {
                (vec![x[a], x[b]], vec![y[a], y[b]])
            } else if b < a {
                (vec![x[b], x[a]], vec![y[b], y[a]])
            } else {
                (vec![x[a]], vec![y[a]])
            }
        })
        .collect();

    let mut rx = Vec::with_capacity(bins * 2);
    let mut ry = Vec::with_capacity(bins * 2);
    for (cx, cy) in results {
        rx.extend(cx);
        ry.extend(cy);
    }
    (rx, ry)
}

/// Min/max envelope decimation (Python-callable, zero-copy input via numpy crate).
#[pyfunction]
#[pyo3(name = "minmax_decimate_rs")]
pub fn minmax_decimate_rs<'py>(
    py: Python<'py>,
    x: numpy::PyReadonlyArray1<'py, f64>,
    y: numpy::PyReadonlyArray1<'py, f64>,
    target_bins: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (rx, ry) = decimate_core(x.as_slice()?, y.as_slice()?, target_bins);
    Ok((
        PyArray1::from_vec(py, rx).unbind(),
        PyArray1::from_vec(py, ry).unbind(),
    ))
}

/// LOD pyramid builder (Python-callable, zero-copy input).
///
/// `max_finest_points`（截顶）：点数超过该上限的顶层不存储——但链式
/// 推导照常进行（更粗的层仍由完整精度的上一层导出，质量不受影响）。
/// 用于压低超大数据集（如 5s 采样年度文件）的金字塔内存。
#[pyfunction]
#[pyo3(name = "build_lod_pyramid_rs")]
#[pyo3(signature = (x, y, max_finest_points=None))]
pub fn build_lod_pyramid_rs<'py>(
    py: Python<'py>,
    x: numpy::PyReadonlyArray1<'py, f64>,
    y: numpy::PyReadonlyArray1<'py, f64>,
    max_finest_points: Option<usize>,
) -> PyResult<Vec<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let n = xs.len();
    let mut levels: Vec<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> = Vec::new();
    let mut cur_x = xs.to_vec();
    let mut cur_y = ys.to_vec();
    let mut target = (n / 5).max(256);

    while target >= 256 && cur_x.len() > target * 2 {
        let (rx, ry) = decimate_core(&cur_x, &cur_y, target);
        let keep = max_finest_points.map_or(true, |cap| rx.len() <= cap);
        if keep {
            levels.push((
                PyArray1::from_vec(py, rx.clone()).unbind(),
                PyArray1::from_vec(py, ry.clone()).unbind(),
            ));
        }
        cur_x = rx;
        cur_y = ry;
        target = (cur_x.len() / 5).max(256);
    }
    Ok(levels)
}

/// Set values to NaN for all indices in the given intervals (zero-copy, parallel).
///
/// Takes a mutable numpy array and a list of ``(start, end)`` interval tuples.
/// All values in each interval are set to ``f64::NAN``.
/// This eliminates the ``np.arange + np.concatenate + fancy indexing`` chain
/// used in ``modify_selected_data``.
///
/// Call from Python::
///
///     intervals = [(0, 1000), (5000, 6000)]
///     rust_lib.apply_nan_intervals_rs(y_data, intervals)
#[pyfunction]
#[pyo3(name = "apply_nan_intervals_rs")]
pub fn apply_nan_intervals_rs<'py>(
    _py: Python<'py>,
    mut y: numpy::PyReadwriteArray1<'py, f64>,
    intervals: Vec<(usize, usize)>,
) -> PyResult<()> {
    let slice = y.as_slice_mut()?;
    let n = slice.len();
    for &(start, end) in &intervals {
        let e = end.min(n);
        let s = start.min(e);
        slice[s..e].fill(f64::NAN);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::nan_boxcar_decimate_core;

    #[test]
    fn boxcar_averages_complete_and_partial_windows() {
        let y = [0.0, 1.0, 2.0, 3.0, 4.0];
        let out = nan_boxcar_decimate_core(&y, 2, 0.25);
        assert_eq!(out, vec![0.5, 2.5, 4.0]);
    }

    #[test]
    fn boxcar_accepts_threshold_and_rejects_excess_missing() {
        let y = [1.0, f64::NAN, 3.0, 4.0, 5.0, f64::NAN, f64::NAN, 8.0];
        let out = nan_boxcar_decimate_core(&y, 4, 0.25);
        assert_eq!(out[0], 8.0 / 3.0);
        assert!(out[1].is_nan());
    }

    #[test]
    fn boxcar_treats_non_finite_values_as_missing() {
        let y = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0];
        let out = nan_boxcar_decimate_core(&y, 2, 1.0);
        assert!(out[0].is_nan());
        assert_eq!(out[1], 1.0);
    }
}
