use numpy::array::PyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

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
                if v < minv { minv = v; imin = j; }
                if v > maxv { maxv = v; imax = j; }
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
