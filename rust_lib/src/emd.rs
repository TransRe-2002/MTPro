//! Empirical mode decomposition (EMD) numerical kernel.
//!
//! This is a classical spline-envelope EMD implementation for one finite
//! channel.  The Python layer owns NaN segmentation and selected-IMF
//! reconstruction, keeping this module independent from pandas and Qt.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

struct EmdOutput {
    imfs: Vec<f64>,
    residue: Vec<f64>,
    siftings: Vec<usize>,
}

/// Decompose a finite real-valued signal into empirical mode functions.
///
/// IMF rows are in extraction order, conventionally high frequency to low
/// frequency.  Their sum plus ``residue`` reconstructs the input signal.
#[pyfunction]
#[pyo3(name = "emd_decompose_rs")]
#[pyo3(signature = (
    values,
    max_imfs=8,
    max_siftings=100,
    sift_tolerance=0.2,
))]
pub fn emd_decompose_rs<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    max_imfs: usize,
    max_siftings: usize,
    sift_tolerance: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, Vec<usize>)> {
    let input = values.as_slice()?;
    let result = py.detach(|| emd_core(input, max_imfs, max_siftings, sift_tolerance));
    let result = result.map_err(PyValueError::new_err)?;
    let sample_count = input.len();
    let imf_count = result.siftings.len();
    let imfs = numpy::ndarray::Array2::from_shape_vec((imf_count, sample_count), result.imfs)
        .map_err(|error| PyValueError::new_err(format!("EMD IMF shape: {error}")))?;

    Ok((
        imfs.into_pyarray(py).unbind(),
        PyArray1::from_vec(py, result.residue).unbind(),
        result.siftings,
    ))
}

fn emd_core(
    input: &[f64],
    max_imfs: usize,
    max_siftings: usize,
    sift_tolerance: f64,
) -> Result<EmdOutput, String> {
    if input.len() < 4 {
        return Err("EMD requires at least four samples".to_owned());
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err("EMD input must be finite".to_owned());
    }
    if max_imfs == 0 {
        return Err("max_imfs must be positive".to_owned());
    }
    if max_siftings == 0 {
        return Err("max_siftings must be positive".to_owned());
    }
    if !sift_tolerance.is_finite() || sift_tolerance <= 0.0 {
        return Err("sift_tolerance must be finite and positive".to_owned());
    }

    let mut residue = input.to_vec();
    let mut imfs = Vec::new();
    let mut siftings = Vec::new();
    for _ in 0..max_imfs {
        let (maxima, minima) = extrema_indices(&residue);
        if maxima.is_empty() || minima.is_empty() {
            break;
        }
        let (imf, count) = sift(&residue, max_siftings, sift_tolerance)?;
        for (residual, component) in residue.iter_mut().zip(&imf) {
            *residual -= *component;
        }
        imfs.extend(imf);
        siftings.push(count);

        let (maxima, minima) = extrema_indices(&residue);
        if maxima.is_empty() || minima.is_empty() {
            break;
        }
    }

    Ok(EmdOutput {
        imfs,
        residue,
        siftings,
    })
}

fn sift(source: &[f64], max_siftings: usize, tolerance: f64) -> Result<(Vec<f64>, usize), String> {
    let mut component = source.to_vec();
    for iteration in 0..max_siftings {
        let (maxima, minima) = extrema_indices(&component);
        if maxima.is_empty() || minima.is_empty() {
            return Ok((component, iteration));
        }
        let upper = spline_envelope(&component, &maxima)?;
        let lower = spline_envelope(&component, &minima)?;
        let mut next = vec![0.0; component.len()];
        let mut delta_norm = 0.0;
        let mut component_norm = 0.0;
        for index in 0..component.len() {
            let mean = 0.5 * (upper[index] + lower[index]);
            next[index] = component[index] - mean;
            let delta = component[index] - next[index];
            delta_norm += delta * delta;
            component_norm += component[index] * component[index];
        }
        let relative_change = delta_norm / (component_norm + f64::EPSILON);
        component = next;

        let (next_maxima, next_minima) = extrema_indices(&component);
        let extrema_count = next_maxima.len() + next_minima.len();
        let zero_crossings = zero_crossing_count(&component);
        if relative_change <= tolerance
            && (extrema_count as isize - zero_crossings as isize).abs() <= 1
        {
            return Ok((component, iteration + 1));
        }
    }
    Ok((component, max_siftings))
}

fn extrema_indices(values: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let mut maxima = Vec::new();
    let mut minima = Vec::new();
    let mut index = 1usize;
    while index + 1 < values.len() {
        let start = index;
        while index + 1 < values.len() - 1 && values[index] == values[index + 1] {
            index += 1;
        }
        let end = index;
        let before = values[start] - values[start - 1];
        let after = values[end + 1] - values[end];
        let midpoint = (start + end) / 2;
        if before > 0.0 && after < 0.0 {
            maxima.push(midpoint);
        } else if before < 0.0 && after > 0.0 {
            minima.push(midpoint);
        }
        index += 1;
    }
    (maxima, minima)
}

fn zero_crossing_count(values: &[f64]) -> usize {
    let mut previous = 0i8;
    let mut count = 0usize;
    for value in values {
        let sign = if *value > 0.0 {
            1
        } else if *value < 0.0 {
            -1
        } else {
            0
        };
        if sign != 0 {
            if previous != 0 && sign != previous {
                count += 1;
            }
            previous = sign;
        }
    }
    count
}

fn spline_envelope(values: &[f64], extrema: &[usize]) -> Result<Vec<f64>, String> {
    let mut nodes: Vec<(usize, f64)> = Vec::with_capacity(extrema.len() + 2);
    nodes.push((0, values[0]));
    for index in extrema {
        if *index > 0 && *index + 1 < values.len() {
            nodes.push((*index, values[*index]));
        }
    }
    nodes.push((values.len() - 1, values[values.len() - 1]));
    nodes.sort_unstable_by_key(|(index, _)| *index);
    nodes.dedup_by_key(|(index, _)| *index);

    if nodes.len() < 2 {
        return Err("EMD envelope has fewer than two spline nodes".to_owned());
    }
    if nodes.len() == 2 {
        return Ok(linear_envelope(values.len(), nodes[0], nodes[1]));
    }

    let point_count = nodes.len();
    let x: Vec<f64> = nodes.iter().map(|(index, _)| *index as f64).collect();
    let a: Vec<f64> = nodes.iter().map(|(_, value)| *value).collect();
    let mut h = vec![0.0; point_count - 1];
    for index in 0..point_count - 1 {
        h[index] = x[index + 1] - x[index];
    }

    let mut alpha = vec![0.0; point_count];
    for index in 1..point_count - 1 {
        alpha[index] = 3.0 / h[index] * (a[index + 1] - a[index])
            - 3.0 / h[index - 1] * (a[index] - a[index - 1]);
    }
    let mut lower = vec![1.0; point_count];
    let mut mu = vec![0.0; point_count];
    let mut z = vec![0.0; point_count];
    for index in 1..point_count - 1 {
        lower[index] = 2.0 * (x[index + 1] - x[index - 1]) - h[index - 1] * mu[index - 1];
        if lower[index].abs() <= f64::EPSILON {
            return Ok(linear_piecewise_envelope(values.len(), &nodes));
        }
        mu[index] = h[index] / lower[index];
        z[index] = (alpha[index] - h[index - 1] * z[index - 1]) / lower[index];
    }
    let mut c = vec![0.0; point_count];
    let mut b = vec![0.0; point_count - 1];
    let mut d = vec![0.0; point_count - 1];
    for index in (0..point_count - 1).rev() {
        c[index] = z[index] - mu[index] * c[index + 1];
        b[index] =
            (a[index + 1] - a[index]) / h[index] - h[index] * (c[index + 1] + 2.0 * c[index]) / 3.0;
        d[index] = (c[index + 1] - c[index]) / (3.0 * h[index]);
    }

    let mut output = vec![0.0; values.len()];
    let mut interval = 0usize;
    for (sample, output_value) in output.iter_mut().enumerate() {
        while interval + 1 < point_count - 1 && sample as f64 > x[interval + 1] {
            interval += 1;
        }
        let delta = sample as f64 - x[interval];
        *output_value = a[interval]
            + b[interval] * delta
            + c[interval] * delta * delta
            + d[interval] * delta * delta * delta;
    }
    Ok(output)
}

fn linear_envelope(length: usize, left: (usize, f64), right: (usize, f64)) -> Vec<f64> {
    let span = (right.0 - left.0) as f64;
    (0..length)
        .map(|index| {
            let weight = (index.saturating_sub(left.0) as f64 / span).clamp(0.0, 1.0);
            left.1 + weight * (right.1 - left.1)
        })
        .collect()
}

fn linear_piecewise_envelope(length: usize, nodes: &[(usize, f64)]) -> Vec<f64> {
    let mut output = vec![0.0; length];
    let mut interval = 0usize;
    for (sample, output_value) in output.iter_mut().enumerate() {
        while interval + 1 < nodes.len() - 1 && sample > nodes[interval + 1].0 {
            interval += 1;
        }
        let left = nodes[interval];
        let right = nodes[interval + 1];
        let span = (right.0 - left.0) as f64;
        let weight = (sample.saturating_sub(left.0) as f64 / span).clamp(0.0, 1.0);
        *output_value = left.1 + weight * (right.1 - left.1);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::emd_core;

    #[test]
    fn imfs_and_residue_reconstruct_input() {
        let input: Vec<f64> = (0..512)
            .map(|index| {
                let sample = index as f64;
                0.002 * sample
                    + (std::f64::consts::TAU * 0.04 * sample).sin()
                    + 0.35 * (std::f64::consts::TAU * 0.20 * sample).sin()
            })
            .collect();
        let result = emd_core(&input, 4, 100, 0.2).expect("valid EMD input");

        assert!(!result.siftings.is_empty());
        for index in 0..input.len() {
            let reconstructed = result
                .imfs
                .chunks(input.len())
                .map(|imf| imf[index])
                .sum::<f64>()
                + result.residue[index];
            assert!((reconstructed - input[index]).abs() < 1.0e-10);
        }
    }
}
