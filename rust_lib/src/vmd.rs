//! Variational mode decomposition (VMD) numerical kernel.
//!
//! The public function accepts one finite, contiguous f64 signal and returns
//! one row per mode.  NaN segmentation, timestamp handling, and user-facing
//! reconstruction choices deliberately remain in Python.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex64};

struct VmdOutput {
    modes: Vec<f64>,
    center_frequencies: Vec<f64>,
    iterations: usize,
    convergence: f64,
}

/// Decompose a finite real-valued signal using the standard VMD ADMM update.
///
/// Frequencies returned by this function are cycles per sample in ascending
/// order.  The Python layer converts them to Hz when a sampling rate is known.
#[pyfunction]
#[pyo3(name = "vmd_decompose_rs")]
#[pyo3(signature = (
    values,
    mode_count=4,
    alpha=2000.0,
    tau=0.0,
    dc=false,
    init="uniform",
    tolerance=1.0e-7,
    max_iter=500,
))]
#[allow(clippy::too_many_arguments)]
pub fn vmd_decompose_rs<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    mode_count: usize,
    alpha: f64,
    tau: f64,
    dc: bool,
    init: &str,
    tolerance: f64,
    max_iter: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, usize, f64)> {
    let input = values.as_slice()?;
    let result =
        py.detach(|| vmd_core(input, mode_count, alpha, tau, dc, init, tolerance, max_iter));
    let result = result.map_err(PyValueError::new_err)?;
    let sample_count = input.len();
    let modes = numpy::ndarray::Array2::from_shape_vec((mode_count, sample_count), result.modes)
        .map_err(|error| PyValueError::new_err(format!("VMD mode shape: {error}")))?;

    Ok((
        modes.into_pyarray(py).unbind(),
        PyArray1::from_vec(py, result.center_frequencies).unbind(),
        result.iterations,
        result.convergence,
    ))
}

#[allow(clippy::too_many_arguments)]
fn vmd_core(
    input: &[f64],
    mode_count: usize,
    alpha: f64,
    tau: f64,
    dc: bool,
    init: &str,
    tolerance: f64,
    max_iter: usize,
) -> Result<VmdOutput, String> {
    if input.len() < 4 {
        return Err("VMD requires at least four samples".to_owned());
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err("VMD input must be finite".to_owned());
    }
    if mode_count == 0 {
        return Err("mode_count must be positive".to_owned());
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err("alpha must be finite and positive".to_owned());
    }
    if !tau.is_finite() || tau < 0.0 {
        return Err("tau must be finite and non-negative".to_owned());
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err("tolerance must be finite and positive".to_owned());
    }
    if max_iter == 0 {
        return Err("max_iter must be positive".to_owned());
    }
    if init != "uniform" && init != "zero" {
        return Err("init must be 'uniform' or 'zero'".to_owned());
    }

    let sample_count = input.len();
    let fft_size = sample_count * 2;
    let half = fft_size / 2;

    // Mirror extension turns the endpoints into locally even boundaries before
    // applying the periodic FFT used by VMD.
    let mut mirrored = Vec::with_capacity(fft_size);
    mirrored.extend(input.iter().rev().map(|value| Complex64::new(*value, 0.0)));
    mirrored.extend(input.iter().map(|value| Complex64::new(*value, 0.0)));

    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(fft_size);
    let inverse = planner.plan_fft_inverse(fft_size);
    forward.process(&mut mirrored);

    // The ADMM update works on the positive-frequency analytic spectrum.
    let mut target = vec![Complex64::new(0.0, 0.0); half + 1];
    target.copy_from_slice(&mirrored[..=half]);
    let frequencies: Vec<f64> = (0..=half)
        .map(|index| index as f64 / fft_size as f64)
        .collect();

    let mut modes_hat = vec![Complex64::new(0.0, 0.0); mode_count * (half + 1)];
    let mut sum_modes = vec![Complex64::new(0.0, 0.0); half + 1];
    let mut lagrange = vec![Complex64::new(0.0, 0.0); half + 1];
    let mut omegas: Vec<f64> = match init {
        "uniform" => (0..mode_count)
            .map(|mode| 0.5 * mode as f64 / mode_count as f64)
            .collect(),
        "zero" => vec![0.0; mode_count],
        _ => unreachable!("validated above"),
    };
    if dc {
        omegas[0] = 0.0;
    }

    let mut convergence = f64::INFINITY;
    let mut iterations = 0usize;
    for iteration in 0..max_iter {
        let mut delta_norm = 0.0;
        let mut previous_norm = 0.0;

        for mode in 0..mode_count {
            let offset = mode * (half + 1);
            for index in 0..=half {
                let old = modes_hat[offset + index];
                let without_current = sum_modes[index] - old;
                let distance = frequencies[index] - omegas[mode];
                let denominator = 1.0 + alpha * distance * distance;
                let new = (target[index] - without_current - lagrange[index] * 0.5) / denominator;
                let difference = new - old;
                delta_norm += difference.norm_sqr();
                previous_norm += old.norm_sqr();
                modes_hat[offset + index] = new;
                sum_modes[index] = without_current + new;
            }

            if !(dc && mode == 0) {
                let mut weighted_frequency = 0.0;
                let mut energy = 0.0;
                for index in 1..=half {
                    let magnitude = modes_hat[offset + index].norm_sqr();
                    weighted_frequency += frequencies[index] * magnitude;
                    energy += magnitude;
                }
                if energy > f64::EPSILON {
                    omegas[mode] = weighted_frequency / energy;
                }
            }
        }

        if tau > 0.0 {
            for index in 0..=half {
                lagrange[index] += (sum_modes[index] - target[index]) * tau;
            }
        }
        convergence = (delta_norm / (previous_norm + f64::EPSILON)).sqrt();
        iterations = iteration + 1;
        if convergence < tolerance {
            break;
        }
    }

    // Stable ascending frequency order makes the UI's "keep modes" setting
    // deterministic even if the ADMM updates exchange two nearby components.
    let mut order: Vec<usize> = (0..mode_count).collect();
    order.sort_by(|left, right| omegas[*left].total_cmp(&omegas[*right]));

    let mut output = vec![0.0; mode_count * sample_count];
    let mut output_frequencies = vec![0.0; mode_count];
    let inverse_scale = 1.0 / fft_size as f64;
    for (output_mode, source_mode) in order.into_iter().enumerate() {
        output_frequencies[output_mode] = omegas[source_mode];
        let offset = source_mode * (half + 1);
        let mut full_spectrum = vec![Complex64::new(0.0, 0.0); fft_size];
        full_spectrum[..=half].copy_from_slice(&modes_hat[offset..=offset + half]);
        for index in 1..half {
            full_spectrum[fft_size - index] = modes_hat[offset + index].conj();
        }
        inverse.process(&mut full_spectrum);
        for index in 0..sample_count {
            output[output_mode * sample_count + index] =
                full_spectrum[sample_count + index].re * inverse_scale;
        }
    }

    Ok(VmdOutput {
        modes: output,
        center_frequencies: output_frequencies,
        iterations,
        convergence,
    })
}

#[cfg(test)]
mod tests {
    use super::vmd_core;

    #[test]
    fn separates_two_stationary_components() {
        let input: Vec<f64> = (0..512)
            .map(|index| {
                let sample = index as f64;
                (std::f64::consts::TAU * 0.04 * sample).sin()
                    + 0.4 * (std::f64::consts::TAU * 0.20 * sample).sin()
            })
            .collect();
        let result = vmd_core(&input, 2, 2000.0, 0.0, false, "uniform", 1.0e-7, 800)
            .expect("valid VMD input");

        assert_eq!(result.modes.len(), input.len() * 2);
        assert!(result.center_frequencies[0] < result.center_frequencies[1]);
        assert!((result.center_frequencies[0] - 0.04).abs() < 0.03);
        assert!((result.center_frequencies[1] - 0.20).abs() < 0.04);
    }
}
