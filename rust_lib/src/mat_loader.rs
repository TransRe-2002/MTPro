//! MATLAB v7.3 (HDF5) file loader.
//!
//! ``probe_mat_v73`` returns top-level variable names.
//! Full loader is WIP — uses Python h5py as fallback for now.

use hdf5_pure::File;
use pyo3::prelude::*;
use std::path::Path;

/// List top-level variable names in a MATLAB v7.3 file.
#[pyfunction]
#[pyo3(name = "probe_mat_v73")]
pub fn probe_mat_v73(_py: Python<'_>, path: &str) -> PyResult<Vec<String>> {
    let file = File::open(Path::new(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))?;
    let root = file.root();
    let mut names: Vec<String> = root.datasets().unwrap_or_default();
    names.extend(root.groups().unwrap_or_default());
    Ok(names.into_iter().filter(|n| !n.starts_with('#')).collect())
}
