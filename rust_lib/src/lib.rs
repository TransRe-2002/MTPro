mod decimate;
mod mat_loader;

use pyo3::prelude::*;

/// MTPro Rust extensions — high-performance signal processing.
#[pymodule]
mod rust_lib {
    #[pymodule_export]
    use super::decimate::minmax_decimate_rs;

    #[pymodule_export]
    use super::decimate::build_lod_pyramid_rs;

    #[pymodule_export]
    use super::decimate::apply_nan_intervals_rs;

    #[pymodule_export]
    use super::mat_loader::probe_mat_v73;
}
