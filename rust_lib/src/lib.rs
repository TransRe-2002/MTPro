mod decimate;
mod diurnal;
mod emd;
mod mat_loader;
mod mt_text;
mod step;
mod vmd;
mod xlsx;

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
    use super::decimate::nan_boxcar_decimate_rs;

    #[pymodule_export]
    use super::emd::emd_decompose_rs;

    #[pymodule_export]
    use super::mat_loader::probe_mat_v73;

    #[pymodule_export]
    use super::mt_text::read_mt_txt_rs;

    #[pymodule_export]
    use super::step::detect_step_indices_rs;

    #[pymodule_export]
    use super::step::windowed_mean_destep_rs;

    #[pymodule_export]
    use super::vmd::vmd_decompose_rs;

    #[pymodule_export]
    use super::diurnal::subtract_periodic_signal_series_rs;

    #[pymodule_export]
    use super::xlsx::read_xlsx_sheet_inline_str_rs;
}
