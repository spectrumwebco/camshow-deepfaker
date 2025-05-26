use pyo3::prelude::*;
use pyo3::wrap_pymodule;

mod face_processing;
mod video_capture;
mod ui;

#[pymodule]
fn camshow_deepfaker(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(face_processing::face_processing_module))?;
    m.add_wrapped(wrap_pymodule!(video_capture::video_capture_module))?;
    
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}
