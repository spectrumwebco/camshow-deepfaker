use pyo3::prelude::*;
use pyo3::wrap_pymodule;

mod face_processing;
mod video_capture;
mod ui;

#[pymodule]
fn camshow_deepfaker(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(face_processing::face_processing_module))?;
    m.add_wrapped(wrap_pymodule!(video_capture::video_capture_module))?;
    
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    py.import("sys")?.getattr("modules")?.set_item(
        "camshow_deepfaker_rs.face_processing_module",
        py.import("camshow_deepfaker_rs")?.getattr("camshow_deepfaker")?.getattr("face_processing_module")?
    )?;
    
    py.import("sys")?.getattr("modules")?.set_item(
        "camshow_deepfaker_rs.video_capture_module",
        py.import("camshow_deepfaker_rs")?.getattr("camshow_deepfaker")?.getattr("video_capture_module")?
    )?;
    
    Ok(())
}
