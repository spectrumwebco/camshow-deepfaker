use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod video_capturer;

use video_capturer::VideoCapturer;

#[pymodule]
pub fn video_capture_module(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VideoCapturer>()?;
    
    m.add_function(wrap_pyfunction!(list_cameras, m)?)?;
    m.add_function(wrap_pyfunction!(get_camera_properties, m)?)?;
    
    Ok(())
}

#[pyfunction]
fn list_cameras(py: Python) -> PyResult<PyObject> {
    let cameras = vec![("0".to_string(), "Default Camera".to_string())];
    Ok(cameras.into_py(py))
}

#[pyfunction]
fn get_camera_properties(py: Python, device_index: usize) -> PyResult<PyObject> {
    let properties = pyo3::types::PyDict::new(py);
    properties.set_item("width", 1280)?;
    properties.set_item("height", 720)?;
    properties.set_item("fps", 30)?;
    Ok(properties.into_py(py))
}
