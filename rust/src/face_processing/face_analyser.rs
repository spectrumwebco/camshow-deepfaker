use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;

#[pyclass]
pub struct FaceAnalyser {
    model_path: String,
    #[pyo3(get)]
    device: String,
}

#[pymethods]
impl FaceAnalyser {
    #[new]
    fn new(model_path: Option<String>, device: Option<String>) -> Self {
        FaceAnalyser {
            model_path: model_path.unwrap_or_else(|| "models/buffalo_l.onnx".to_string()),
            device: device.unwrap_or_else(|| "cuda".to_string()),
        }
    }

    fn detect_faces(&self, py: Python, frame: &PyAny) -> PyResult<PyObject> {
        
        Ok(vec![].into_py(py))
    }

    fn get_landmarks(&self, py: Python, face: &PyAny) -> PyResult<PyObject> {
        
        Ok(vec![].into_py(py))
    }

    #[getter]
    fn name(&self) -> PyResult<String> {
        Ok("CAMSHOW.FACE-ANALYSER".to_string())
    }

    fn has_cuda(&self) -> PyResult<bool> {
        Ok(self.device == "cuda")
    }

    fn has_coreml(&self) -> PyResult<bool> {
        Ok(self.device == "coreml")
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("FaceAnalyser(model_path='{}', device='{}')", self.model_path, self.device))
    }
}
