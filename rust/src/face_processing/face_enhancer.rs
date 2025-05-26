use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;

#[pyclass]
pub struct FaceEnhancer {
    model_path: String,
    #[pyo3(get)]
    device: String,
}

#[pymethods]
impl FaceEnhancer {
    #[new]
    fn new(model_path: Option<String>, device: Option<String>) -> Self {
        FaceEnhancer {
            model_path: model_path.unwrap_or_else(|| "models/gfpgan_1.4.onnx".to_string()),
            device: device.unwrap_or_else(|| "cuda".to_string()),
        }
    }

    fn process_frame(&self, py: Python, target_frame: &PyAny) -> PyResult<PyObject> {
        
        Ok(target_frame.into_py(py))
    }

    #[getter]
    fn name(&self) -> PyResult<String> {
        Ok("CAMSHOW.FACE-ENHANCER".to_string())
    }

    fn has_cuda(&self) -> PyResult<bool> {
        Ok(self.device == "cuda")
    }

    fn has_coreml(&self) -> PyResult<bool> {
        Ok(self.device == "coreml")
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("FaceEnhancer(model_path='{}', device='{}')", self.model_path, self.device))
    }
}
