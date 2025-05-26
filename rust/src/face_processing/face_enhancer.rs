use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::path::Path;
use std::sync::Arc;
use anyhow::Result;
use log::{info, warn, error};

use crate::platform::{ExecutionProvider, get_optimal_provider};
use super::onnx_runtime::{OnnxSession, OnnxError, ModelManager};

#[pyclass]
pub struct FaceEnhancer {
    model_path: String,
    #[pyo3(get)]
    device: String,
    #[allow(dead_code)]
    session: Option<Arc<OnnxSession>>,
    #[allow(dead_code)]
    model_manager: Arc<ModelManager>,
}

#[pymethods]
impl FaceEnhancer {
    #[new]
    fn new(model_path: Option<String>, device: Option<String>) -> Self {
        let model_path = model_path.unwrap_or_else(|| "models/gfpgan_1.4.onnx".to_string());
        let device = device.unwrap_or_else(|| "cuda".to_string());
        
        let model_manager = Arc::new(ModelManager::new("models"));
        
        let provider = match device.as_str() {
            "cuda" => ExecutionProvider::CUDA,
            "coreml" => ExecutionProvider::CoreML,
            _ => ExecutionProvider::CPU,
        };
        
        let session = match model_manager.get_model_path(&model_path) {
            Ok(path) => {
                match OnnxSession::new(&path, Some(provider)) {
                    Ok(session) => Some(Arc::new(session)),
                    Err(e) => {
                        warn!("Failed to create ONNX session: {}", e);
                        None
                    }
                }
            },
            Err(e) => {
                warn!("Failed to get model path: {}", e);
                None
            }
        };
        
        FaceEnhancer {
            model_path,
            device,
            session,
            model_manager,
        }
    }

    fn process_frame(&self, py: Python, target_frame: &PyAny) -> PyResult<PyObject> {
        
        if let Some(session) = &self.session {
            info!("Using ONNX session with provider: {:?}", session.execution_provider());
        } else {
            warn!("No ONNX session available, returning unmodified frame");
        }
        
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
    
    fn get_model_info(&self, py: Python) -> PyResult<PyObject> {
        let info = PyDict::new(py);
        
        info.set_item("model_path", &self.model_path)?;
        info.set_item("device", &self.device)?;
        
        if let Some(session) = &self.session {
            info.set_item("provider", format!("{:?}", session.execution_provider()))?;
            info.set_item("session_active", true)?;
        } else {
            info.set_item("session_active", false)?;
        }
        
        Ok(info.into())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("FaceEnhancer(model_path='{}', device='{}')", self.model_path, self.device))
    }
}
