use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;
use log::{info, warn, error};
use ndarray::{Array, ArrayD, Dimension, IxDyn};

use crate::platform::{PlatformExecutionProvider, get_optimal_provider};
use super::onnx_runtime::{OnnxSession, OnnxError, ModelManager};

#[pyclass]
pub struct FaceSwapper {
    model_path: String,
    #[pyo3(get)]
    device: String,
    #[allow(dead_code)]
    session: Option<Arc<OnnxSession>>,
    #[allow(dead_code)]
    model_manager: Arc<ModelManager>,
}

#[pymethods]
impl FaceSwapper {
    #[new]
    fn new(model_path: Option<String>, device: Option<String>) -> Self {
        let model_path = model_path.unwrap_or_else(|| "models/inswapper_128.onnx".to_string());
        let device = device.unwrap_or_else(|| "cuda".to_string());
        
        let model_manager = Arc::new(ModelManager::new("models"));
        
        let provider = match device.as_str() {
            "cuda" => PlatformExecutionProvider::CUDA,
            "coreml" => PlatformExecutionProvider::CoreML,
            _ => PlatformExecutionProvider::CPU,
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
        
        FaceSwapper {
            model_path,
            device,
            session,
            model_manager,
        }
    }

    fn process_frame(&self, py: Python, source_face: &PyAny, target_frame: &PyAny) -> PyResult<PyObject> {
        if let Some(session) = &self.session {
            info!("Using ONNX session with provider: {:?}", session.execution_provider());
            
            let source_face_array = self.numpy_to_ndarray(py, source_face)?;
            let target_frame_array = self.numpy_to_ndarray(py, target_frame)?;
            
            let mut inputs = HashMap::new();
            inputs.insert("source_face".to_string(), source_face_array);
            inputs.insert("target_frame".to_string(), target_frame_array);
            
            match session.run(inputs) {
                Ok(outputs) => {
                    if let Some(output) = outputs.get("output") {
                        return self.ndarray_to_numpy(py, output);
                    } else {
                        warn!("No output found in inference result");
                    }
                },
                Err(e) => {
                    error!("Failed to run inference: {}", e);
                }
            }
        } else {
            warn!("No ONNX session available, returning unmodified frame");
        }
        
        Ok(target_frame.into_py(py))
    }
    
    fn numpy_to_ndarray(&self, py: Python, array: &PyAny) -> PyResult<ArrayD<f32>> {
        if !array.hasattr("shape")? || !array.hasattr("dtype")? {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input must be a numpy array"
            ));
        }
        
        let shape: Vec<usize> = array.getattr("shape")?.extract()?;
        
        let numpy = py.import("numpy")?;
        let flat_array = array.call_method1("astype", (numpy.getattr("float32")?,))?;
        let buffer = flat_array.call_method0("tobytes")?;
        let bytes: &[u8] = buffer.extract()?;
        
        let float_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                bytes.len() / std::mem::size_of::<f32>(),
            )
        };
        
        let array = ArrayD::from_shape_vec(
            IxDyn(&shape),
            float_slice.to_vec(),
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to create ndarray: {}", e)
        ))?;
        
        Ok(array)
    }
    
    fn ndarray_to_numpy(&self, py: Python, array: &ArrayD<f32>) -> PyResult<PyObject> {
        let numpy = py.import("numpy")?;
        
        let shape = array.shape();
        let data = array.as_slice().ok_or_else(|| 
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to get array data")
        )?;
        
        let py_shape = PyTuple::new(py, shape.iter().map(|&d| d as i64));
        
        let py_array = numpy.call_method1(
            "frombuffer",
            (PyBytes::new(py, unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<f32>(),
                )
            }),)
        )?;
        
        let py_array = py_array.call_method1("astype", (numpy.getattr("float32")?,))?;
        let py_array = py_array.call_method1("reshape", (py_shape,))?;
        
        Ok(py_array.into_py(py))
    }

    #[getter]
    fn name(&self) -> PyResult<String> {
        Ok("CAMSHOW.FACE-SWAPPER".to_string())
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
        Ok(format!("FaceSwapper(model_path='{}', device='{}')", self.model_path, self.device))
    }
}
