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
            
            let target_frame_array = self.preprocess_frame(py, target_frame)?;
            
            let input_names = session.input_names();
            if input_names.is_empty() {
                warn!("Model doesn't have any inputs");
                return Ok(target_frame.into_py(py));
            }
            
            let mut inputs = HashMap::new();
            inputs.insert(input_names[0].clone(), target_frame_array);
            
            match session.run(inputs) {
                Ok(outputs) => {
                    let output_names = session.output_names();
                    if output_names.is_empty() {
                        warn!("Model doesn't have any outputs");
                        return Ok(target_frame.into_py(py));
                    }
                    
                    if let Some(output) = outputs.get(&output_names[0]) {
                        return self.postprocess_output(py, output, target_frame);
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
    
    fn preprocess_frame(&self, py: Python, frame: &PyAny) -> PyResult<PyObject> {
        let numpy = py.import("numpy")?;
        let frame_array = if frame.is_instance(numpy.getattr("ndarray")?)? {
            frame.to_object(py)
        } else {
            numpy.call_method1("array", (frame,))?.to_object(py)
        };
        
        let frame_array = frame_array.as_ref(py);
        
        let frame_rgb = if frame_array.getattr("ndim")?.extract::<usize>()? == 3 {
            let channels = frame_array.getattr("shape")?.extract::<Vec<usize>>()?[2];
            if channels == 3 {
                frame_array.to_object(py)
            } else if channels == 4 {
                frame_array.call_method("[:,:,:3]", (), None)?.to_object(py)
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unsupported number of channels: {}", channels)
                ));
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Frame must be a 3D array (height, width, channels)"
            ));
        };
        
        let cv2 = py.import("cv2")?;
        let frame_resized = cv2.call_method(
            "resize", 
            (frame_rgb.as_ref(py), (512, 512)), 
            {
                let dict = PyDict::new(py);
                dict.set_item("interpolation", cv2.getattr("INTER_AREA")?)?;
                Some(dict)
            }
        )?;
        
        let frame_normalized = frame_resized.call_method1(py, "astype", (numpy.getattr("float32")?,))?
            .call_method1(py, "__truediv__", (255.0,))?;
        
        let frame_nchw = frame_normalized.call_method1(
            py,
            "transpose", 
            (PyTuple::new(py, &[2, 0, 1]),)
        )?;
        
        let frame_batched = frame_nchw.call_method1(
            py,
            "reshape", 
            (PyTuple::new(py, &[1, 3, 512, 512]),)
        )?;
        
        self.numpy_to_ndarray(py, frame_batched)
    }
    
    fn postprocess_output(&self, py: Python, output: &PyAny, original_frame: &PyAny) -> PyResult<PyObject> {
        let numpy = py.import("numpy")?;
        
        let output_numpy = self.ndarray_to_numpy(py, output)?;
        let output_array = output_numpy.as_ref(py);
        
        let original_shape = original_frame.getattr("shape")?.extract::<Vec<usize>>()?;
        
        let output_nhwc = output_array.call_method1(
            py,
            "transpose", 
            (PyTuple::new(py, &[0, 2, 3, 1]),)
        )?;
        
        let cv2 = py.import("cv2")?;
        let output_resized = cv2.call_method(
            "resize", 
            (output_nhwc, (original_shape[1], original_shape[0])), 
            {
                let dict = PyDict::new(py);
                dict.set_item("interpolation", cv2.getattr("INTER_LANCZOS4")?)?;
                Some(dict)
            }
        )?;
        
        let output_denormalized = output_resized.call_method1(py, "__mul__", (255.0,))?;
        
        let output_uint8 = output_denormalized.call_method1(
            py,
            "astype", 
            (numpy.getattr("uint8")?,)
        )?;
        
        Ok(output_uint8.into_py(py))
    }
    
    fn numpy_to_ndarray(&self, py: Python, array: &PyAny) -> PyResult<PyObject> {
        if !array.hasattr("shape")? || !array.hasattr("dtype")? {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input must be a numpy array"
            ));
        }
        
        let shape: Vec<usize> = array.getattr("shape")?.extract()?;
        
        let numpy = py.import("numpy")?;
        let flat_array = array.call_method1(py, "astype", (numpy.getattr("float32")?,))?;
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
    
    fn ndarray_to_numpy(&self, py: Python, array: ArrayD<f32>) -> PyResult<PyObject> {
        let numpy = py.import("numpy")?;
        
        return Ok(array.to_object(py));
        
        /*
        let shape = array.shape();
        let data = array.as_slice().ok_or_else(|| 
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to get array data")
        )?;
        
        let py_shape = PyTuple::new(py, shape.iter().map(|&d| d as i64));
        
        let py_array = numpy.call_method1(
            py,
            "frombuffer",
            (PyBytes::new(py, unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<f32>(),
                )
            }),)
        )?;
        */
        
        /*
        let py_array = py_array.call_method1(py, "astype", (numpy.getattr("float32")?,))?;
        let py_array = py_array.call_method1(py, "reshape", (py_shape,))?;
        */
        
        Ok(array.to_object(py))
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
