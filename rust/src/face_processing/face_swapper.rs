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
            
            let source_face_array = self.preprocess_face(py, source_face)?;
            
            let target_frame_array = self.preprocess_frame(py, target_frame)?;
            
            let input_names = session.input_names();
            if input_names.len() < 2 {
                warn!("Model doesn't have enough inputs, expected at least 2, got {}", input_names.len());
                return Ok(target_frame.into_py(py));
            }
            
            let mut inputs = HashMap::new();
            inputs.insert(input_names[0].clone(), source_face_array);
            inputs.insert(input_names[1].clone(), target_frame_array);
            
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
    
    fn preprocess_face(&self, py: Python, face: &PyAny) -> PyResult<ArrayD<f32>> {
        let numpy = py.import("numpy")?;
        let face_array = if face.is_instance(numpy.getattr("ndarray")?)? {
            face.to_object(py)
        } else {
            numpy.call_method1("array", (face,))?.to_object(py)
        };
        
        let face_array = face_array.as_ref(py);
        
        let face_rgb = if face_array.getattr("ndim")?.extract::<usize>()? == 3 {
            let channels = face_array.getattr("shape")?.extract::<Vec<usize>>()?[2];
            if channels == 3 {
                face_array.to_object(py)
            } else if channels == 4 {
                face_array.call_method("[:,:,:3]", (), None)?.to_object(py)
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unsupported number of channels: {}", channels)
                ));
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Face must be a 3D array (height, width, channels)"
            ));
        };
        
        let cv2 = py.import("cv2")?;
        let face_resized = cv2.call_method(
            "resize", 
            (face_rgb.as_ref(py), (128, 128)), 
            Some(PyDict::new(py).set_item("interpolation", cv2.getattr("INTER_AREA")?)?)
        )?;
        
        let face_normalized = face_resized.call_method1("astype", (numpy.getattr("float32")?,))?
            .call_method1("__truediv__", (255.0,))?;
        
        let face_nchw = face_normalized.call_method1(
            "transpose", 
            (PyTuple::new(py, &[2, 0, 1]),)
        )?;
        
        let face_batched = face_nchw.call_method1(
            "reshape", 
            (PyTuple::new(py, &[1, 3, 128, 128]),)
        )?;
        
        self.numpy_to_ndarray(py, face_batched)
    }
    
    fn preprocess_frame(&self, py: Python, frame: &PyAny) -> PyResult<ArrayD<f32>> {
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
        
        let frame_normalized = frame_rgb.call_method1("astype", (numpy.getattr("float32")?,))?
            .call_method1("__truediv__", (255.0,))?;
        
        let frame_nchw = frame_normalized.call_method1(
            "transpose", 
            (PyTuple::new(py, &[2, 0, 1]),)
        )?;
        
        let shape = frame_array.getattr("shape")?.extract::<Vec<usize>>()?;
        let frame_batched = frame_nchw.call_method1(
            "reshape", 
            (PyTuple::new(py, &[1, 3, shape[0], shape[1]]),)
        )?;
        
        self.numpy_to_ndarray(py, frame_batched)
    }
    
    fn postprocess_output(&self, py: Python, output: &ArrayD<f32>, original_frame: &PyAny) -> PyResult<PyObject> {
        let numpy = py.import("numpy")?;
        
        let output_numpy = self.ndarray_to_numpy(py, output)?;
        let output_array = output_numpy.as_ref(py);
        
        let original_shape = original_frame.getattr("shape")?.extract::<Vec<usize>>()?;
        
        let output_nhwc = output_array.call_method1(
            "transpose", 
            (PyTuple::new(py, &[0, 2, 3, 1]),)
        )?;
        
        let output_reshaped = output_nhwc.call_method1(
            "reshape", 
            (PyTuple::new(py, &[original_shape[0], original_shape[1], 3]),)
        )?;
        
        let output_denormalized = output_reshaped.call_method1("__mul__", (255.0,))?;
        
        let output_uint8 = output_denormalized.call_method1(
            "astype", 
            (numpy.getattr("uint8")?,)
        )?;
        
        Ok(output_uint8.into_py(py))
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
