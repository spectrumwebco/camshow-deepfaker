use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple, PyList};
use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;
use log::{info, warn, error};
use ndarray::{Array, ArrayD, Dimension, IxDyn};

use crate::platform::{PlatformExecutionProvider, get_optimal_provider};
use super::onnx_runtime::{OnnxSession, OnnxError, ModelManager};

#[pyclass]
pub struct FaceAnalyser {
    model_path: String,
    #[pyo3(get)]
    device: String,
    #[allow(dead_code)]
    session: Option<Arc<OnnxSession>>,
    #[allow(dead_code)]
    model_manager: Arc<ModelManager>,
}

#[pymethods]
impl FaceAnalyser {
    #[new]
    fn new(model_path: Option<String>, device: Option<String>) -> Self {
        let model_path = model_path.unwrap_or_else(|| "models/buffalo_l.onnx".to_string());
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
        
        FaceAnalyser {
            model_path,
            device,
            session,
            model_manager,
        }
    }

    fn detect_faces(&self, py: Python, frame: &PyAny) -> PyResult<PyObject> {
        if let Some(session) = &self.session {
            info!("Using ONNX session with provider: {:?}", session.execution_provider());
            
            let frame_array = self.preprocess_frame(py, frame)?;
            
            let input_names = session.input_names();
            if input_names.is_empty() {
                warn!("Model doesn't have any inputs");
                return Ok(PyList::empty(py).into_py(py));
            }
            
            let mut inputs = HashMap::new();
            inputs.insert(input_names[0].clone(), frame_array);
            
            match session.run(inputs) {
                Ok(outputs) => {
                    let output_names = session.output_names();
                    if output_names.is_empty() {
                        warn!("Model doesn't have any outputs");
                        return Ok(PyList::empty(py).into_py(py));
                    }
                    
                    if let Some(output) = outputs.get(&output_names[0]) {
                        return self.process_detection_output(py, output, frame);
                    } else {
                        warn!("No output found in inference result");
                    }
                },
                Err(e) => {
                    error!("Failed to run inference: {}", e);
                }
            }
        } else {
            warn!("No ONNX session available, returning empty result");
        }
        
        Ok(PyList::empty(py).into_py(py))
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
        
        let cv2 = py.import("cv2")?;
        let frame_resized = cv2.call_method(
            "resize", 
            (frame_rgb.as_ref(py), (640, 640)), 
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
            (PyTuple::new(py, &[1, 3, 640, 640]),)
        )?;
        
        self.numpy_to_ndarray(py, frame_batched)
    }
    
    fn process_detection_output(&self, py: Python, output: &PyAny, original_frame: &PyAny) -> PyResult<PyObject> {
        let numpy = py.import("numpy")?;
        
        let output_numpy = self.ndarray_to_numpy(py, output)?;
        let output_array = output_numpy.as_ref(py);
        
        let original_shape = original_frame.getattr("shape")?.extract::<Vec<usize>>()?;
        let orig_height = original_shape[0] as f32;
        let orig_width = original_shape[1] as f32;
        
        let scale_x = orig_width / 640.0;
        let scale_y = orig_height / 640.0;
        
        let faces = PyList::empty(py);
        
        let output_shape = output_array.getattr("shape")?.extract::<Vec<usize>>()?;
        if output_shape.len() < 3 {
            warn!("Unexpected output shape: {:?}", output_shape);
            return Ok(faces.into_py(py));
        }
        
        let num_detections = output_shape[1];
        let detection_size = output_shape[2];
        
        let confidence_threshold = 0.5;
        
        for i in 0..num_detections {
            let detection = output_array.call_method1(py, "__getitem__", (PyTuple::new(py, &[0, i]),))?;
            
            let confidence = detection.call_method1(py, "__getitem__", (4,))?.extract::<f32>()?;
            
            if confidence < confidence_threshold {
                continue;
            }
            
            let x1 = detection.call_method1(py, "__getitem__", (0,))?.extract::<f32>()? * scale_x;
            let y1 = detection.call_method1(py, "__getitem__", (1,))?.extract::<f32>()? * scale_y;
            let x2 = detection.call_method1(py, "__getitem__", (2,))?.extract::<f32>()? * scale_x;
            let y2 = detection.call_method1(py, "__getitem__", (3,))?.extract::<f32>()? * scale_y;
            
            let face_dict = PyDict::new(py);
            face_dict.set_item("confidence", confidence)?;
            face_dict.set_item("bbox", PyTuple::new(py, &[
                (x1 as i32).max(0),
                (y1 as i32).max(0),
                (x2 as i32).min(orig_width as i32),
                (y2 as i32).min(orig_height as i32)
            ]))?;
            
            if detection_size >= 15 {
                let landmarks = PyList::empty(py);
                
                for j in 0..5 {
                    let landmark_x = detection.call_method1(py, "__getitem__", (5 + j * 2,))?.extract::<f32>()? * scale_x;
                    let landmark_y = detection.call_method1(py, "__getitem__", (6 + j * 2,))?.extract::<f32>()? * scale_y;
                    
                    landmarks.append(PyTuple::new(py, &[landmark_x as i32, landmark_y as i32]))?;
                }
                
                face_dict.set_item("landmarks", landmarks)?;
            }
            
            let x1_int = (x1 as i32).max(0);
            let y1_int = (y1 as i32).max(0);
            let x2_int = (x2 as i32).min(orig_width as i32);
            let y2_int = (y2 as i32).min(orig_height as i32);
            
            let face_img = original_frame.call_method(
                "__getitem__", 
                (PyTuple::new(py, &[
                    format!("{}:{}", y1_int, y2_int),
                    format!("{}:{}", x1_int, x2_int)
                ]),), 
                None
            )?;
            
            face_dict.set_item("image", face_img)?;
            
            faces.append(face_dict)?;
        }
        
        Ok(faces.into_py(py))
    }

    fn get_landmarks(&self, py: Python, face: &PyAny) -> PyResult<PyObject> {
        if let Some(session) = &self.session {
            info!("Using ONNX session with provider: {:?}", session.execution_provider());
            
            if !face.hasattr("get")? {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Face must be a dictionary with bbox and image"
                ));
            }
            
            let face_image = match face.call_method1("get", ("image",)) {
                Ok(image) => {
                    if image.is_none() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Face dictionary must contain 'image' key"
                        ));
                    }
                    image
                },
                Err(_) => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Failed to get 'image' from face dictionary"
                    ));
                }
            };
            
            let face_array = self.preprocess_face_image(py, face_image)?;
            
            let input_names = session.input_names();
            if input_names.is_empty() {
                warn!("Model doesn't have any inputs");
                return Ok(PyList::empty(py).into_py(py));
            }
            
            let mut inputs = HashMap::new();
            inputs.insert(input_names[0].clone(), face_array);
            
            match session.run(inputs) {
                Ok(outputs) => {
                    let output_names = session.output_names();
                    if output_names.is_empty() {
                        warn!("Model doesn't have any outputs");
                        return Ok(PyList::empty(py).into_py(py));
                    }
                    
                    if let Some(output) = outputs.get(&output_names[0]) {
                        return self.process_landmark_output(py, output, face);
                    } else {
                        warn!("No output found in inference result");
                    }
                },
                Err(e) => {
                    error!("Failed to run inference: {}", e);
                }
            }
        } else {
            warn!("No ONNX session available, returning empty result");
        }
        
        Ok(PyList::empty(py).into_py(py))
    }
    
    fn preprocess_face_image(&self, py: Python, face_image: &PyAny) -> PyResult<ArrayD<f32>> {
        let numpy = py.import("numpy")?;
        let face_array = if face_image.is_instance(numpy.getattr("ndarray")?)? {
            face_image.to_object(py)
        } else {
            numpy.call_method1("array", (face_image,))?.to_object(py)
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
            (face_rgb.as_ref(py), (192, 192)), 
            {
                let dict = PyDict::new(py);
                dict.set_item("interpolation", cv2.getattr("INTER_AREA")?)?;
                Some(dict)
            }
        )?;
        
        let face_normalized = face_resized.call_method1(py, "astype", (numpy.getattr("float32")?,))?
            .call_method1(py, "__truediv__", (255.0,))?;
        
        let face_nchw = face_normalized.call_method1(
            py,
            "transpose", 
            (PyTuple::new(py, &[2, 0, 1]),)
        )?;
        
        let face_batched = face_nchw.call_method1(
            py,
            "reshape", 
            (PyTuple::new(py, &[1, 3, 192, 192]),)
        )?;
        
        self.numpy_to_ndarray(py, face_batched)
    }
    
    fn process_landmark_output(&self, py: Python, output: &PyAny, face: &PyAny) -> PyResult<PyObject> {
        let numpy = py.import("numpy")?;
        
        let output_numpy = self.ndarray_to_numpy(py, output)?;
        let output_array = output_numpy.as_ref(py);
        
        let bbox = match face.call_method1("get", ("bbox",)) {
            Ok(bbox) => {
                if bbox.is_none() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Face dictionary must contain 'bbox' key"
                    ));
                }
                bbox
            },
            Err(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Failed to get 'bbox' from face dictionary"
                ));
            }
        };
        
        let x1 = bbox.call_method1("__getitem__", (0,))?.extract::<i32>()?;
        let y1 = bbox.call_method1("__getitem__", (1,))?.extract::<i32>()?;
        let x2 = bbox.call_method1("__getitem__", (2,))?.extract::<i32>()?;
        let y2 = bbox.call_method1("__getitem__", (3,))?.extract::<i32>()?;
        
        let face_width = (x2 - x1) as f32;
        let face_height = (y2 - y1) as f32;
        
        
        let output_shape = output_array.getattr("shape")?.extract::<Vec<usize>>()?;
        if output_shape.len() < 3 {
            warn!("Unexpected output shape: {:?}", output_shape);
            return Ok(PyList::empty(py).into_py(py));
        }
        
        let num_landmarks = output_shape[1];
        
        let landmarks = PyList::empty(py);
        
        for i in 0..num_landmarks {
            let landmark = output_array.call_method1(py, "__getitem__", (PyTuple::new(py, &[0, i]),))?;
            let x_norm = landmark.call_method1(py, "__getitem__", (0,))?.extract::<f32>()?;
            let y_norm = landmark.call_method1(py, "__getitem__", (1,))?.extract::<f32>()?;
            
            let x = x1 as f32 + x_norm * face_width;
            let y = y1 as f32 + y_norm * face_height;
            
            landmarks.append(PyTuple::new(py, &[x as i32, y as i32]))?;
        }
        
        Ok(landmarks.into_py(py))
    }
    
    fn numpy_to_ndarray(&self, py: Python, array: &PyAny) -> PyResult<PyObject> {
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
        
        let py_array = py_array.call_method1(py, "astype", (numpy.getattr("float32")?,))?;
        let py_array = py_array.call_method1(py, "reshape", (py_shape,))?;
        
        Ok(py_array.into_py(py))
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
        Ok(format!("FaceAnalyser(model_path='{}', device='{}')", self.model_path, self.device))
    }
}
