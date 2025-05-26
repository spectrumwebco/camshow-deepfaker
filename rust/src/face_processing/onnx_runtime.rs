use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyDict, PyList};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::{Result, anyhow};
use thiserror::Error;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use log::{info, warn, error};
use ndarray::{Array, ArrayD, IxDyn, Ix1, Ix2, Ix3, Ix4, Dimension};
use num_cpus;

const SIZE_MAX: u64 = u64::MAX;

impl From<PyErr> for OnnxError {
    fn from(err: PyErr) -> Self {
        OnnxError::InvalidInput(format!("Python error: {}", err))
    }
}

#[cfg(feature = "onnxruntime")]
use ort::{
    Session, SessionBuilder, GraphOptimizationLevel, Environment, 
    LoggingLevel, OrtError, 
    CUDAExecutionProviderOptions, CoreMLExecutionProviderOptions,
    Value, ValueType, TensorElementDataType
};

use crate::platform::{PlatformExecutionProvider, get_optimal_provider};

#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),
    
    #[error("Failed to run inference: {0}")]
    InferenceError(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Model not found at path: {0}")]
    ModelNotFound(String),
    
    #[error("Unsupported execution provider: {0}")]
    UnsupportedProvider(String),
    
    #[error("Failed to download model: {0}")]
    DownloadError(String),
    
    #[cfg(feature = "onnxruntime")]
    #[error("ONNX Runtime error: {0}")]
    OrtError(#[from] OrtError),
}

pub struct OnnxSession {
    model_path: PathBuf,
    execution_provider: PlatformExecutionProvider,
    
    #[cfg(feature = "onnxruntime")]
    session: Option<Session>,
    
    #[cfg(not(feature = "onnxruntime"))]
    session: Option<()>,
    
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl OnnxSession {
    pub fn new(model_path: impl AsRef<Path>, execution_provider: Option<PlatformExecutionProvider>) -> Result<Self, OnnxError> {
        let model_path = model_path.as_ref().to_path_buf();
        
        if !model_path.exists() {
            return Err(OnnxError::ModelNotFound(model_path.display().to_string()));
        }
        
        let execution_provider = execution_provider.unwrap_or_else(get_optimal_provider);
        
        #[cfg(feature = "onnxruntime")]
        let (session, input_names, output_names) = {
            let environment = Arc::new(Environment::builder()
                .with_name("camshow_deepfaker")
                .with_log_level(LoggingLevel::Warning)
                .build()?);
            
            let mut session_builder = SessionBuilder::new(&environment)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(num_cpus::get() as i16)?;
            
            match execution_provider {
                PlatformExecutionProvider::CUDA => {
                    #[cfg(feature = "cuda")]
                    {
                        info!("Using CUDA execution provider");
                        let cuda_options = CUDAExecutionProviderOptions::default();
                        session_builder = session_builder.with_cuda_ep(cuda_options)?;
                    }
                    
                    #[cfg(not(feature = "cuda"))]
                    {
                        warn!("CUDA requested but not available, falling back to CPU");
                    }
                },
                PlatformExecutionProvider::CoreML => {
                    #[cfg(feature = "coreml")]
                    {
                        info!("Using CoreML execution provider");
                        let coreml_options = CoreMLExecutionProviderOptions::default();
                        session_builder = session_builder.with_coreml_ep(coreml_options)?;
                    }
                    
                    #[cfg(not(feature = "coreml"))]
                    {
                        warn!("CoreML requested but not available, falling back to CPU");
                    }
                },
                PlatformExecutionProvider::CPU => {
                    info!("Using CPU execution provider");
                }
            }
            
            let session = session_builder.with_model_from_file(&model_path)?;
            
            let input_names = session.inputs
                .iter()
                .map(|input| input.name.clone())
                .collect::<Vec<String>>();
            
            let output_names = session.outputs
                .iter()
                .map(|output| output.name.clone())
                .collect::<Vec<String>>();
            
            (Some(session), input_names, output_names)
        };
        
        #[cfg(not(feature = "onnxruntime"))]
        let (session, input_names, output_names) = {
            warn!("ONNX Runtime feature not enabled, using placeholder session");
            (None, vec![], vec![])
        };
        
        Ok(Self {
            model_path,
            execution_provider,
            session,
            input_names,
            output_names,
        })
    }
    
    pub fn run(&self, inputs: HashMap<String, PyObject>) -> Result<HashMap<String, PyObject>, OnnxError> {
        #[cfg(feature = "onnxruntime")]
        {
            if let Some(session) = &self.session {
                Python::with_gil(|py| {
                    let mut input_tensors = Vec::new();
                    
                    for name in &self.input_names {
                        if let Some(array) = inputs.get(name) {
                            let numpy = py.import("numpy")?;
                            let array_ref = array.as_ref(py);
                            
                            let array_c = if array_ref.call_method0("flags")?.getattr("c_contiguous")?.extract::<bool>()? {
                                array_ref.to_object(py)
                            } else {
                                array_ref.call_method0("copy")?.to_object(py)
                            };
                            
                            let shape = array_ref.getattr("shape")?.extract::<Vec<i64>>()?;
                            let dtype = array_ref.getattr("dtype")?.getattr("name")?.extract::<String>()?;
                            
                            let data = array_c.as_ref(py).call_method0("tobytes")?;
                            let bytes = data.extract::<&[u8]>()?;
                            
                            let tensor = match dtype.as_str() {
                                "float32" => {
                                    let data = unsafe {
                                        std::slice::from_raw_parts(
                                            bytes.as_ptr() as *const f32,
                                            bytes.len() / std::mem::size_of::<f32>()
                                        )
                                    };
                                    
                                    let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                                    
                                    let array = ndarray::Array::from_shape_vec(
                                        ndarray::IxDyn(&shape_usize), 
                                        data.to_vec()
                                    ).map_err(|e| OnnxError::InvalidInput(format!("Failed to create ndarray: {}", e)))?;
                                    
                                    let cow_array = array.into_dyn();
                                    Value::from_array(session.allocator(), &cow_array)?
                                },
                                "float64" => {
                                    let data = unsafe {
                                        std::slice::from_raw_parts(
                                            bytes.as_ptr() as *const f64,
                                            bytes.len() / std::mem::size_of::<f64>()
                                        )
                                    };
                                    
                                    let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                                    
                                    let array = ndarray::Array::from_shape_vec(
                                        ndarray::IxDyn(&shape_usize), 
                                        data.to_vec()
                                    ).map_err(|e| OnnxError::InvalidInput(format!("Failed to create ndarray: {}", e)))?;
                                    
                                    let cow_array = array.into_dyn();
                                    Value::from_array(session.allocator(), &cow_array)?
                                },
                                "int32" => {
                                    let data = unsafe {
                                        std::slice::from_raw_parts(
                                            bytes.as_ptr() as *const i32,
                                            bytes.len() / std::mem::size_of::<i32>()
                                        )
                                    };
                                    
                                    let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                                    
                                    let array = ndarray::Array::from_shape_vec(
                                        ndarray::IxDyn(&shape_usize), 
                                        data.to_vec()
                                    ).map_err(|e| OnnxError::InvalidInput(format!("Failed to create ndarray: {}", e)))?;
                                    
                                    let cow_array = array.into_dyn();
                                    Value::from_array(session.allocator(), &cow_array)?
                                },
                                "int64" => {
                                    let data = unsafe {
                                        std::slice::from_raw_parts(
                                            bytes.as_ptr() as *const i64,
                                            bytes.len() / std::mem::size_of::<i64>()
                                        )
                                    };
                                    
                                    let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                                    
                                    let array = ndarray::Array::from_shape_vec(
                                        ndarray::IxDyn(&shape_usize), 
                                        data.to_vec()
                                    ).map_err(|e| OnnxError::InvalidInput(format!("Failed to create ndarray: {}", e)))?;
                                    
                                    let cow_array = array.into_dyn();
                                    Value::from_array(session.allocator(), &cow_array)?
                                },
                                "uint8" => {
                                    let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                                    
                                    let array = ndarray::Array::from_shape_vec(
                                        ndarray::IxDyn(&shape_usize), 
                                        bytes.to_vec()
                                    ).map_err(|e| OnnxError::InvalidInput(format!("Failed to create ndarray: {}", e)))?;
                                    
                                    let cow_array = array.into_dyn();
                                    Value::from_array(session.allocator(), &cow_array)?
                                },
                                _ => return Err(OnnxError::InvalidInput(format!("Unsupported data type: {}", dtype))),
                            };
                            
                            input_tensors.push(tensor);
                        } else {
                            return Err(OnnxError::InvalidInput(format!("Missing input: {}", name)));
                        }
                    }
                    
                    let outputs = session.run(input_tensors)?;
                    
                    let mut result = HashMap::new();
                    
                    for (i, name) in self.output_names.iter().enumerate() {
                        if i < outputs.len() {
                            let output = &outputs[i];
                            
                            let tensor_type = if output.is_tensor()? {
                                match output.tensor_element_type()? {
                                    TensorElementDataType::Float => ValueType::Float32,
                                    TensorElementDataType::Double => ValueType::Float64,
                                    TensorElementDataType::Int32 => ValueType::Int32,
                                    TensorElementDataType::Int64 => ValueType::Int64,
                                    TensorElementDataType::Uint8 => ValueType::Uint8,
                                    _ => {
                                        warn!("Unsupported tensor element type: {:?}", output.tensor_element_type()?);
                                        ValueType::Float32 // Default to float32
                                    }
                                }
                            } else {
                                warn!("Output is not a tensor");
                                ValueType::Float32 // Default to float32
                            };
                            
                            let shape: Vec<i64> = output.dimensions()?
                                .iter()
                                .map(|&d| d.unwrap_or(1))
                                .collect();
                            
                            let numpy_array = match tensor_type {
                                ValueType::Float32 => {
                                    let tensor_data = output.try_extract::<f32>()?;
                                    let data_vec = tensor_data.view().as_slice().unwrap_or(&[]).to_vec();
                                    
                                    let numpy = py.import("numpy")?;
                                    let array = numpy.call_method1("array", (data_vec,))?;
                                    array.call_method1("reshape", (shape,))?
                                },
                                ValueType::Float64 => {
                                    let tensor_data = output.try_extract::<f64>()?;
                                    let data_vec = tensor_data.view().as_slice().unwrap_or(&[]).to_vec();
                                    
                                    let numpy = py.import("numpy")?;
                                    let array = numpy.call_method1("array", (data_vec,))?;
                                    array.call_method1("reshape", (shape,))?
                                },
                                ValueType::Int32 => {
                                    let tensor_data = output.try_extract::<i32>()?;
                                    let data_vec = tensor_data.view().as_slice().unwrap_or(&[]).to_vec();
                                    
                                    let numpy = py.import("numpy")?;
                                    let array = numpy.call_method1("array", (data_vec,))?;
                                    array.call_method1("reshape", (shape,))?
                                },
                                ValueType::Int64 => {
                                    let tensor_data = output.try_extract::<i64>()?;
                                    let data_vec = tensor_data.view().as_slice().unwrap_or(&[]).to_vec();
                                    
                                    let numpy = py.import("numpy")?;
                                    let array = numpy.call_method1("array", (data_vec,))?;
                                    array.call_method1("reshape", (shape,))?
                                },
                                ValueType::Uint8 => {
                                    let tensor_data = output.try_extract::<u8>()?;
                                    let data_vec = tensor_data.view().as_slice().unwrap_or(&[]).to_vec();
                                    
                                    let numpy = py.import("numpy")?;
                                    let array = numpy.call_method1("array", (data_vec,))?;
                                    array.call_method1("reshape", (shape,))?
                                },
                                _ => {
                                    warn!("Unsupported tensor type: {:?}", tensor_type);
                                    py.import("numpy")?.call_method0("array")?
                                }
                            };
                            
                            result.insert(name.clone(), numpy_array.to_object(py));
                        } else {
                            warn!("Output '{}' not found in inference results (index {})", name, i);
                        }
                    }
                    
                    Ok(result)
                })
            } else {
                Err(OnnxError::InferenceError("No session available".to_string()))
            }
        }
        
        #[cfg(not(feature = "onnxruntime"))]
        {
            warn!("ONNX Runtime feature not enabled, returning empty result");
            Python::with_gil(|py| {
                let mut result = HashMap::new();
                for name in &self.output_names {
                    let numpy = py.import("numpy")?;
                    let empty_array = numpy.call_method0("array")?;
                    result.insert(name.clone(), empty_array.to_object(py));
                }
                
                Ok(result)
            })
        }
    }
    
    pub fn execution_provider(&self) -> PlatformExecutionProvider {
        self.execution_provider
    }
    
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
    
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }
    
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
}

pub struct ModelManager {
    model_dir: PathBuf,
    model_urls: HashMap<String, String>,
}

impl ModelManager {
    pub fn new(model_dir: impl AsRef<Path>) -> Self {
        let model_dir = model_dir.as_ref().to_path_buf();
        
        if !model_dir.exists() {
            if let Err(e) = fs::create_dir_all(&model_dir) {
                warn!("Failed to create model directory: {}", e);
            }
        }
        
        let mut model_urls = HashMap::new();
        model_urls.insert(
            "inswapper_128.onnx".to_string(),
            "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx".to_string(),
        );
        model_urls.insert(
            "inswapper_256.onnx".to_string(),
            "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_256.onnx".to_string(),
        );
        model_urls.insert(
            "buffalo_l.onnx".to_string(),
            "https://huggingface.co/hacksider/deep-live-cam/resolve/main/buffalo_l.onnx".to_string(),
        );
        model_urls.insert(
            "gfpgan_1.4.onnx".to_string(),
            "https://huggingface.co/hacksider/deep-live-cam/resolve/main/gfpgan_1.4.onnx".to_string(),
        );
        
        Self {
            model_dir,
            model_urls,
        }
    }
    
    pub fn get_model_path(&self, model_name: &str) -> Result<PathBuf, OnnxError> {
        let model_path = self.model_dir.join(model_name);
        
        if model_path.exists() {
            return Ok(model_path);
        }
        
        info!("Model {} not found, attempting to download", model_name);
        
        let url = self.model_urls.get(model_name)
            .ok_or_else(|| OnnxError::ModelNotFound(format!("No URL found for model: {}", model_name)))?;
        
        self.download_model(url, &model_path)
            .map_err(|e| OnnxError::DownloadError(format!("Failed to download model {}: {}", model_name, e)))?;
        
        Ok(model_path)
    }
    
    pub fn download_model(&self, url: &str, output_path: &Path) -> Result<(), anyhow::Error> {
        info!("Downloading model from {} to {}", url, output_path.display());
        
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        let client = reqwest::blocking::Client::builder()
            .user_agent("camshow-deepfaker-rust/0.1.0")
            .build()?;
        
        let response = client.get(url).send()?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Failed to download model: HTTP status {}", response.status()));
        }
        
        let content_length = response.content_length().unwrap_or(0);
        info!("Downloading {} bytes", content_length);
        
        let mut file = fs::File::create(output_path)?;
        let bytes = response.bytes()?;
        file.write_all(&bytes)?;
        
        info!("Downloaded {} bytes to {}", bytes.len(), output_path.display());
        
        Ok(())
    }
    
    pub fn model_exists(&self, model_name: &str) -> bool {
        self.model_dir.join(model_name).exists()
    }
    
    pub fn download_all_models(&self) -> Result<Vec<String>, PyErr> {
        let required_models = vec![
            "inswapper_128.onnx",
            "buffalo_l.onnx",
            "gfpgan_1.4.onnx",
        ];
        
        let mut downloaded_paths = Vec::new();
        
        for model_name in required_models {
            match self.get_model_path(model_name) {
                Ok(path) => {
                    info!("Successfully downloaded or found model: {}", model_name);
                    downloaded_paths.push(path.to_string_lossy().to_string());
                },
                Err(e) => {
                    error!("Failed to download model {}: {}", model_name, e);
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        format!("Failed to download model {}: {}", model_name, e)
                    ));
                }
            }
        }
        
        Ok(downloaded_paths)
    }
}

#[pyclass]
pub struct PyOnnxSession {
    #[pyo3(get)]
    model_path: String,
    
    #[pyo3(get)]
    provider: String,
    
    #[allow(dead_code)]
    session: Arc<OnnxSession>,
}

#[pymethods]
impl PyOnnxSession {
    #[new]
    fn new(model_path: String, provider: Option<String>) -> PyResult<Self> {
        let execution_provider = match provider.as_deref() {
            Some("cuda") => Some(PlatformExecutionProvider::CUDA),
            Some("coreml") => Some(PlatformExecutionProvider::CoreML),
            Some("cpu") | None => Some(PlatformExecutionProvider::CPU),
            Some(other) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported execution provider: {}", other)
            )),
        };
        
        let session = OnnxSession::new(&model_path, execution_provider)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self {
            model_path,
            provider: format!("{:?}", session.execution_provider()).to_lowercase(),
            session: Arc::new(session),
        })
    }
    
    fn run(&self, py: Python, inputs: &PyAny) -> PyResult<PyObject> {
        let inputs_dict = inputs.extract::<pyo3::types::PyDict>()?;
        let mut input_map = HashMap::new();
        
        for (key, value) in inputs_dict.iter() {
            let key_str = key.extract::<String>()?;
            input_map.insert(key_str, value.to_object(py));
        }
        
        match self.session.run(input_map) {
            Ok(outputs) => {
                let output_dict = pyo3::types::PyDict::new(py);
                for (key, value) in outputs {
                    output_dict.set_item(key, value.as_ref(py))?;
                }
                Ok(output_dict.into())
            },
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
        }
    }
    
    fn input_names(&self) -> PyResult<Vec<String>> {
        Ok(self.session.input_names().to_vec())
    }
    
    fn output_names(&self) -> PyResult<Vec<String>> {
        Ok(self.session.output_names().to_vec())
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("OnnxSession(model_path='{}', provider='{}')", self.model_path, self.provider))
    }
}

#[pyclass]
pub struct PyModelManager {
    #[pyo3(get)]
    model_dir: String,
    
    #[allow(dead_code)]
    manager: Arc<ModelManager>,
}

#[pymethods]
impl PyModelManager {
    #[new]
    fn new(model_dir: Option<String>) -> PyResult<Self> {
        let model_dir = model_dir.unwrap_or_else(|| "models".to_string());
        let manager = ModelManager::new(&model_dir);
        
        Ok(Self {
            model_dir: model_dir.clone(),
            manager: Arc::new(manager),
        })
    }
    
    fn get_model_path(&self, model_name: &str) -> PyResult<String> {
        self.manager.get_model_path(model_name)
            .map(|p| p.to_string_lossy().to_string())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    fn model_exists(&self, model_name: &str) -> bool {
        self.manager.model_exists(model_name)
    }
    
    fn download_all_models(&self) -> PyResult<Vec<String>> {
        self.manager.download_all_models()
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("ModelManager(model_dir='{}')", self.model_dir))
    }
}
