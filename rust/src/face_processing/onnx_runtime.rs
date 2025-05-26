use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyBytes};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::{Result, anyhow};
use thiserror::Error;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use log::{info, warn, error};
use ndarray::{ArrayD, IxDyn};
use num_cpus;

const SIZE_MAX: u64 = u64::MAX;

#[cfg(feature = "onnxruntime")]
use ort::{
    Session, SessionBuilder, GraphOptimizationLevel, Environment, 
    LoggingLevel, OrtError, 
    execution_providers::{CUDAExecutionProviderOptions, CoreMLExecutionProviderOptions},
    tensor::OrtOwnedTensor
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
                        let cuda_options = CUDAExecutionProviderOptions::default()
                            .device_id(0)
                            .gpu_mem_limit(SIZE_MAX as usize);
                        session_builder = session_builder.with_execution_providers([cuda_options.into()])?;
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
                        session_builder = session_builder.with_execution_providers([coreml_options.into()])?;
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
    
    pub fn run(&self, inputs: HashMap<String, ArrayD<f32>>) -> Result<HashMap<String, ArrayD<f32>>, OnnxError> {
        #[cfg(feature = "onnxruntime")]
        {
            if let Some(session) = &self.session {
                let mut input_values = Vec::new();
                let mut input_names = Vec::new();
                
                for (name, array) in &inputs {
                    if !self.input_names.contains(name) {
                        return Err(OnnxError::InvalidInput(format!("Unknown input name: {}", name)));
                    }
                    
                    let array_view = array.view();
                    let tensor_values = array.as_slice().unwrap_or(&[]).to_vec();
                    let tensor_shape = array.shape().iter().map(|&d| d as i64).collect::<Vec<i64>>();
                    
                    let value = ort::Value::tensor_from_data(
                        &tensor_shape,
                        tensor_values,
                    )?;
                    
                    input_values.push(value);
                    input_names.push(name.clone());
                }
                
                let outputs = session.run(input_values)?;
                
                let mut result = HashMap::new();
                
                for (i, name) in self.output_names.iter().enumerate() {
                    if i < outputs.len() {
                        let tensor: OrtOwnedTensor<f32, _> = outputs[i].try_extract().map_err(|e| {
                            OnnxError::InferenceError(format!("Failed to extract tensor data for {}: {}", name, e))
                        })?;
                        
                        let dims: Vec<usize> = tensor.shape().iter()
                            .map(|&d| d as usize)
                            .collect();
                        
                        let data = tensor.view().as_slice().unwrap_or(&[]).to_vec();
                        
                        let array = ArrayD::from_shape_vec(IxDyn(&dims), data)
                            .map_err(|e| OnnxError::InferenceError(
                                format!("Failed to convert output to ndarray: {}", e)
                            ))?;
                        
                        result.insert(name.clone(), array);
                    } else {
                        warn!("Output '{}' not found in inference results (index {})", name, i);
                    }
                }
                
                Ok(result)
            } else {
                Err(OnnxError::InferenceError("No session available".to_string()))
            }
        }
        
        #[cfg(not(feature = "onnxruntime"))]
        {
            Err(OnnxError::InferenceError("ONNX Runtime feature not enabled".to_string()))
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
