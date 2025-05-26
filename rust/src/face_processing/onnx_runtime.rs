use pyo3::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::{Result, anyhow};
use thiserror::Error;
use std::collections::HashMap;
use std::fs;
use log::{info, warn, error};

use crate::platform::{ExecutionProvider, get_optimal_provider};

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
}

pub struct OnnxSession {
    model_path: PathBuf,
    
    execution_provider: ExecutionProvider,
    
    #[allow(dead_code)]
    session: Option<()>, // Will be replaced with actual ONNX session type
    
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl OnnxSession {
    pub fn new(model_path: impl AsRef<Path>, execution_provider: Option<ExecutionProvider>) -> Result<Self, OnnxError> {
        let model_path = model_path.as_ref().to_path_buf();
        
        if !model_path.exists() {
            return Err(OnnxError::ModelNotFound(model_path.display().to_string()));
        }
        
        let execution_provider = execution_provider.unwrap_or_else(get_optimal_provider);
        
        
        Ok(Self {
            model_path,
            execution_provider,
            session: None,
            input_names: vec![],
            output_names: vec![],
        })
    }
    
    #[allow(dead_code)]
    pub fn run(&self, _inputs: HashMap<String, ndarray::Array4<f32>>) -> Result<HashMap<String, ndarray::Array4<f32>>, OnnxError> {
        Ok(HashMap::new())
    }
    
    pub fn execution_provider(&self) -> ExecutionProvider {
        self.execution_provider
    }
    
    pub fn model_path(&self) -> &Path {
        &self.model_path
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
        
        Err(OnnxError::DownloadError(format!("Download not implemented yet for model: {}", model_name)))
    }
    
    pub fn model_exists(&self, model_name: &str) -> bool {
        self.model_dir.join(model_name).exists()
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
            Some("cuda") => Some(ExecutionProvider::CUDA),
            Some("coreml") => Some(ExecutionProvider::CoreML),
            Some("cpu") | None => Some(ExecutionProvider::CPU),
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
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("ModelManager(model_dir='{}')", self.model_dir))
    }
}
