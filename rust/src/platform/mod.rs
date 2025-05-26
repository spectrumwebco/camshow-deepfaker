
use pyo3::prelude::*;
use std::env;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformExecutionProvider {
    CPU,
    CUDA,
    CoreML,
}

#[pyclass]
pub struct PlatformInfo {
    #[pyo3(get)]
    pub os_name: String,
    #[pyo3(get)]
    pub is_linux: bool,
    #[pyo3(get)]
    pub is_macos: bool,
    #[pyo3(get)]
    pub is_apple_silicon: bool,
    #[pyo3(get)]
    pub has_cuda: bool,
    #[pyo3(get)]
    pub has_coreml: bool,
    #[pyo3(get)]
    pub recommended_provider: String,
}

#[pymethods]
impl PlatformInfo {
    #[new]
    fn new() -> Self {
        let os_name = env::consts::OS.to_string();
        let is_linux = os_name == "linux";
        let is_macos = os_name == "macos";
        
        let is_apple_silicon = is_macos && detect_apple_silicon();
        
        let has_cuda = is_linux && detect_cuda();
        
        let has_coreml = is_macos && detect_coreml();
        
        let recommended_provider = if has_cuda {
            "CUDA".to_string()
        } else if has_coreml {
            "CoreML".to_string()
        } else {
            "CPU".to_string()
        };
        
        PlatformInfo {
            os_name,
            is_linux,
            is_macos,
            is_apple_silicon,
            has_cuda,
            has_coreml,
            recommended_provider,
        }
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "PlatformInfo(os='{}', recommended_provider='{}')",
            self.os_name, self.recommended_provider
        ))
    }
}

pub fn detect_cuda() -> bool {
    #[cfg(feature = "cuda")]
    {
        if let Ok(output) = std::process::Command::new("nvidia-smi").output() {
            return output.status.success();
        }
    }
    
    false
}

pub fn detect_apple_silicon() -> bool {
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("uname").arg("-m").output() {
            if let Ok(arch) = String::from_utf8(output.stdout) {
                return arch.trim() == "arm64";
            }
        }
    }
    
    false
}

pub fn detect_coreml() -> bool {
    #[cfg(feature = "coreml")]
    {
        if detect_apple_silicon() {
            return true;
        }
    }
    
    false
}

pub fn get_optimal_provider() -> PlatformExecutionProvider {
    if detect_cuda() {
        PlatformExecutionProvider::CUDA
    } else if detect_coreml() {
        PlatformExecutionProvider::CoreML
    } else {
        PlatformExecutionProvider::CPU
    }
}

#[pymodule]
pub fn platform_module(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PlatformInfo>()?;
    
    m.add_function(wrap_pyfunction!(py_detect_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_apple_silicon, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_coreml, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_optimal_provider, m)?)?;
    
    Ok(())
}

#[pyfunction]
fn py_detect_cuda() -> bool {
    detect_cuda()
}

#[pyfunction]
fn py_detect_apple_silicon() -> bool {
    detect_apple_silicon()
}

#[pyfunction]
fn py_detect_coreml() -> bool {
    detect_coreml()
}

#[pyfunction]
fn py_get_optimal_provider() -> String {
    match get_optimal_provider() {
        PlatformExecutionProvider::CUDA => "CUDA".to_string(),
        PlatformExecutionProvider::CoreML => "CoreML".to_string(),
        PlatformExecutionProvider::CPU => "CPU".to_string(),
    }
}
