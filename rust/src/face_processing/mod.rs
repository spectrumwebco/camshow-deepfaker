use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod face_swapper;
mod face_enhancer;
mod face_analyser;
mod onnx_runtime;

use face_swapper::FaceSwapper;
use face_enhancer::FaceEnhancer;
use face_analyser::FaceAnalyser;
use onnx_runtime::{PyOnnxSession, PyModelManager};

#[pymodule]
pub fn face_processing_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FaceSwapper>()?;
    m.add_class::<FaceEnhancer>()?;
    m.add_class::<FaceAnalyser>()?;
    m.add_class::<PyOnnxSession>()?;
    m.add_class::<PyModelManager>()?;
    
    m.add_function(wrap_pyfunction!(swap_face, m)?)?;
    m.add_function(wrap_pyfunction!(enhance_face, m)?)?;
    m.add_function(wrap_pyfunction!(detect_faces, m)?)?;
    m.add_function(wrap_pyfunction!(download_models, m)?)?;
    
    Ok(())
}

#[pyfunction]
fn swap_face(
    py: Python,
    source_image: &PyAny,
    target_image: &PyAny,
    model_path: Option<String>,
) -> PyResult<PyObject> {
    let model_path = model_path.unwrap_or_else(|| "models/inswapper_128.onnx".to_string());
    
    let swapper = FaceSwapper::new(Some(model_path), None);
    
    swapper.process_frame(py, source_image, target_image)
}

#[pyfunction]
fn enhance_face(
    py: Python,
    image: &PyAny,
    model_path: Option<String>,
) -> PyResult<PyObject> {
    let model_path = model_path.unwrap_or_else(|| "models/gfpgan_1.4.onnx".to_string());
    
    let enhancer = FaceEnhancer::new(Some(model_path), None);
    
    enhancer.process_frame(py, image)
}

#[pyfunction]
fn detect_faces(
    py: Python,
    image: &PyAny,
    model_path: Option<String>,
) -> PyResult<PyObject> {
    let model_path = model_path.unwrap_or_else(|| "models/buffalo_l.onnx".to_string());
    
    let analyser = FaceAnalyser::new(Some(model_path), None);
    
    analyser.detect_faces(py, image)
}

#[pyfunction]
fn download_models(
    model_dir: Option<String>,
    models: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    let model_dir = model_dir.unwrap_or_else(|| "models".to_string());
    let model_manager = onnx_runtime::ModelManager::new(model_dir);
    
    if let Some(models) = models {
        let mut downloaded = Vec::new();
        
        for model in models {
            if model_manager.model_exists(&model) {
                println!("Model already exists: {}", model);
                let model_path = model_manager.get_model_path(&model)
                    .map_err(|e| {
                        println!("Error getting model path: {}", e);
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                let path = model_path.to_string_lossy().to_string();
                downloaded.push(path);
            } else {
                match model_manager.get_model_path(&model) {
                    Ok(path) => {
                        println!("Downloaded model: {}", model);
                        downloaded.push(path.to_string_lossy().to_string());
                    },
                    Err(e) => println!("Failed to download model {}: {}", model, e),
                }
            }
        }
        
        Ok(downloaded)
    } else {
        // Download all required models
        let required_models = vec![
            "inswapper_128.onnx".to_string(),
            "buffalo_l.onnx".to_string(),
            "gfpgan_1.4.onnx".to_string(),
        ];
        
        let mut downloaded = Vec::new();
        
        for model_name in required_models {
            match model_manager.get_model_path(&model_name) {
                Ok(path) => {
                    println!("Downloaded model: {}", model_name);
                    downloaded.push(path.to_string_lossy().to_string());
                },
                Err(e) => println!("Failed to download model {}: {}", model_name, e),
            }
        }
        
        Ok(downloaded)
    }
}
