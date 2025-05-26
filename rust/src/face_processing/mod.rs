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
pub fn face_processing_module(py: Python, m: &PyModule) -> PyResult<()> {
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
    Ok(target_image.into_py(py))
}

#[pyfunction]
fn enhance_face(
    py: Python,
    image: &PyAny,
    model_path: Option<String>,
) -> PyResult<PyObject> {
    Ok(image.into_py(py))
}

#[pyfunction]
fn detect_faces(
    py: Python,
    image: &PyAny,
) -> PyResult<PyObject> {
    Ok(vec![].into_py(py))
}

#[pyfunction]
fn download_models(
    model_dir: Option<String>,
    models: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    let model_dir = model_dir.unwrap_or_else(|| "models".to_string());
    let model_manager = PyModelManager::new(Some(model_dir))?;
    
    if models.is_some() {
        let models = models.unwrap();
        let mut downloaded = Vec::new();
        
        for model in models {
            if !model_manager.model_exists(&model) {
                match model_manager.get_model_path(&model) {
                    Ok(path) => {
                        println!("Downloaded model: {}", model);
                        downloaded.push(path);
                    },
                    Err(e) => println!("Failed to download model {}: {}", model, e),
                }
            } else {
                println!("Model already exists: {}", model);
                downloaded.push(model_manager.model_dir.clone() + "/" + &model);
            }
        }
        
        Ok(downloaded)
    } else {
        match model_manager.download_all_models() {
            Ok(paths) => {
                println!("Downloaded all required models");
                Ok(paths)
            },
            Err(e) => {
                println!("Failed to download all models: {}", e);
                Err(e)
            }
        }
    }
}
