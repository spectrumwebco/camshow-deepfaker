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
) -> PyResult<bool> {
    let model_dir = model_dir.unwrap_or_else(|| "models".to_string());
    let model_manager = PyModelManager::new(Some(model_dir))?;
    
    let models = models.unwrap_or_else(|| vec![
        "inswapper_128.onnx".to_string(),
        "buffalo_l.onnx".to_string(),
        "gfpgan_1.4.onnx".to_string(),
    ]);
    
    for model in models {
        if !model_manager.model_exists(&model) {
            match model_manager.get_model_path(&model) {
                Ok(_) => println!("Downloaded model: {}", model),
                Err(e) => println!("Failed to download model {}: {}", model, e),
            }
        }
    }
    
    Ok(true)
}
