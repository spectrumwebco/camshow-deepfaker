use pyo3::prelude::*;


#[pymodule]
pub fn ui_module(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AppWrapper>()?;
    
    Ok(())
}

#[pyclass]
pub struct AppWrapper {}

#[pymethods]
impl AppWrapper {
    #[new]
    fn new() -> Self {
        AppWrapper {}
    }
    
    fn launch(&self) -> PyResult<()> {
        println!("UI module temporarily disabled to focus on ONNX integration");
        println!("Will be re-enabled when Dioxus dependencies are uncommented in Cargo.toml");
        
        Ok(())
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok("AppWrapper(disabled=True)".to_string())
    }
}
