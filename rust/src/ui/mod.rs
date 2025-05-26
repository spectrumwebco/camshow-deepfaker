use pyo3::prelude::*;

mod app;

pub use app::App;

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
        println!("Launching Dioxus UI...");
        
        
        Ok(())
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok("AppWrapper()".to_string())
    }
}
