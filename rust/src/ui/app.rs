use pyo3::prelude::*;


pub struct App {
    theme: String,
}

impl App {
    pub fn new() -> Self {
        App {
            theme: "dark".to_string(),
        }
    }
    
    pub fn toggle_theme(&mut self) {
        self.theme = if self.theme == "dark" {
            "light".to_string()
        } else {
            "dark".to_string()
        };
    }
    
    pub fn get_theme(&self) -> &str {
        &self.theme
    }
}

pub struct CameraPreview {}

impl CameraPreview {
    pub fn new() -> Self {
        CameraPreview {}
    }
}

pub struct ControlPanel {
    selected_model: String,
    enhance_faces: bool,
}

impl ControlPanel {
    pub fn new() -> Self {
        ControlPanel {
            selected_model: "inswapper_128.onnx".to_string(),
            enhance_faces: true,
        }
    }
    
    pub fn set_model(&mut self, model: &str) {
        self.selected_model = model.to_string();
    }
    
    pub fn toggle_enhancement(&mut self) {
        self.enhance_faces = !self.enhance_faces;
    }
    
    pub fn get_model(&self) -> &str {
        &self.selected_model
    }
    
    pub fn is_enhancement_enabled(&self) -> bool {
        self.enhance_faces
    }
}

pub fn encode_base64(data: &[u8]) -> String {
    use base64::{Engine as _, engine::general_purpose};
    general_purpose::STANDARD.encode(data)
}
