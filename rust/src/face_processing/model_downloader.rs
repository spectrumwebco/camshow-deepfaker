use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{self, Write};
use anyhow::{Result, anyhow};
use log::{info, warn, error};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};

pub fn download_model(url: &str, output_path: &Path) -> Result<()> {
    info!("Downloading model from {} to {}", url, output_path.display());
    
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("camshow-deepfaker-rust/0.1.0"));
    
    let client = Client::builder()
        .default_headers(headers)
        .build()?;
    
    let response = client.get(url).send()?;
    
    if !response.status().is_success() {
        return Err(anyhow!("Failed to download model: HTTP status {}", response.status()));
    }
    
    let content_length = response.content_length().unwrap_or(0);
    
    let mut file = File::create(output_path)?;
    
    let bytes = response.bytes()?;
    file.write_all(&bytes)?;
    
    info!("Downloaded {} bytes to {}", bytes.len(), output_path.display());
    
    Ok(())
}

pub fn get_model_url(model_name: &str) -> Option<String> {
    match model_name {
        "inswapper_128.onnx" => Some("https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx".to_string()),
        "inswapper_256.onnx" => Some("https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_256.onnx".to_string()),
        "buffalo_l.onnx" => Some("https://huggingface.co/hacksider/deep-live-cam/resolve/main/buffalo_l.onnx".to_string()),
        "gfpgan_1.4.onnx" => Some("https://huggingface.co/hacksider/deep-live-cam/resolve/main/gfpgan_1.4.onnx".to_string()),
        _ => None,
    }
}

pub fn download_all_models(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let models = vec![
        "inswapper_128.onnx",
        "buffalo_l.onnx",
        "gfpgan_1.4.onnx",
    ];
    
    let mut downloaded_paths = Vec::new();
    
    for model_name in models {
        let model_path = model_dir.join(model_name);
        
        if model_path.exists() {
            info!("Model {} already exists at {}", model_name, model_path.display());
            downloaded_paths.push(model_path);
            continue;
        }
        
        let url = get_model_url(model_name)
            .ok_or_else(|| anyhow!("No URL found for model: {}", model_name))?;
        
        match download_model(&url, &model_path) {
            Ok(_) => {
                info!("Successfully downloaded model {} to {}", model_name, model_path.display());
                downloaded_paths.push(model_path);
            },
            Err(e) => {
                error!("Failed to download model {}: {}", model_name, e);
                return Err(anyhow!("Failed to download model {}: {}", model_name, e));
            }
        }
    }
    
    Ok(downloaded_paths)
}
