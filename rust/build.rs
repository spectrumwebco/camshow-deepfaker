use std::env;
use std::path::Path;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    
    if target_os == "macos" {
        if cfg!(target_arch = "aarch64") {
            println!("cargo:rustc-cfg=feature=\"macos_silicon\"");
        } else {
            println!("cargo:rustc-cfg=feature=\"macos_intel\"");
        }
    }
    
    if target_os == "linux" {
        if env::var("CUDA_HOME").is_ok() || Path::new("/usr/local/cuda").exists() {
            println!("cargo:rustc-cfg=feature=\"cuda\"");
        }
    }
    
    println!("cargo:warning=Building for {}", target_os);
}
