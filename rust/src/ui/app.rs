use dioxus::prelude::*;
use pyo3::prelude::*;

#[component]
pub fn App() -> Element {
    let theme = use_state(|| "dark");
    
    rsx! {
        div { class: "app-container {theme}",
            header { class: "app-header",
                h1 { "Camshow Deepfaker" }
                div { class: "app-controls",
                    button { 
                        onclick: move |_| theme.set("light"),
                        "Toggle Theme" 
                    }
                }
            }
            main { class: "app-content",
                CameraPreview {}
                ControlPanel {}
            }
            footer { class: "app-footer",
                p { "Powered by Rust + Dioxus" }
            }
        }
    }
}

#[component]
fn CameraPreview() -> Element {
    let frame = use_state(|| None::<Vec<u8>>);
    
    rsx! {
        div { class: "camera-preview",
            h2 { "Camera Preview" }
            div { class: "preview-container",
                if let Some(frame_data) = frame.get() {
                    img { 
                        src: "data:image/jpeg;base64,{encode_base64(frame_data)}" 
                    }
                } else {
                    div { class: "no-camera",
                        p { "No camera feed available" }
                        button { "Start Camera" }
                    }
                }
            }
        }
    }
}

#[component]
fn ControlPanel() -> Element {
    let selected_model = use_state(|| "inswapper_128.onnx");
    let enhance_faces = use_state(|| true);
    
    rsx! {
        div { class: "control-panel",
            h2 { "Face Processing Controls" }
            
            div { class: "control-group",
                label { "Face Swap Model:" }
                select { 
                    value: "{selected_model}",
                    onchange: move |evt| selected_model.set(evt.value.clone()),
                    option { value: "inswapper_128.onnx", "Insightface 128" }
                    option { value: "inswapper_256.onnx", "Insightface 256" }
                }
            }
            
            div { class: "control-group",
                label { "Face Enhancement:" }
                input { 
                    r#type: "checkbox",
                    checked: "{enhance_faces}",
                    onchange: move |_| enhance_faces.set(!enhance_faces.get())
                }
            }
            
            div { class: "control-group",
                button { class: "primary-button", "Process Frame" }
                button { class: "secondary-button", "Save Image" }
            }
        }
    }
}

fn encode_base64(data: &[u8]) -> String {
    use base64::{Engine as _, engine::general_purpose};
    general_purpose::STANDARD.encode(data)
}
