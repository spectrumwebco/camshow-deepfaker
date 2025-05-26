#!/usr/bin/env python3
"""
FastAPI web server for Camshow Deepfaker.
This provides a headless API interface for the deepfake functionality.
"""
import os
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from modules import core, globals

app = FastAPI(
    title="Camshow Deepfaker API",
    description="API for real-time face swap and video deepfake",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    source_path: str
    target_path: str
    output_path: Optional[str] = None
    frame_processors: List[str] = ["face_swapper"]
    keep_fps: bool = True
    keep_audio: bool = True
    many_faces: bool = False
    execution_provider: str = "cpu"

class StatusResponse(BaseModel):
    status: str
    message: str

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "Camshow Deepfaker API",
        "version": "1.0.0",
        "description": "API for real-time face swap and video deepfake"
    }

@app.post("/upload", response_class=JSONResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a source or target file."""
    try:
        os.makedirs("uploads", exist_ok=True)
        
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_class=JSONResponse)
async def process_media(request: ProcessRequest):
    """Process media with face swap."""
    try:
        globals.source_path = request.source_path
        globals.target_path = request.target_path
        globals.output_path = request.output_path or f"output_{os.path.basename(request.target_path)}"
        globals.frame_processors = request.frame_processors
        globals.keep_fps = request.keep_fps
        globals.keep_audio = request.keep_audio
        globals.many_faces = request.many_faces
        
        globals.headless = True
        
        core.start()
        
        return {
            "status": "success", 
            "message": "Processing completed", 
            "output_path": globals.output_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}", response_class=FileResponse)
async def download_file(filename: str):
    """Download a processed file."""
    file_path = f"output_{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/status", response_class=JSONResponse)
async def get_status():
    """Get the current status of the system."""
    return {
        "status": "online",
        "execution_providers": globals.execution_providers,
        "frame_processors": globals.frame_processors
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
