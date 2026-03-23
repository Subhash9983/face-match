import os
import shutil
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response

from services.embedding_generation import get_multiple_embeddings
from services.video_processing import process_video_and_match
router = APIRouter()

# Configuration
BASE_DIR = os.path.dirname(__file__)
PHOTOS_DIR = os.path.join(BASE_DIR, "uploads", "photos")
VIDEOS_DIR = os.path.join(BASE_DIR, "uploads", "videos")

os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

@router.post("/upload-photos")
async def upload_photos(files: List[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        
        file_path = os.path.join(PHOTOS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file.filename)
    
    # Generate and save embeddings once during upload
    photo_files = [os.path.join(PHOTOS_DIR, f) for f in os.listdir(PHOTOS_DIR) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if photo_files:
        _ = get_multiple_embeddings(photo_files)
    
    return {
        "message": f"Successfully uploaded {len(uploaded_files)} photos and updated embeddings",
        "photos": uploaded_files
    }

@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
    
    file_path = os.path.join(VIDEOS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "Video uploaded successfully",
        "filename": file.filename
    }


@router.post("/start-matching")
async def start_matching():
    # milvus/mongo will provide embeddings during search
    
    # 2. Get the latest uploaded video
    video_files = [os.path.join(VIDEOS_DIR, f) for f in os.listdir(VIDEOS_DIR) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        raise HTTPException(status_code=400, detail="No video found. Please upload a video first.")
    
    # Using the most recently uploaded video
    latest_video = max(video_files, key=os.path.getmtime)
    
    # 3. Process video and match faces
    try:
        match_results = process_video_and_match(latest_video)
        return {
            "status": "success",
            "results": match_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")

@router.get("/download-processed-video")
async def download_processed_video():
    results_dir = os.path.join(BASE_DIR, "results")
    video_path = os.path.join(results_dir, "processed_video.mp4")
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Processed video not found. Please run matching first.")
    
    # Read file and return with explicit headers to force download
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    
    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "Content-Disposition": 'attachment; filename="face_match_output.mp4"',
            "Content-Length": str(len(video_bytes))
        }
    )
