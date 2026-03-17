import os
import uvicorn
from fastapi import FastAPI
from routes import router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app
app = FastAPI(title="Video Face Matching System", version="1.0.0")

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directories exist
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
PHOTOS_DIR = os.path.join(UPLOADS_DIR, "photos")
VIDEOS_DIR = os.path.join(UPLOADS_DIR, "videos")

for path in [PHOTOS_DIR, VIDEOS_DIR]:
    os.makedirs(path, exist_ok=True)

# Mount static files if you want to view uploaded content directly
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

@app.get("/")
async def root():
    return {"message": "Video Face Matching API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Routes inclusion
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
