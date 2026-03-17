import cv2
import json
import os
import numpy as np
from services.model_loader import model_loader
from services.database.search import search_face

# Load InsightFace model once
face_app = model_loader.face_app

def process_video_and_match(video_path):
    """
    Processes video frames with optimized resizing.
    Accuracy is maintained because we resize to a width (1280px) that is 
    still much larger than the face recognition model's input (112x112).
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Could not open video file"}

    # Get video properties
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0: fps = 25.0 # Fallback
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # milvus will handle search

    results = []
    frame_idx = 0
    frame_skip = 15 # Process every 15th frame

    print(f"Processing video: {frame_count} frames, {fps} FPS. Target resolution width: 1280px")

    while cap.isOpened():
        # Fast frame skipping using grab()
        if frame_idx % frame_skip != 0:
            if not cap.grab(): break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # LOGIC: Resize Frame Before Detection
        # We resize to 1280px width. This is enough to keep full facial detail 
        # (accuracy) while being 3x-10x faster than 4K/2K processing.
        target_width = 1280
        if frame_width > target_width:
            scale = target_width / float(frame_width)
            # Use INTER_AREA for downsizing (best quality/accuracy)
            proc_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            proc_frame = frame

        # Detect and Recognize faces on the scaled frame
        faces = face_app.get(proc_frame)
        frame_results = []

        for face in faces:
            target_embedding = face.embedding
            
            if target_embedding is not None:
                # Normalize embedding for stable comparison
                norm_val = np.linalg.norm(target_embedding)
                if norm_val > 0:
                    target_embedding = target_embedding / norm_val

            # Find best match from Milvus search
            match_result = search_face(target_embedding)
            
            # Rescale bbox back to original video size for consistent UI display
            bbox = (face.bbox / scale).astype(int).tolist()

            frame_results.append({
                "name": match_result["person_name"],
                "status": match_result["status_text"],
                "is_matched": match_result["is_matched"],
                "similarity": match_result["similarity"],
                "bbox": bbox
            })

        if frame_results:
            # Fixed potential type issue for linter
            timestamp = int((float(frame_idx) / fps) * 1000.0)
            results.append({
                "millisecond": timestamp,
                "faces": frame_results
            })

        if frame_idx % 150 == 0:
            print(f"Progress: {frame_idx}/{frame_count} frames processed...")

        frame_idx += 1

    cap.release()
    print("Optimization complete: Video processing finished.")

    # Save results to JSON
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(current_dir, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    output_json = os.path.join(results_dir, "match_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    return results