import cv2
import numpy as np
from insightface.utils import face_align
from services.model_loader import model_loader

def detect_faces(image_path_or_buf):
    """
    Detects faces in an image and returns face objects containing:
    - bbox: bounding box
    - kps: keypoints
    - det_score: detection confidence
    - embedding: face representation
    """
    if isinstance(image_path_or_buf, str):
        img = cv2.imread(image_path_or_buf)
    else:
        img = image_path_or_buf
        
    if img is None:
        return []
        
    faces = model_loader.face_app.get(img)
    return faces

def align_face(img, face):
    """
    Aligns and crops the face image to a standard 112x112 size.
    This ensures same facial region and orientation for better matching.
    """
    if face.kps is not None:
        return face_align.norm_crop(img, landmark=face.kps)
    return None

def draw_faces(image, faces):
    """
    Utility to draw bounding boxes on an image for debugging.
    """
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    return image  