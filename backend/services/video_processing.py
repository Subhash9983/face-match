import cv2
import json
import os
import numpy as np
import supervision as sv
from services.model_loader import model_loader
from services.database.search import search_face
from services.face_detection import align_face

face_app = model_loader.face_app

# ================= GLOBAL MEMORY =================
# Reset at start of each video processing call
global_identities: dict = {}
# gid -> {
#   embedding        : np.ndarray — running avg
#   last_seen        : int        — frame_idx
#   name             : str        — "Unknown" or Recognized
#   confidence       : float      — best match score
#   is_locked        : bool       — True if confirmed over multiple frames
#   lock_expiry      : int        — frame_idx when lock should expire
#   confirm_count    : int        — consecutive frames with same identity
#   last_db_check    : int        — frame_idx of last Milvus call
# }

track_to_global: dict = {}      
track_embedding_buffer: dict = {}
track_db_cooldown: dict = {}
next_global_id: int = 1

# ---- Production Constants ----
MATCH_IDENTITY_THRESH = 0.40   # Lowered to 0.40 for much better stability through occlusions
MATCH_MIN_SIM        = 0.42   # Min similarity to show any name
LOCK_SIM_THRESH      = 0.42   # Same as match — every match increments lock counter
LOCK_CONFIRM_FRAMES  = 1      # Immediate locking: just 1 match needed
LOCK_DURATION_SEC    = 60     # How many seconds to cache identity
DB_COOLDOWN_FRAMES   = 40     # Gap between Milvus calls for unlocked tracks
EMBED_BLEND_RATIO    = 0.30   # 70% old + 30% new (Stable but adaptive)


# ================= UTILS =================
def normalize(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def match_identity(embedding: np.ndarray, threshold: float = MATCH_IDENTITY_THRESH):
    """Link a current track's face to a previously seen Global ID (GID)."""
    best_gid, best_sim = None, 0.0
    for gid, data in global_identities.items():
        sim = float(np.dot(data["embedding"], embedding))
        if sim > threshold and sim > best_sim:
            best_sim, best_gid = sim, gid
    return best_gid, best_sim


def update_embedding_buffer(track_id: int, emb: np.ndarray, max_size: int = 5) -> np.ndarray:
    """Keep a short buffer of last 5 normalized embeddings to smooth noise."""
    if track_id not in track_embedding_buffer:
        track_embedding_buffer[track_id] = []
    buf = track_embedding_buffer[track_id]
    buf.append(emb)
    if len(buf) > max_size:
        buf.pop(0)
    return normalize(np.mean(buf, axis=0))


def should_query_db(track_id: int, gid: int, frame_idx: int, is_new: bool) -> bool:
    """Smart Gatekeeper: Minimizes Milvus calls without losing precision."""
    identity = global_identities[gid]

    # Rule 1: LOCKED & NOT EXPIRED? Skip DB call.
    if identity["is_locked"] and frame_idx < identity["lock_expiry"]:
        return False

    # Rule 2: Brand new identity? Always check DB.
    if is_new:
        return True

    # Rule 3: Lock Expired? Re-check DB.
    if frame_idx >= identity["lock_expiry"]:
        return True

    # Rule 4: Cooldown for unlocked tracks
    if not identity["is_locked"]:
        # If we haven't checked for a while, try again
        if frame_idx - identity["last_db_check"] > DB_COOLDOWN_FRAMES:
             return True

    return False


def process_video_and_match(video_path: str):
    """Main pipeline: Detect -> Track -> Recognize -> Draw -> Save."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1 second = fps frames. 60 seconds = 60 * fps.
    lock_frames_duration = int(LOCK_DURATION_SEC * fps)

    # Reset Globals for this run
    global next_global_id, global_identities, track_to_global
    next_global_id = 1
    global_identities.clear()
    track_to_global.clear()
    tracker = sv.ByteTrack(
        frame_rate=fps,
        lost_track_buffer=400,          # 16 seconds of tracking memory
        minimum_matching_threshold=0.2
    )

    frame_idx    = 0
    frame_skip   = 4
    detect_width = 1280   # High resolution for accurate detection of small faces

    current_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.abspath(os.path.join(current_dir, "..", "results"))
    os.makedirs(res_dir, exist_ok=True)

    out = cv2.VideoWriter(
        os.path.join(res_dir, "processed_video.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )

    last_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % frame_skip == 0:
            # --- Detection Pre-processing ---
            scale = detect_width / width if width > detect_width else 1.0
            small = cv2.resize(frame, (int(width*scale), int(height*scale))) if scale < 1.0 else frame
            
            # --- InsightFace Step ---
            faces = face_app.get(small)

            if not faces:
                detections = sv.Detections.empty()
                face_bboxes_orig = []
            else:
                face_bboxes_orig = [f.bbox / scale for f in faces]
                detections = sv.Detections(
                    xyxy=np.array(face_bboxes_orig, dtype=np.float32),
                    confidence=np.array([f.det_score for f in faces]),
                    class_id=np.zeros(len(faces), dtype=int)
                )

            # --- ByteTrack step ---
            tracks = tracker.update_with_detections(detections)
            last_detections = []

            for det in tracks:
                track_id = int(det[4])
                bbox     = det[0].astype(int)

                # --- IoU Match to find original Face object ---
                best_face, best_iou = None, 0.15
                for fi, fb_orig in enumerate(face_bboxes_orig):
                    iou = compute_iou(bbox, fb_orig)
                    if iou > best_iou:
                        best_iou, best_face = iou, faces[fi]

                if best_face is None or best_face.embedding is None:
                    continue

                # --- HIGH-QUALITY ALIGNMENT ---
                # Keypoints scaled back to original frame for precision
                if best_face.kps is not None:
                    kps_high = best_face.kps / scale
                    best_face.kps = kps_high # update to original coords
                    align_face(frame, best_face) # internal alignment logic

                # --- Embedding Update ---
                curr_emb = normalize(best_face.embedding)
                avg_emb  = update_embedding_buffer(track_id, curr_emb)

                # --- Identity Matching (Cross-frame memory) ---
                gid, sim = match_identity(avg_emb)

                if gid is None:
                    gid = next_global_id
                    next_global_id += 1
                    is_new = True
                    global_identities[gid] = {
                        "embedding": avg_emb, "last_seen": frame_idx, "name": "Unknown",
                        "confidence": -1.0, "is_locked": False, "lock_expiry": 0,
                        "confirm_count": 0, "last_db_check": -1
                    }
                else:
                    is_new = False
                    # Adaptive Blend: 70% old + 30% new (Production Standard)
                    old_emb = global_identities[gid]["embedding"]
                    global_identities[gid]["embedding"] = normalize(
                        (1 - EMBED_BLEND_RATIO) * old_emb + EMBED_BLEND_RATIO * avg_emb
                    )
                    global_identities[gid]["last_seen"] = frame_idx

                track_to_global[track_id] = gid

                # --- SMART RECOGNITION (Database Calls) ---
                if should_query_db(track_id, gid, frame_idx, is_new):
                    res = search_face(avg_emb)
                    sim = res["similarity"]
                    track_db_cooldown[track_id] = frame_idx
                    global_identities[gid]["last_db_check"] = frame_idx

                    if sim > global_identities[gid]["confidence"]:
                        global_identities[gid]["confidence"] = sim

                    if res["is_matched"] and sim >= MATCH_MIN_SIM:
                        global_identities[gid]["name"] = res["person_name"]

                        # --- TEMPORAL CONSISTENCY LOCKING ---
                        if sim >= LOCK_SIM_THRESH:
                            global_identities[gid]["confirm_count"] += 1
                            if global_identities[gid]["confirm_count"] >= LOCK_CONFIRM_FRAMES:
                                global_identities[gid]["is_locked"] = True
                                global_identities[gid]["lock_expiry"] = frame_idx + lock_frames_duration
                                print(f"[PROD-LOCK] Identity '{res['person_name']}' confirmed and LOCKED for 1 minute.")
                        else:
                            global_identities[gid]["confirm_count"] = 0

                # Data for drawing
                data = global_identities[gid]
                last_detections.append({
                    "gid": gid, "bbox": bbox, "name": data["name"],
                    "confidence": data["confidence"], "is_locked": data["is_locked"]
                })

        # ===== Drawing Logic =====
        for d in last_detections:
            x1, y1, x2, y2 = d["bbox"]
            color = (0, 255, 0) if d["name"] != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Visual indicators for Production UI
            lock_status = " 🔒" if d["is_locked"] else ""
            label = f"P{d['gid']} {d['name']} ({d['confidence']:.2f}){lock_status}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        if frame_idx % 150 == 0:
            print(f"Progress: {frame_idx}/{f_count} frames processed...")
        frame_idx += 1

        # Clean old identities (unseen > 1000 frames)
        for gid in list(global_identities.keys()):
            if frame_idx - global_identities[gid]["last_seen"] > 1000:
                del global_identities[gid]

    cap.release()
    out.release()

    # Create Summary
    summary = []
    seen_names = set()
    for gid, data in global_identities.items():
        if data["name"] != "Unknown" and data["name"] not in seen_names:
            seen_names.add(data["name"])
            summary.append({
                "id": f"P{gid}", "name": data["name"], 
                "confidence": round(float(data["confidence"]), 2), "status": "Present"
            })

    output_json = os.path.join(res_dir, "match_results.json")
    with open(output_json, "w") as f:
        json.dump({"summary": summary, "video_url": "/results/processed_video.mp4"}, f, indent=4)

    return {"summary": summary, "video_url": "/results/processed_video.mp4"}


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)