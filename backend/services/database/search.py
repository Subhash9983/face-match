from services.database.milvus import collection
from services.database.mongo import get_metadata

# Match threshold (Lowered for CCTV sensitivity)
MATCH_THRESHOLD = 0.45

def search_face(embedding):
    """
    Searches for the closest face embedding in Milvus and retrieves metadata from MongoDB.
    """
    if embedding is None:
        return _unknown_result()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    
    # Search in Milvus
    results = collection.search(
        data=[embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=1,
        expr=None
    )

    if not results or len(results[0]) == 0:
        return _unknown_result()

    best_hit = results[0][0]
    best_id = best_hit.id
    best_score = best_hit.distance # In COSINE, higher is better? Actually Milvus COSINE is distance 1-sim usually? 
    # Actually Milvus MetricType.COSINE: the distance range is [-1, 1]. The larger the distance, the more similar the vectors are.
    
    metadata = get_metadata(best_id)
    if not metadata:
        return _unknown_result()

    best_name = metadata["name"]
    is_matched = (best_score >= MATCH_THRESHOLD)

    if is_matched:
        display_name = best_name.rsplit('.', 1)[0]
    else:
        display_name = "Unknown"

    return {
        "person_name": display_name,
        "similarity": float(best_score),
        "is_matched": is_matched,
        "status_text": "Matched Face" if is_matched else "Not Matched",
        "threshold_used": MATCH_THRESHOLD
    }

def _unknown_result():
    return {
        "person_name": "Unknown",
        "similarity": 0.0,
        "is_matched": False,
        "status_text": "Not Matched",
        "threshold_used": MATCH_THRESHOLD
    }
