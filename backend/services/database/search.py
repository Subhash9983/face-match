from services.database.milvus import collection
from services.database.mongo import get_metadata

# Match threshold - 0.42: catches more valid matches
# If too many false names appear, increase to 0.47
MATCH_THRESHOLD = 0.42

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
        limit=1,   # Check top 3 to see if Abhishek is hiding at rank 2 or 3
        expr=None
    )

    if not results or len(results[0]) == 0:
        return _unknown_result()

    # DEBUG: Print top 3 matches to terminal
    print(f"--- Top 3 Matches for current face ---")
    for i, hit in enumerate(results[0]):
        meta = get_metadata(hit.id)
        name = meta["name"] if meta else "No Metadata"
        print(f"  Rank {i+1}: '{name}' score={hit.distance:.4f}")

    best_hit = results[0][0]
    best_id = best_hit.id
    best_score = best_hit.distance
    
    metadata = get_metadata(best_id)
    if not metadata:
        return _unknown_result()

    best_name = metadata["name"]
    is_matched = (best_score >= MATCH_THRESHOLD)

    # DEBUG: print every search result so you can see actual scores
    print(f"[DB SEARCH] Best match: '{best_name}' score={best_score:.4f} "
          f"threshold={MATCH_THRESHOLD} matched={is_matched}")

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
