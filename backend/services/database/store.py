from services.database.milvus import collection
from services.database.mongo import save_metadata, get_id_by_name

def save_face(name, embedding):
    """
    Saves a face embedding to Milvus and its metadata to MongoDB.
    Prevents duplicates by checking the name first.
    """
    if embedding is None:
        return None
    
    # 1. Check if person already exists
    existing_id = get_id_by_name(name)
    if existing_id:
        print(f"Face already exists: {name} (Milvus ID: {existing_id}). Skipping insert.")
        return existing_id

    # 2. Insert new face
    # Milvus expects a list of vectors
    mr = collection.insert([[embedding.tolist()]])
    milvus_id = mr.primary_keys[0]
    
    # 3. Save metadata to MongoDB
    save_metadata(milvus_id, name)
    
    print(f"Saved face: {name} with Milvus ID: {milvus_id}")
    return milvus_id
