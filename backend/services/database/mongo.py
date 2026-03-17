from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["face_match_db"]
faces_metadata = db["faces_metadata"]

def save_metadata(milvus_id, name):
    faces_metadata.insert_one({
        "milvus_id": milvus_id,
        "name": name
    })

def get_metadata(milvus_id):
    return faces_metadata.find_one({"milvus_id": milvus_id})

def get_id_by_name(name):
    """Check if person already exists by filename/name."""
    record = faces_metadata.find_one({"name": name})
    return record["milvus_id"] if record else None
