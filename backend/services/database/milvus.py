from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

def init_milvus():
    connections.connect("default", host="localhost", port="19530")
    
    collection_name = "face_collection"
    dim = 512  # InsightFace embedding size
    
    if utility.has_collection(collection_name):
        return Collection(collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    
    schema = CollectionSchema(fields, "Face recognition embeddings")
    collection = Collection(collection_name, schema)
    
    # Create index for vector search
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    collection.create_index("embedding", index_params)
    collection.load()
    return collection

collection = init_milvus()
