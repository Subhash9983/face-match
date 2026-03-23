
from insightface.app import FaceAnalysis

class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            print("Loading InsightFace model...")
            cls._instance = super(ModelLoader, cls).__new__(cls)
            # Initialize InsightFace App
            # buffalo_l is the largest and most accurate model set
            cls._instance.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            cls._instance.app.prepare(ctx_id=1, det_size=(640, 640))
        return cls._instance

    @property
    def face_app(self):
        return self.app

# Global instance for easy access
model_loader = ModelLoader()
