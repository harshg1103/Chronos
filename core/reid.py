import chromadb

class PersistentReID:
    def __init__(self, db_path="./chroma_db", threshold=0.15):
        self.client = chromadb.PersistentClient(path=db_path)
        # Using cosine similarity to match the 1280D appearance vectors
        self.collection = self.client.get_or_create_collection(
            name="person_reid",
            metadata={"hnsw:space": "cosine"}
        )
        self.threshold = threshold
        self.session_cache = {}
        
    def identify(self, track_id, feature_vector):
        """
        Receives an active track ID and its 1D numpy array appearance vector.
        Returns the persistent Subject name (e.g. 'Subject-1').
        """
        if feature_vector is None:
            return f"ID-{track_id}"
            
        if track_id in self.session_cache:
            return self.session_cache[track_id]
            
        vector = feature_vector.tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=1
            )
            
            if results['distances'] and len(results['distances'][0]) > 0:
                dist = results['distances'][0][0]
                subject_id = results['ids'][0][0]
                
                if dist < self.threshold:
                    self.session_cache[track_id] = subject_id
                    return subject_id
        except Exception:
            pass
            
        # If no match or DB is empty, register new person permanently
        try:
            count = self.collection.count()
        except:
            count = 0
            
        new_subject_id = f"Subject-{count + 1}"
        
        try:
            self.collection.add(
                embeddings=[vector],
                ids=[new_subject_id],
                metadatas=[{"system": "reid"}]
            )
        except Exception:
            pass
            
        self.session_cache[track_id] = new_subject_id
        return new_subject_id
