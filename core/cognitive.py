import os
import asyncio
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception:
    pass

class CognitiveLayer:
    def __init__(self, db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="security_events")
        try:
            self.model = genai.GenerativeModel('gemini-3.1-pro')
        except Exception:
            self.model = None
            
        self.processing = set()
    
    async def process_anomaly(self, frame_image, timestamp):
        event_id = str(timestamp)
        if event_id in self.processing:
            return ""
        self.processing.add(event_id)
        
        try:
            if not self.model:
                return ""
            prompt = "Describe the security event in one sentence"
            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt, frame_image]
            )
            description = response.text
            self.collection.add(
                documents=[description],
                metadatas=[{"timestamp": timestamp}],
                ids=[event_id]
            )
            return description
        except Exception:
            return ""
        finally:
            self.processing.discard(event_id)
            
    def log_event(self, text, timestamp):
        event_id = f"log_{timestamp}_{hash(text)}"
        try:
            self.collection.add(
                documents=[text],
                metadatas=[{"timestamp": timestamp}],
                ids=[event_id]
            )
        except Exception:
            pass
            
    def search_events(self, query, n_results=5):
        try:
            count = self.collection.count()
            if count == 0:
                return {"documents": [], "metadatas": []}
                
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, count)
            )
            return results
        except Exception as e:
            return {"documents": [], "metadatas": []}
