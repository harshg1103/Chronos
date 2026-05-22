import cv2
import threading
import queue
from PIL import Image

class ActionRecognizer:
    def __init__(self):
        # Use LifoQueue so we analyze the most recent crop if there's a backlog
        self.q = queue.LifoQueue(maxsize=30)
        self.cached_actions = {}
        self.action_history = {}
        # Zero-Shot categories. Expandable.
        self.label_map = {
            "a photo of a person walking naturally": "WALKING",
            "a photo of a person running or jogging": "RUNNING",
            "a photo of a person standing still": "STANDING",
            "a photo of a person loitering, browsing items, or looking around a store": "LOITERING",
            "a close-up photo of a person explicitly using a mobile phone": "USING PHONE",
            "a photo of a person carrying a heavy bag or backpack": "CARRYING BAG",
            "a photo of people physically fighting or punching": "FIGHTING",
            "a photo of a person sitting down on a chair or bench": "SITTING",
            "a photo of a person reaching for an item on a store shelf": "REACHING",
            "a photo of a person secretly stealing, shoplifting, and putting an item into their pocket or jacket": "SHOPLIFTING",
            "a blurry ambiguous photo of a person": "UNKNOWN"
        }
        self.labels = list(self.label_map.keys())
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            print(f"[Action Engine] Loading HuggingFace CLIP Zero-Shot on Device: {device}")
            print("[Action Engine] WARNING: If this is the first run, the CLIP model (~600MB) is downloading silently! Please wait...")
            
            # Using openai/clip-vit-base-patch32 for zero-shot image classification
            self.classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32", device=device)
            print("[Action Engine] Model loaded perfectly. Awaiting crops...")
        except BaseException as e:
            import traceback
            print(f"[Action Engine] Failed to load PyTorch: {e}")
            traceback.print_exc()
            self.classifier = None

        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                continue
                
            track_id, image_crop = item
            
            if self.classifier is None:
                # If PyTorch failed, fallback
                self.cached_actions[track_id] = "UNKNOWN"
                self.q.task_done()
                continue
                
            try:
                # Convert BGR crop to generic RGB for CLIP
                pil_img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
                result = self.classifier(pil_img, candidate_labels=self.labels)
                
                # The first label is the highest probability
                best_label = result[0]['label']
                if best_label in self.label_map:
                    mapped_label = self.label_map[best_label]
                else:
                    mapped_label = best_label.upper()
                    
                # Temporal Logic Override: Cannot be shoplifting unless previously loitering or reaching
                if mapped_label == "SHOPLIFTING":
                    history = self.action_history.get(track_id, set())
                    if "LOITERING" not in history and "REACHING" not in history:
                        mapped_label = "LOITERING"
                        
                self.cached_actions[track_id] = mapped_label
                
                if track_id not in self.action_history:
                    self.action_history[track_id] = set()
                self.action_history[track_id].add(mapped_label)
            except Exception as e:
                import traceback
                print(f"[Action Engine] Inference Exception: {e}")
                traceback.print_exc()
                
            self.q.task_done()

    def analyze_crop(self, track_id, image_crop):
        """Asynchronously dispatches the bounding box pixel crop to the PyTorch thread."""
        # Fast reject bad boxes
        if image_crop.shape[0] < 20 or image_crop.shape[1] < 20:
            return
            
        # Fast reject if queue full (stops video lagging)
        if self.q.full():
            return
            
        # Deepcopy pixels physically before threading
        self.q.put_nowait((track_id, image_crop.copy()))
        
    def get_action(self, track_id):
        """Returns the dynamically updating string action of the bounding box!"""
        return self.cached_actions.get(track_id, "ANALYZING...")
