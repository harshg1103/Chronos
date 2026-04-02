import cv2
import threading
import queue
from PIL import Image

class ActionRecognizer:
    def __init__(self):
        self.q = queue.Queue(maxsize=10)
        self.cached_actions = {}
        # Zero-Shot categories. Expandable.
        self.labels = ["walking", "running", "standing", "using phone", "carrying bag", "fighting", "sitting", "stealing"]
        
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
                self.cached_actions[track_id] = best_label.upper()
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
