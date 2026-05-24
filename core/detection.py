import torch
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        # Move the YOLOv8 model explicitly to the GPU (CUDA)
        self.model.to(self.device)
        print(f"[Detector Engine] Locked YOLOv8 model onto hardware: {self.device.upper()}")
    
    def detect(self, frame):
        # Force inference to run on the GPU device
        results = self.model(frame, classes=[0], conf=0.45, verbose=False, device=self.device)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))
        return detections
