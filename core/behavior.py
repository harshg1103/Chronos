import math

class BehaviorAnalyzer:
    def __init__(self, duration_threshold=10.0, speed_threshold=150.0):
        self.duration_threshold = duration_threshold
        self.speed_threshold = speed_threshold
        self.track_history = {}
    
    def analyze(self, active_tracks, timestamp):
        anomalies = []
        current_ids = set()
        
        for track_id, ltrb, _ in active_tracks:
            current_ids.add(track_id)
            cx = (ltrb[0] + ltrb[2]) / 2
            cy = (ltrb[1] + ltrb[3]) / 2
            
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    "start_time": timestamp,
                    "last_time": timestamp,
                    "last_pos": (cx, cy)
                }
            else:
                history = self.track_history[track_id]
                duration = timestamp - history["start_time"]
                
                dt = timestamp - history["last_time"]
                speed = 0.0
                if dt > 0:
                    px, py = history["last_pos"]
                    distance = math.hypot(cx - px, cy - py)
                    speed = distance / dt
                
                history["last_time"] = timestamp
                history["last_pos"] = (cx, cy)
                
                if duration > self.duration_threshold or speed > self.speed_threshold:
                    anomalies.append(track_id)
                    
        for tid in list(self.track_history.keys()):
            if tid not in current_ids:
                del self.track_history[tid]
                
        return anomalies
