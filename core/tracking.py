from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age)
    
    def update(self, detections, frame):
        tracks = self.tracker.update_tracks(detections, frame=frame)
        active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            feature = track.features[-1] if track.features and len(track.features) > 0 else None
            active_tracks.append((track_id, ltrb, feature))
        return active_tracks
