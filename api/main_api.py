import asyncio
import time
import cv2
import PIL.Image
import os
import shutil
import numpy as np
from typing import List
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.detection import PersonDetector
from core.tracking import Tracker
from core.behavior import BehaviorAnalyzer
from core.cognitive import CognitiveLayer
from core.reid import PersistentReID
from core.action import ActionRecognizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = PersonDetector()
tracker = Tracker()
behavior_analyzer = BehaviorAnalyzer()
cognitive_layer = CognitiveLayer()
reid_system = PersistentReID()
action_engine = ActionRecognizer()

is_running = False
latest_alert = ""
processing_task = None
cooldown_until = 0

latest_frame_jpeg = b""
alert_queues = []

system_start_time = time.time()
total_anomalies_detected = 0

video_sources = [0]
os.makedirs("uploads", exist_ok=True)

async def video_processing_loop():
    global is_running, latest_alert, cooldown_until, latest_frame_jpeg, total_anomalies_detected
    
    global_active_subjects = set()
    subject_actions = {}
    
    def dispatch_alert(text):
        global latest_alert
        latest_alert = text
        for q in alert_queues:
            try:
                q.put_nowait(text)
            except Exception:
                pass

    caps = [cv2.VideoCapture(src) for src in video_sources]
    trackers = [Tracker() for _ in range(len(caps))]
    analyzers = [BehaviorAnalyzer() for _ in range(len(caps))]
    
    target_fps = 60
    delay = 1.0 / target_fps
    
    frame_counter = 0
    
    while is_running:
        loop_start = time.time()
        frame_counter += 1
        dispatch_actions = (frame_counter % 15 == 0)
        
        current_frame_subjects = set()
        
        frames = []
        statuses = []
        for i, cap in enumerate(caps):
            ret, frame = await asyncio.to_thread(cap.read)
            statuses.append(ret)
            if ret:
                frame = cv2.resize(frame, (640, 480))
                if video_sources[i] == 0:
                    frame = cv2.flip(frame, 1)
                frames.append(frame)
            else:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                
        if not any(statuses):
            is_running = False
            break
            
        any_anomalies = False
        final_frames = []
        for i, frame in enumerate(frames):
            if not statuses[i]:
                cv2.putText(frame, "NO SIGNAL", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                final_frames.append(frame)
                continue
                
            detections = await asyncio.to_thread(detector.detect, frame)
            active_tracks = await asyncio.to_thread(trackers[i].update, detections, frame)
            
            current_time = time.time()
            anomalies = analyzers[i].analyze(active_tracks, current_time)
            
            if anomalies:
                any_anomalies = True
                
            for track_id, ltrb, feature in active_tracks:
                x1, y1, x2, y2 = map(int, ltrb)
                
                if dispatch_actions:
                    h, w = frame.shape[:2]
                    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    if cx2 > cx1 and cy2 > cy1:
                        action_engine.analyze_crop(track_id, frame[cy1:cy2, cx1:cx2])
                        
                persistent_id = await asyncio.to_thread(reid_system.identify, track_id, feature)
                act = action_engine.get_action(track_id)
                
                current_frame_subjects.add(persistent_id)
                if act != "ANALYZING..." and act != "UNKNOWN":
                    prev_act = subject_actions.get(persistent_id)
                    if prev_act != act:
                        subject_actions[persistent_id] = act
                        msg = f"[ACTION] {persistent_id} is {act}"
                        dispatch_alert(msg)
                        cognitive_layer.log_event(msg, current_time)
                
                color = (0, 0, 255) if track_id in anomalies else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{persistent_id} | ACT: {act}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            cv2.putText(frame, f"CAM {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,165,0), 2)
            final_frames.append(frame)
            
        entered = current_frame_subjects - global_active_subjects
        for subject in entered:
            msg = f"[SYSTEM] {subject} entered visual sector."
            dispatch_alert(msg)
            cognitive_layer.log_event(msg, time.time())
            
        exited = global_active_subjects - current_frame_subjects
        for subject in exited:
            msg = f"[SYSTEM] {subject} left visual sector."
            dispatch_alert(msg)
            cognitive_layer.log_event(msg, time.time())
            if subject in subject_actions:
                del subject_actions[subject]
                
        global_active_subjects = current_frame_subjects
            
        n = len(final_frames)
        if n == 1:
            stitched = final_frames[0]
        elif n == 2:
            stitched = np.hstack(final_frames)
        elif n == 3 or n == 4:
            top = np.hstack(final_frames[:2])
            if n == 3:
                bottom = np.hstack([final_frames[2], np.zeros_like(final_frames[0])])
            else:
                bottom = np.hstack(final_frames[2:4])
            stitched = np.vstack([top, bottom])
        else:
            stitched = final_frames[0]
            
        success, encoded_image = cv2.imencode('.jpg', stitched)
        if success:
            latest_frame_jpeg = encoded_image.tobytes()
            
        current_time = time.time()
        if any_anomalies and current_time > cooldown_until:
            cooldown_until = current_time + 5.0
            total_anomalies_detected += 1
            image = PIL.Image.fromarray(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
            task = asyncio.create_task(cognitive_layer.process_anomaly(image, current_time))
            def callback(t):
                global latest_alert
                try:
                    res = t.result()
                    if res:
                        latest_alert = res
                        for q in alert_queues:
                            q.put_nowait(res)
                except Exception:
                    pass
            task.add_done_callback(callback)
            
        if video_sources[0] != 0:
            elapsed = time.time() - loop_start
            sleep_time = delay - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0.001)
        else:
            await asyncio.sleep(0.001)
            
    for cap in caps:
        cap.release()
    latest_frame_jpeg = b""

@app.post("/upload_cameras")
async def upload_cameras(files: List[UploadFile] = File(...)):
    global video_sources, is_running, processing_task
    
    is_running = False
    if processing_task and not processing_task.done():
        await asyncio.sleep(0.2)
        
    video_sources = []
    for file in files:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        video_sources.append(file_path)
    
    is_running = True
    processing_task = asyncio.create_task(video_processing_loop())
    return {"status": "started", "sources": video_sources}

@app.post("/start")
async def start_processing():
    global is_running, processing_task, video_sources
    if not is_running:
        video_sources = [0]
        is_running = True
        processing_task = asyncio.create_task(video_processing_loop())
    return {"status": "started live"}

@app.post("/stop")
async def stop_processing():
    global is_running
    is_running = False
    return {"status": "stopped"}

@app.get("/search")
async def search(query: str):
    results = cognitive_layer.search_events(query)
    return {"results": results}

async def mjpeg_generator():
    while True:
        if not is_running or not latest_frame_jpeg:
            await asyncio.sleep(0.1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_jpeg + b'\r\n')
        await asyncio.sleep(0.016)

@app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

async def sse_generator(request: Request):
    q = asyncio.Queue()
    alert_queues.append(q)
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                alert_text = await asyncio.wait_for(q.get(), timeout=1.0)
                yield f"data: {alert_text}\n\n"
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
    finally:
        alert_queues.remove(q)

@app.get("/api/live_alerts")
async def live_alerts(request: Request):
    return StreamingResponse(sse_generator(request), media_type="text/event-stream")

@app.get("/api/stats")
async def get_stats():
    uptime = int(time.time() - system_start_time)
    try:
        subjects_known = reid_system.collection.count()
    except Exception:
        subjects_known = 0
    return {
        "uptime": uptime,
        "subjects_known": subjects_known,
        "anomalies_detected": total_anomalies_detected,
        "is_running": is_running
    }
