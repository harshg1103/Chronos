# Chronos: Cognitive Surveillance Terminal

Chronos is an advanced, AI-powered multi-camera surveillance system built for proactive threat detection, continuous action recognition, and persistent subject tracking. Unlike traditional CCTVs that statically record footage, Chronos utilizes cutting-edge Machine Learning and Large Language Models to mathematically analyze, log, and classify every biological entity passing through its optical sensors in real-time.

## 🚀 Key Features

* **Multi-Camera Grid Compositing:** Upload and process up to 4 simultaneous video feeds. The backend syncs and stitches them into a unified MJPEG stream to conserve UI bandwidth.
* **Persistent Re-Identification (Re-ID):** Uses high-dimensional embedding vectors stored in ChromaDB to track individuals across different cameras seamlessly. If `Subject-1` leaves Camera A and enters Camera B, the system instantly recognizes them.
* **Local Zero-Shot Action Engine:** Deploys a background PyTorch thread utilizing HuggingFace's `CLIP` transformer to dynamically classify human actions (`"Walking"`, `"Running"`, `"Using Phone"`, `"Fighting"`) completely offline using bounding-box matrix crops.
* **Multimodal Cognitive Anomaly Detection:** When physical kinematic thresholds (like excessive sprinting or loitering) are breached, the exact video frame is dispatched to Google's Gemini 3.1 Pro Vision model to generate a fluent English description of the suspicious event.
* **Vector Semantic Search:** Every entry, exit, action change, and anomaly is embedded into ChromaDB. Operators can perform natural language queries (e.g., *"Who was walking?"*) to instantly filter telemetry logs.
* **Cyber-Telemetry Dashboard:** A React-based glassmorphism User Interface featuring active system polling, Server-Sent Events (SSE) for live event logs, and dynamic Subject AR overlays.

## 🧠 Technology Stack

### Backend (Python/FastAPI)
* **Core API:** `FastAPI` + `Uvicorn`
* **Object Detection:** `YOLOv8` (Ultralytics)
* **Kinematic Tracking:** `DeepSORT Realtime`
* **Action Recognition:** `PyTorch` + `Transformers` (OpenAI CLIP Zero-Shot)
* **Memory & Re-ID:** `ChromaDB` (Persistent Vector Database)
* **Vision LLM:** `google-generativeai` (Gemini 3.1 Pro)
* **Computer Vision:** `OpenCV` + `NumPy` + `PIL`

### Frontend (React/Vite)
* **Framework:** `React` (Vite build system)
* **Styling:** CSS3 variables, Glassmorphism, Cyberpunk keyframe animations.
* **Icons:** `lucide-react`

## 📂 Architecture Overview

* `api/main_api.py`: The central hub that routes video looping, aggregates multi-camera frames, handles WebSocket/SSE streams, and dispatches the PyTorch threads.
* `core/detection.py`: YOLOv8 wrapper strictly filtered to biological humans (`class 0`, `conf 0.45`).
* `core/tracking.py`: DeepSORT wrapper that extracts 1280-dimension visual feature vectors from detected humans.
* `core/reid.py`: Compares new feature vectors against the ChromaDB database using Cosine Similarity to assign permanent Subject IDs.
* `core/action.py`: Asynchronous PyTorch worker queue that crunches image crops to classify exact physical states.
* `core/behavior.py`: Mathematical bounding-box engine identifying kinematic anomalies (Speed > 150, Loitering > 10s).
* `core/cognitive.py`: Interface to Gemini 3.1 Pro for Anomaly text-generation and text-embedding into ChromaDB for search.

## 🛠️ Installation & Setup

1. **Clone the Repository**
2. **Setup Python Virtual Environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. **Install Core Dependencies:**
   ```powershell
   pip install fastapi uvicorn opencv-python numpy pillow python-multipart python-dotenv chromadb google-generativeai deep-sort-realtime ultralytics
   ```
4. **Install PyTorch Action Engine:**
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers
   ```
5. **Configure Environment:**
   Create a `.env` file in the root directory and add your Gemini key:
   `GEMINI_API_KEY="your_api_key_here"`

6. **Install Frontend Subsystem:**
   ```powershell
   cd frontend
   npm install
   ```

## ⚡ How to Run

You will need two terminals to run the system matrix.

**Terminal 1: FastAPI Backend**
```powershell
.venv\Scripts\Activate.ps1
uvicorn api.main_api:app --reload
```

**Terminal 2: React Frontend**
```powershell
cd frontend
npm run dev
```

Browse to `http://localhost:5173`. Click **UPLOAD ARCHIVES**, grab 1 to 4 test `.mp4` CCTV videos (like `cctv_mall_angle.mp4`), and watch Chronos analyze the timeline.
