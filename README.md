# 🧠 AI-Powered Live Session Monitoring System

A multimodal AI system that combines **speech**, **context**, and **visual cues** to evaluate communication quality, engagement, and behavior in live sessions (interviews, meetings, presentations).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    React Dashboard (Vite)                       │
│  ┌──────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ ┌──────────┐   │
│  │Gauge │ │Sentiment │ │ Speaker  │ │Topic  │ │  Video   │   │
│  │Score │ │Timeline  │ │Analysis  │ │Radar  │ │Engagement│   │
│  └──────┘ └──────────┘ └──────────┘ └───────┘ └──────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST + WebSocket
┌────────────────────────────▼────────────────────────────────────┐
│                    FastAPI Backend (:8000)                       │
│  ┌──────────────┐ ┌─────────────┐ ┌───────────────┐            │
│  │Audio Process │ │   NLP       │ │    Video      │            │
│  │(Whisper +    │ │Analysis     │ │  Processing   │            │
│  │ Diarization) │ │(Sentiment,  │ │(DeepFace,     │            │
│  │              │ │ Sarcasm,    │ │ MediaPipe,    │            │
│  │              │ │ Topics)     │ │ OpenCV)       │            │
│  └──────┬───────┘ └──────┬──────┘ └──────┬────────┘            │
│         └────────────────┼───────────────┘                     │
│                    ┌─────▼──────┐                               │
│                    │  Scoring   │                               │
│                    │  Engine    │                               │
│                    │(Multimodal │                               │
│                    │  Fusion)   │                               │
│                    └────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Audio Processing
- 🎙️ Whisper speech-to-text transcription
- 👥 Speaker diarization (pyannote / energy-based fallback)
- 🏷️ Dynamic role mapping (Interviewer / Candidate)
- 🎤 Live microphone capture support

### NLP Intelligence
- 😊 Sentiment analysis (TextBlob polarity)
- ❓ Interaction classification (Question / Answer / Prompt)
- 📚 Topic detection & coverage scoring
- ⚡ Sarcasm detection (text + context + visual mismatch)
- 🔍 Clarification tracking (confusion → resolution)
- 📊 Candidate metrics: Communication, Confidence, Positivity, Depth, Relevance

### Video Processing
- 👤 Face detection & counting (OpenCV)
- 😊 Facial emotion recognition (DeepFace)
- 👁️ Attention/gaze estimation (MediaPipe)
- 🧘 Head stability & posture tracking

### Multimodal Fusion
- 🧠 Composite **Conversation Intelligence Score** (0-100)
- Weighted combination: Candidate (30%) + Engagement (20%) + Topic Coverage (15%) + Clarification (10%) + Visual (25%)

## Quick Start

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** & npm

### 1. Backend Setup

```bash
# From project root
pip install -r requirements.txt

# Optional: Set HuggingFace token for pyannote diarization
set HF_TOKEN=your_token_here    # Windows
# export HF_TOKEN=your_token_here  # Linux/Mac

# Start the API server
python main.py
```

The API will be available at **http://localhost:8000** (Swagger docs at `/docs`).

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The dashboard opens at **http://localhost:5173**.

### 3. Usage

1. **Demo Mode**: Click "📋 Load Demo" to analyze the built-in interview data
2. **File Upload**: Drag & drop an audio file (.wav/.mp3) and optionally a video file
3. **Live Mode**: Use the WebSocket endpoint `/ws/live` for real-time streaming

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/demo` | Run demo analysis on built-in data |
| POST | `/api/analyze/audio` | Upload audio file for analysis |
| POST | `/api/analyze/video` | Upload video file for visual analysis |
| POST | `/api/analyze/full` | Upload audio + video for multimodal analysis |
| GET | `/api/sessions` | List all sessions |
| GET | `/api/results/{id}` | Get results for a session |
| WS | `/ws/live` | WebSocket for live streaming |

## Project Structure

```
online monitor sys/
├── backend/
│   ├── __init__.py
│   ├── config.py              # Central configuration
│   ├── audio_processing.py    # Whisper + diarization
│   ├── nlp_analysis.py        # Sentiment, sarcasm, topics, scoring
│   ├── video_processing.py    # Face, emotion, attention analysis
│   ├── scoring_engine.py      # Multimodal fusion engine
│   └── api.py                 # FastAPI server
├── frontend/
│   ├── src/
│   │   ├── components/        # React dashboard components
│   │   ├── services/api.js    # API client
│   │   ├── App.jsx            # Main app
│   │   ├── main.jsx           # Entry point
│   │   └── index.css          # Design system
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── main.py                    # Entry point (starts server)
├── requirements.txt           # Python dependencies
├── session_analysis.csv       # Sample demo data
├── dashboard.py               # Legacy Streamlit dashboard
└── README.md
```

## Sample Data

The project includes sample interview data:
- `session_analysis.csv` — Pre-analyzed 58-segment mock interview
- `mock-interview (1).wav` — Audio recording for testing

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Audio | OpenAI Whisper, pyannote.audio |
| NLP | TextBlob, scikit-learn |
| Video | OpenCV, DeepFace, MediaPipe |
| Frontend | React 18, Vite, Recharts, Framer Motion |
| Styling | CSS (glassmorphism dark theme) |
