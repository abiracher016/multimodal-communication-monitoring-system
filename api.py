"""
FastAPI Backend Server
- REST endpoints for file-upload analysis
- WebSocket for live streaming
- Session management
- CORS enabled
"""
import os
import json
import uuid
import asyncio
import traceback
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd

from backend.config import CORS_ORIGINS, UPLOAD_DIR, RESULTS_DIR, SAMPLE_RATE, CHUNK_DURATION
from backend.audio_processing import process_audio_file, transcribe_audio_chunk, record_audio_chunk
from backend.nlp_analysis import analyze_segments
from backend.video_processing import process_video
from backend.scoring_engine import build_full_report

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Session Monitor API",
    description="Multimodal AI-powered live session monitoring system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (for demo; swap with DB in production)
sessions: dict[str, dict] = {}


# ─── Helper ──────────────────────────────────────────────────────────────────
def save_upload(file: UploadFile, session_id: str, suffix: str) -> str:
    """Save uploaded file and return its path."""
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    file_path = os.path.join(session_dir, f"upload{suffix}")
    
    # Reset file pointer to handle multiple reads of the same file stream
    file.file.seek(0)
    content = file.file.read()
    
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path


def save_results(session_id: str, report: dict):
    """Persist results as JSON and CSV."""
    result_dir = os.path.join(RESULTS_DIR, session_id)
    os.makedirs(result_dir, exist_ok=True)

    # JSON full report
    with open(os.path.join(result_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    # CSV of segments
    if report.get("segments"):
        df = pd.DataFrame(report["segments"])
        csv_cols = [c for c in df.columns if c != "sarcasm"]
        df[csv_cols].to_csv(os.path.join(result_dir, "session_analysis.csv"), index=False)


# ─── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "AI Session Monitor API is running", "version": "1.0.0"}


@app.get("/api/sessions")
async def list_sessions():
    """List all analysis sessions."""
    session_list = []
    for sid, data in sessions.items():
        session_list.append({
            "session_id": sid,
            "created_at": data.get("created_at"),
            "status": data.get("status"),
            "composite_score": data.get("report", {}).get("intelligence_score", {}).get("composite_score"),
        })
    return {"sessions": session_list}


@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    """Get results for a specific session."""
    if session_id not in sessions:
        # Try loading from disk
        result_path = os.path.join(RESULTS_DIR, session_id, "report.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                report = json.load(f)
            return {"session_id": session_id, "report": report}
        raise HTTPException(404, f"Session {session_id} not found")
    return {"session_id": session_id, "report": sessions[session_id].get("report", {})}


@app.post("/api/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    """Upload and analyze an audio file."""
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {"created_at": datetime.now().isoformat(), "status": "processing"}

    try:
        # Determine file extension
        ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        audio_path = save_upload(file, session_id, ext)

        # Process audio
        segments = process_audio_file(audio_path)

        # NLP analysis
        nlp_results = analyze_segments(segments)

        # Build report (no video)
        report = build_full_report(nlp_results, video_metrics=None)
        report["session_id"] = session_id

        # Save
        sessions[session_id]["status"] = "completed"
        sessions[session_id]["report"] = report
        save_results(session_id, report)

        return {"session_id": session_id, "report": report}

    except Exception as e:
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.post("/api/analyze/video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    """Upload and analyze a video file (video-only metrics)."""
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {"created_at": datetime.now().isoformat(), "status": "processing"}

    try:
        ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
        video_path = save_upload(file, session_id, ext)

        video_metrics = process_video(video_path)

        report = {
            "session_id": session_id,
            "video_metrics": video_metrics,
        }

        sessions[session_id]["status"] = "completed"
        sessions[session_id]["report"] = report
        save_results(session_id, report)

        return {"session_id": session_id, "report": report}

    except Exception as e:
        sessions[session_id]["status"] = "error"
        traceback.print_exc()
        raise HTTPException(500, f"Video analysis failed: {str(e)}")


@app.post("/api/analyze/full")
async def analyze_full(
    audio: UploadFile = File(...),
    video: Optional[UploadFile] = File(None),
):
    """Full multimodal analysis — audio + optional video."""
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {"created_at": datetime.now().isoformat(), "status": "processing"}

    try:
        # Save audio
        audio_ext = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
        audio_path = save_upload(audio, session_id, audio_ext)

        # Process audio
        segments = process_audio_file(audio_path)

        # Process video if provided
        video_metrics = None
        if video:
            video_ext = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
            video_path = save_upload(video, session_id, video_ext)
            video_metrics = process_video(video_path)

        # NLP analysis (pass video emotion map for sarcasm detection)
        video_emotions = {}
        if video_metrics and video_metrics.get("frame_results"):
            for fr in video_metrics["frame_results"]:
                if "emotion" in fr:
                    # Map frame time to closest segment index
                    frame_time = fr.get("time", 0)
                    for i, seg in enumerate(segments):
                        if seg["start"] <= frame_time <= seg["end"]:
                            video_emotions[i] = fr["emotion"]
                            break

        nlp_results = analyze_segments(segments, video_emotions=video_emotions)

        # Build report
        report = build_full_report(nlp_results, video_metrics)
        report["session_id"] = session_id

        sessions[session_id]["status"] = "completed"
        sessions[session_id]["report"] = report
        save_results(session_id, report)

        return {"session_id": session_id, "report": report}

    except Exception as e:
        sessions[session_id]["status"] = "error"
        traceback.print_exc()
        raise HTTPException(500, f"Full analysis failed: {str(e)}")


# ─── WebSocket for Live Streaming ─────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    WebSocket endpoint for live audio streaming.
    Client sends audio chunks (base64-encoded), server returns incremental analysis.
    """
    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    all_segments = []

    try:
        await ws.send_json({"type": "connected", "session_id": session_id})

        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "audio_chunk":
                # Decode base64 audio
                import base64
                import numpy as np

                audio_b64 = data.get("audio", "")
                if not audio_b64:
                    continue

                audio_bytes = base64.b64decode(audio_b64)
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

                # Transcribe chunk
                text = transcribe_audio_chunk(audio_array, SAMPLE_RATE)
                if not text:
                    continue

                # Create segment
                seg = {
                    "start": len(all_segments) * CHUNK_DURATION,
                    "end": (len(all_segments) + 1) * CHUNK_DURATION,
                    "text": text,
                    "speaker": "Speaker",
                    "speaker_raw": "SPEAKER_00",
                    "timestamp": datetime.now().isoformat(),
                }
                all_segments.append(seg)

                # Run NLP on all segments so far
                nlp_results = analyze_segments(list(all_segments))
                report = build_full_report(nlp_results)
                report["session_id"] = session_id

                # Send update
                await ws.send_json({
                    "type": "update",
                    "segment": seg,
                    "report": report,
                })

            elif msg_type == "stop":
                # Final report
                if all_segments:
                    nlp_results = analyze_segments(list(all_segments))
                    report = build_full_report(nlp_results)
                    report["session_id"] = session_id

                    sessions[session_id] = {
                        "created_at": datetime.now().isoformat(),
                        "status": "completed",
                        "report": report,
                    }
                    save_results(session_id, report)

                    await ws.send_json({"type": "final", "report": report})

                await ws.close()
                break

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {session_id}")
    except Exception as e:
        traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ─── Demo endpoint — analyze existing CSV ────────────────────────────────────

@app.get("/api/demo")
async def demo_analysis():
    """Run analysis on the existing session_analysis.csv for demo purposes."""
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "session_analysis.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(404, "session_analysis.csv not found")

    df = pd.read_csv(csv_path)
    segments = df.to_dict("records")

    # Ensure required fields
    for seg in segments:
        seg.setdefault("start", seg.get("segment_start", 0))
        seg.setdefault("end", seg.get("segment_end", 0))

    nlp_results = analyze_segments(segments)
    report = build_full_report(nlp_results)
    report["session_id"] = "demo"

    return {"session_id": "demo", "report": report}
