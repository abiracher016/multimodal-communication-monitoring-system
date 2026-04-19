import os
from datetime import datetime

import pandas as pd
import whisper
from textblob import TextBlob

from speaker_detection import diarize_audio

AUDIO_FILE = "mock-interview.wav"
OUTPUT_CSV = "session_analysis.csv"

# paste your Hugging Face token here temporarily for testing
HF_TOKEN = "YOUR_HF_TOKEN_HERE"


def classify_interaction(text: str, speaker: str) -> str:
    t = text.lower().strip()

    if "?" in text:
        return "Question" if speaker == "Interviewer" else "Answer"

    if speaker == "Interviewer":
        return "Prompt"

    return "Answer"


def detect_topic(text: str) -> str:
    t = text.lower()

    if any(k in t for k in ["project", "built", "developed", "system", "application"]):
        return "Projects"
    elif any(k in t for k in ["internship", "worked", "company", "experience", "job", "role"]):
        return "Experience"
    elif any(k in t for k in ["study", "studied", "college", "university", "degree", "education"]):
        return "Education"
    elif any(k in t for k in ["goal", "future", "career", "long term", "down the line"]):
        return "Career Goals"
    elif any(k in t for k in ["skill", "python", "machine learning", "data", "analysis", "technical"]):
        return "Skills"
    else:
        return "General"


def match_speaker(seg_start: float, seg_end: float, diarization_segments: list[dict]) -> str:
    mid = (seg_start + seg_end) / 2

    for sp in diarization_segments:
        if sp["start"] <= mid <= sp["end"]:
            return sp["speaker"]

    return "UNKNOWN"


def map_roles(df: pd.DataFrame) -> dict:
    speaker_counts = df["speaker_raw"].value_counts().to_dict()
    speakers_sorted = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)

    role_map = {}
    if len(speakers_sorted) >= 1:
        role_map[speakers_sorted[0][0]] = "Candidate"
    if len(speakers_sorted) >= 2:
        role_map[speakers_sorted[1][0]] = "Interviewer"

    return role_map


def calculate_candidate_scores(df: pd.DataFrame) -> dict:
    candidate_df = df[df["speaker"] == "Candidate"].copy()

    if candidate_df.empty:
        return {
            "communication_score": 0.0,
            "positivity_score": 0.0,
            "confidence_score": 0.0,
            "response_depth_score": 0.0,
            "final_candidate_score": 0.0
        }

    avg_length = candidate_df["text"].astype(str).apply(len).mean()
    communication_score = min(100, (avg_length / 150) * 100)

    avg_sentiment = candidate_df["sentiment_score"].mean()
    positivity_score = max(0, min(100, ((avg_sentiment + 1) / 2) * 100))

    hesitation_words = ["um", "uh", "maybe", "i think", "sort of", "kind of", "probably"]
    hesitation_count = candidate_df["text"].astype(str).str.lower().apply(
        lambda x: sum(word in x for word in hesitation_words)
    ).sum()
    confidence_score = max(0, 100 - hesitation_count * 10)

    response_depth_score = min(100, len(candidate_df) * 10)

    final_candidate_score = round(
        (communication_score * 0.30) +
        (positivity_score * 0.25) +
        (confidence_score * 0.25) +
        (response_depth_score * 0.20),
        2
    )

    return {
        "communication_score": round(communication_score, 2),
        "positivity_score": round(positivity_score, 2),
        "confidence_score": round(confidence_score, 2),
        "response_depth_score": round(response_depth_score, 2),
        "final_candidate_score": final_candidate_score
    }


def calculate_engagement_score(df: pd.DataFrame) -> float:
    question_count = len(df[df["interaction"].isin(["Question", "Prompt"])])
    answer_count = len(df[df["interaction"] == "Answer"])

    if question_count == 0:
        return 0.0

    return round(min(100, (answer_count / question_count) * 100), 2)


if not os.path.exists(AUDIO_FILE):
    print(f"Audio file not found: {AUDIO_FILE}")
    raise SystemExit

print("Running diarization...")
diarization_segments = diarize_audio(AUDIO_FILE, HF_TOKEN)

print("Running Whisper transcription...")
model = whisper.load_model("base")
result = model.transcribe(AUDIO_FILE)
segments = result.get("segments", [])

rows = []

for seg in segments:
    text = seg.get("text", "").strip()
    if not text:
        continue

    seg_start = round(seg.get("start", 0), 2)
    seg_end = round(seg.get("end", 0), 2)

    speaker_raw = match_speaker(seg_start, seg_end, diarization_segments)

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    rows.append({
        "timestamp": datetime.now(),
        "segment_start": seg_start,
        "segment_end": seg_end,
        "speaker_raw": speaker_raw,
        "text": text,
        "sentiment_score": round(polarity, 4),
        "sentiment": sentiment
    })

df = pd.DataFrame(rows)

role_map = map_roles(df)
df["speaker"] = df["speaker_raw"].map(role_map).fillna(df["speaker_raw"])
df["interaction"] = df.apply(lambda r: classify_interaction(r["text"], r["speaker"]), axis=1)
df["topic"] = df["text"].apply(detect_topic)

scores = calculate_candidate_scores(df)
engagement_score = calculate_engagement_score(df)

for key, value in scores.items():
    df[key] = value

df["engagement_score"] = engagement_score

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved to {OUTPUT_CSV}")
print("Role map:", role_map)
print("Candidate score:", scores["final_candidate_score"])
print("Engagement score:", engagement_score)