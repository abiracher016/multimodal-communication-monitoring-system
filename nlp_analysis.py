"""
NLP Analysis Module
- Sentiment analysis
- Interaction classification
- Topic detection & coverage scoring
- Sarcasm detection (text + context patterns)
- Clarification tracking
- Candidate evaluation metrics (communication, confidence, positivity, depth, relevance)
"""
import re
import math
from collections import Counter

from textblob import TextBlob

from backend.config import (
    TOPIC_KEYWORDS, HESITATION_WORDS, SARCASM_PATTERNS,
    CONFUSION_PHRASES, EXPECTED_TOPICS, CANDIDATE_SCORE_WEIGHTS,
)


# ─── Sentiment Analysis ──────────────────────────────────────────────────────
def analyze_sentiment(text: str) -> dict:
    """Return sentiment polarity, subjectivity, and label."""
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "sentiment_score": polarity,
        "subjectivity": subjectivity,
        "sentiment": label,
    }


# ─── Interaction Classification ──────────────────────────────────────────────
def classify_interaction(text: str, speaker: str) -> str:
    """Classify as Question, Answer, or Prompt."""
    text_lower = text.lower().strip()

    if "?" in text:
        return "Question" if speaker == "Interviewer" else "Answer"

    question_starters = ["tell me", "could you", "can you", "what", "how", "why", "describe", "explain"]
    if speaker == "Interviewer" and any(text_lower.startswith(q) for q in question_starters):
        return "Question"

    if speaker == "Interviewer":
        return "Prompt"

    return "Answer"


# ─── Topic Detection ─────────────────────────────────────────────────────────
def detect_topic(text: str) -> str:
    """Detect the primary topic of a text segment."""
    text_lower = text.lower()
    topic_scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            topic_scores[topic] = score

    if topic_scores:
        return max(topic_scores, key=topic_scores.get)
    return "General"


def calculate_topic_coverage(segments: list[dict]) -> dict:
    """
    Calculate which expected topics were covered and which are missing.
    Returns { covered: [...], missing: [...], coverage_pct: float, per_topic: {...} }
    """
    detected_topics = set()
    topic_counts = Counter()

    for seg in segments:
        topic = seg.get("topic", "General")
        if topic != "General":
            detected_topics.add(topic)
            topic_counts[topic] += 1

    covered = [t for t in EXPECTED_TOPICS if t in detected_topics]
    missing = [t for t in EXPECTED_TOPICS if t not in detected_topics]
    coverage_pct = round((len(covered) / len(EXPECTED_TOPICS)) * 100, 2) if EXPECTED_TOPICS else 0

    per_topic = {}
    for topic in EXPECTED_TOPICS:
        per_topic[topic] = {
            "covered": topic in detected_topics,
            "mentions": topic_counts.get(topic, 0),
        }

    return {
        "covered": covered,
        "missing": missing,
        "coverage_pct": coverage_pct,
        "per_topic": per_topic,
    }


# ─── Sarcasm Detection ───────────────────────────────────────────────────────
def detect_sarcasm(text: str, sentiment_score: float, visual_emotion: str = None) -> dict:
    """
    Detect sarcasm using:
    1. Text-sentiment mismatch (positive words + negative sentiment or vice versa)
    2. Known sarcasm context patterns
    3. Visual emotion mismatch (if available)
    """
    text_lower = text.lower()
    signals = []
    is_sarcastic = False

    # Check known sarcasm patterns
    for pattern in SARCASM_PATTERNS:
        if pattern in text_lower:
            signals.append(f"sarcasm_pattern: '{pattern}'")
            is_sarcastic = True
            break

    # Sentiment mismatch: text sounds positive but sentiment is negative (or vice versa)
    positive_words = ["great", "amazing", "wonderful", "fantastic", "love", "perfect", "excellent", "awesome"]
    negative_words = ["terrible", "awful", "horrible", "worst", "hate", "disgusting"]

    has_positive_words = any(w in text_lower for w in positive_words)
    has_negative_words = any(w in text_lower for w in negative_words)

    if has_positive_words and sentiment_score < -0.1:
        signals.append("positive_words_negative_sentiment")
        is_sarcastic = True
    if has_negative_words and sentiment_score > 0.1:
        signals.append("negative_words_positive_sentiment")
        is_sarcastic = True

    # Visual emotion mismatch (if video data available)
    if visual_emotion:
        if has_positive_words and visual_emotion in ["angry", "sad", "disgust", "fear"]:
            signals.append(f"visual_mismatch: text positive but face shows {visual_emotion}")
            is_sarcastic = True
        if sentiment_score > 0.2 and visual_emotion in ["angry", "sad", "disgust"]:
            signals.append(f"sentiment_visual_mismatch: positive text, {visual_emotion} face")
            is_sarcastic = True

    return {
        "is_sarcastic": is_sarcastic,
        "confidence": min(1.0, len(signals) * 0.4) if signals else 0.0,
        "signals": signals,
    }


# ─── Clarification Tracking ──────────────────────────────────────────────────
def track_clarifications(segments: list[dict]) -> list[dict]:
    """
    Detect confusion phrases and track if they were resolved later.
    Returns list of clarification events.
    """
    clarifications = []

    for i, seg in enumerate(segments):
        text_lower = seg.get("text", "").lower()

        for phrase in CONFUSION_PHRASES:
            if phrase in text_lower:
                # Look for resolution in next 3 segments
                resolved = False
                resolution_text = ""
                for j in range(i + 1, min(i + 4, len(segments))):
                    next_text = segments[j].get("text", "").lower()
                    resolution_indicators = [
                        "i see", "makes sense", "got it", "understand",
                        "okay", "oh", "right", "ah", "thanks",
                        "that explains", "clear now", "i get it"
                    ]
                    if any(ri in next_text for ri in resolution_indicators):
                        resolved = True
                        resolution_text = segments[j].get("text", "")
                        break

                clarifications.append({
                    "segment_index": i,
                    "speaker": seg.get("speaker", "Unknown"),
                    "confusion_text": seg.get("text", ""),
                    "confusion_phrase": phrase,
                    "resolved": resolved,
                    "resolution_text": resolution_text,
                    "timestamp": seg.get("start", 0),
                })
                break  # one detection per segment

    return clarifications


# ─── Candidate Evaluation Metrics ────────────────────────────────────────────
def calculate_communication_score(candidate_segments: list[dict]) -> float:
    """Score based on average response length."""
    if not candidate_segments:
        return 0.0
    avg_length = sum(len(seg.get("text", "")) for seg in candidate_segments) / len(candidate_segments)
    return round(min(100, (avg_length / 150) * 100), 2)


def calculate_confidence_score(candidate_segments: list[dict]) -> float:
    """Score based on absence of hesitation words."""
    if not candidate_segments:
        return 0.0
    hesitation_count = 0
    for seg in candidate_segments:
        text_lower = seg.get("text", "").lower()
        hesitation_count += sum(1 for w in HESITATION_WORDS if w in text_lower)
    score = max(0, 100 - hesitation_count * 5)
    return round(score, 2)


def calculate_positivity_score(candidate_segments: list[dict]) -> float:
    """Score based on average sentiment."""
    if not candidate_segments:
        return 0.0
    avg_sentiment = sum(seg.get("sentiment_score", 0) for seg in candidate_segments) / len(candidate_segments)
    score = max(0, min(100, ((avg_sentiment + 1) / 2) * 100))
    return round(score, 2)


def calculate_response_depth_score(candidate_segments: list[dict]) -> float:
    """Score based on response detail — word count and unique words."""
    if not candidate_segments:
        return 0.0
    total_words = sum(len(seg.get("text", "").split()) for seg in candidate_segments)
    avg_words = total_words / len(candidate_segments)
    # Also factor in vocabulary diversity
    all_words = " ".join(seg.get("text", "") for seg in candidate_segments).lower().split()
    diversity = len(set(all_words)) / len(all_words) if all_words else 0
    depth = min(100, (avg_words / 30) * 60 + diversity * 40)
    return round(depth, 2)


def calculate_relevance_score(candidate_segments: list[dict], interviewer_segments: list[dict]) -> float:
    """
    Score based on overlap between candidate answers and interviewer questions.
    Uses simple word overlap (TF-IDF-like approach).
    """
    if not candidate_segments or not interviewer_segments:
        return 50.0  # neutral default

    # Extract question keywords
    question_words = set()
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
                  "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
                  "you", "your", "me", "my", "i", "we", "our", "it", "so", "that",
                  "this", "can", "could", "would", "tell", "about", "have", "has",
                  "been", "be", "not", "just", "also", "very", "more", "some", "any"}

    for seg in interviewer_segments:
        words = re.findall(r'\b\w+\b', seg.get("text", "").lower())
        question_words.update(w for w in words if w not in stop_words and len(w) > 2)

    if not question_words:
        return 50.0

    # Calculate overlap
    total_overlap = 0
    for seg in candidate_segments:
        words = set(re.findall(r'\b\w+\b', seg.get("text", "").lower()))
        overlap = len(words & question_words)
        total_overlap += overlap

    relevance = min(100, (total_overlap / (len(question_words) * 0.5)) * 100)
    return round(relevance, 2)


def calculate_candidate_scores(segments: list[dict]) -> dict:
    """Calculate all candidate evaluation metrics."""
    candidate_segs = [s for s in segments if s.get("speaker") == "Candidate"]
    interviewer_segs = [s for s in segments if s.get("speaker") == "Interviewer"]

    communication = calculate_communication_score(candidate_segs)
    confidence = calculate_confidence_score(candidate_segs)
    positivity = calculate_positivity_score(candidate_segs)
    depth = calculate_response_depth_score(candidate_segs)
    relevance = calculate_relevance_score(candidate_segs, interviewer_segs)

    w = CANDIDATE_SCORE_WEIGHTS
    final = round(
        communication * w["communication"] +
        positivity * w["positivity"] +
        confidence * w["confidence"] +
        depth * w["response_depth"],
        2
    )

    return {
        "communication_score": communication,
        "confidence_score": confidence,
        "positivity_score": positivity,
        "response_depth_score": depth,
        "relevance_score": relevance,
        "final_candidate_score": final,
    }


def calculate_engagement_score(segments: list[dict]) -> float:
    """Engagement based on question-answer ratio."""
    questions = sum(1 for s in segments if s.get("interaction") in ("Question", "Prompt"))
    answers = sum(1 for s in segments if s.get("interaction") == "Answer")
    if questions == 0:
        return 0.0
    return round(min(100, (answers / questions) * 100), 2)


# ─── Full NLP Pipeline ───────────────────────────────────────────────────────
def analyze_segments(segments: list[dict], video_emotions: dict = None) -> dict:
    """
    Run complete NLP analysis pipeline on transcribed segments.
    Returns enriched segments + aggregate metrics.
    """
    # Enrich each segment
    for i, seg in enumerate(segments):
        text = seg.get("text", "")

        # Sentiment
        sentiment_data = analyze_sentiment(text)
        seg.update(sentiment_data)

        # Interaction type
        seg["interaction"] = classify_interaction(text, seg.get("speaker", "Unknown"))

        # Topic
        seg["topic"] = detect_topic(text)

        # Sarcasm (with optional visual emotion)
        visual_emotion = None
        if video_emotions and i in video_emotions:
            visual_emotion = video_emotions[i]
        sarcasm = detect_sarcasm(text, seg.get("sentiment_score", 0), visual_emotion)
        seg["sarcasm"] = sarcasm

    # Aggregate metrics
    scores = calculate_candidate_scores(segments)
    engagement = calculate_engagement_score(segments)
    topic_coverage = calculate_topic_coverage(segments)
    clarifications = track_clarifications(segments)

    # Clarification success rate
    if clarifications:
        resolved_count = sum(1 for c in clarifications if c["resolved"])
        clarification_success = round((resolved_count / len(clarifications)) * 100, 2)
    else:
        clarification_success = 100.0  # no confusion = perfect

    # Sarcasm events
    sarcasm_events = [
        {
            "index": i,
            "text": seg["text"],
            "speaker": seg.get("speaker", ""),
            "signals": seg["sarcasm"]["signals"],
            "confidence": seg["sarcasm"]["confidence"],
        }
        for i, seg in enumerate(segments)
        if seg.get("sarcasm", {}).get("is_sarcastic")
    ]

    return {
        "segments": segments,
        "scores": scores,
        "engagement_score": engagement,
        "topic_coverage": topic_coverage,
        "clarifications": clarifications,
        "clarification_success": clarification_success,
        "sarcasm_events": sarcasm_events,
    }
