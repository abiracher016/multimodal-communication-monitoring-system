"""
Scoring Engine — Multimodal Fusion
Combines audio, text, and video insights into a unified intelligence score.
"""
from backend.config import COMPOSITE_SCORE_WEIGHTS


def compute_visual_engagement_score(video_metrics: dict) -> float:
    """
    Compute visual engagement from video metrics.
    Considers: attention %, emotion positivity, head stability, face presence.
    """
    if not video_metrics or video_metrics.get("status") != "success":
        return 50.0  # neutral default when no video

    attention = video_metrics.get("attention_percentage", 0)
    stability = video_metrics.get("head_stability", 0)
    face_present = min(100, video_metrics.get("avg_faces_detected", 0) * 100)

    # Emotion positivity: happy/neutral are positive, angry/sad/fear are negative
    emotion_dist = video_metrics.get("emotion_distribution", {})
    positive_emotions = emotion_dist.get("happy", 0) + emotion_dist.get("neutral", 0) + emotion_dist.get("surprise", 0)
    negative_emotions = emotion_dist.get("angry", 0) + emotion_dist.get("sad", 0) + emotion_dist.get("fear", 0) + emotion_dist.get("disgust", 0)
    emotion_positivity = max(0, min(100, positive_emotions - negative_emotions * 0.5 + 50))

    visual_score = (
        attention * 0.35 +
        emotion_positivity * 0.25 +
        stability * 0.20 +
        face_present * 0.20
    )

    return round(min(100, max(0, visual_score)), 2)


def compute_conversation_intelligence_score(
    candidate_score: float,
    engagement_score: float,
    topic_coverage_pct: float,
    clarification_success: float,
    visual_engagement: float,
) -> dict:
    """
    Compute the composite Conversation Intelligence Score.
    Returns breakdown + composite.
    """
    w = COMPOSITE_SCORE_WEIGHTS

    components = {
        "candidate_score": {
            "value": round(candidate_score, 2),
            "weight": w["candidate_score"],
            "weighted": round(candidate_score * w["candidate_score"], 2),
        },
        "engagement_score": {
            "value": round(engagement_score, 2),
            "weight": w["engagement_score"],
            "weighted": round(engagement_score * w["engagement_score"], 2),
        },
        "topic_coverage": {
            "value": round(topic_coverage_pct, 2),
            "weight": w["topic_coverage"],
            "weighted": round(topic_coverage_pct * w["topic_coverage"], 2),
        },
        "clarification_success": {
            "value": round(clarification_success, 2),
            "weight": w["clarification_success"],
            "weighted": round(clarification_success * w["clarification_success"], 2),
        },
        "visual_engagement": {
            "value": round(visual_engagement, 2),
            "weight": w["visual_engagement"],
            "weighted": round(visual_engagement * w["visual_engagement"], 2),
        },
    }

    composite = sum(c["weighted"] for c in components.values())
    composite = round(min(100, max(0, composite)), 2)

    # Grade
    if composite >= 85:
        grade = "A"
        label = "Excellent"
    elif composite >= 70:
        grade = "B"
        label = "Good"
    elif composite >= 55:
        grade = "C"
        label = "Average"
    elif composite >= 40:
        grade = "D"
        label = "Below Average"
    else:
        grade = "F"
        label = "Needs Improvement"

    return {
        "composite_score": composite,
        "grade": grade,
        "label": label,
        "components": components,
    }


def build_full_report(nlp_results: dict, video_metrics: dict = None) -> dict:
    """
    Build the full analysis report combining NLP and video results.
    This is the final output structure sent to the frontend.
    """
    scores = nlp_results.get("scores", {})
    engagement = nlp_results.get("engagement_score", 0)
    coverage = nlp_results.get("topic_coverage", {})
    coverage_pct = coverage.get("coverage_pct", 0)
    clarification_success = nlp_results.get("clarification_success", 100)

    # Visual engagement
    visual_engagement = compute_visual_engagement_score(video_metrics)

    # Composite score
    intelligence = compute_conversation_intelligence_score(
        candidate_score=scores.get("final_candidate_score", 0),
        engagement_score=engagement,
        topic_coverage_pct=coverage_pct,
        clarification_success=clarification_success,
        visual_engagement=visual_engagement,
    )

    return {
        "session_id": None,  # filled by API layer
        "segments": nlp_results.get("segments", []),
        "candidate_scores": scores,
        "engagement_score": engagement,
        "topic_coverage": coverage,
        "clarifications": nlp_results.get("clarifications", []),
        "clarification_success": clarification_success,
        "sarcasm_events": nlp_results.get("sarcasm_events", []),
        "video_metrics": video_metrics or {},
        "visual_engagement_score": visual_engagement,
        "intelligence_score": intelligence,
    }
