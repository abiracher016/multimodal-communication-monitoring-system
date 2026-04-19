import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI Interview Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1020, #111827, #1f2937);
        color: white;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        animation: glow 2.5s ease-in-out infinite alternate;
    }

    .subtitle {
        font-size: 16px;
        color: #cbd5e1;
        margin-bottom: 25px;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 6px rgba(96,165,250,0.3)); }
        to { filter: drop-shadow(0 0 16px rgba(167,139,250,0.6)); }
    }

    .metric-card {
        padding: 20px;
        border-radius: 20px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        backdrop-filter: blur(10px);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.35);
    }

    .metric-title {
        font-size: 14px;
        color: #cbd5e1;
        margin-bottom: 10px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: white;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 18px;
        margin-bottom: 14px;
    }

    .insight-box {
        padding: 16px;
        border-radius: 16px;
        background: rgba(255,255,255,0.05);
        border-left: 5px solid #60a5fa;
        color: #e5e7eb;
        margin-bottom: 12px;
    }

    .positive-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(34,197,94,0.18);
        color: #4ade80;
        font-size: 12px;
        font-weight: 700;
    }

    .negative-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(239,68,68,0.18);
        color: #f87171;
        font-size: 12px;
        font-weight: 700;
    }

    .neutral-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(234,179,8,0.18);
        color: #facc15;
        font-size: 12px;
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD DATA
# ----------------------------
try:
    data = pd.read_csv("session_analysis.csv")
except Exception:
    st.error("❌ Run transcribe.py first to generate session_analysis.csv")
    st.stop()

required_cols = [
    "timestamp", "speaker", "interaction", "topic", "text",
    "sentiment_score", "sentiment",
    "communication_score", "positivity_score",
    "confidence_score", "response_depth_score",
    "final_candidate_score", "engagement_score"
]

missing = [c for c in required_cols if c not in data.columns]
if missing:
    st.error(f"❌ Missing columns in CSV: {missing}")
    st.stop()

# ----------------------------
# HEADER
# ----------------------------
st.markdown('<div class="main-title">🎥 AI Interview Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time style analytics for interviews and live interactive sessions</div>', unsafe_allow_html=True)

# ----------------------------
# TOP METRICS
# ----------------------------
candidate_score = float(data["final_candidate_score"].iloc[0])
engagement_score = float(data["engagement_score"].iloc[0])
avg_sentiment = round(data["sentiment_score"].mean(), 2)
total_turns = len(data)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🎯 Candidate Score</div>
        <div class="metric-value">{candidate_score}/100</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🤝 Engagement Score</div>
        <div class="metric-value">{engagement_score}%</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">😊 Avg Sentiment</div>
        <div class="metric-value">{avg_sentiment}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🗣 Total Turns</div>
        <div class="metric-value">{total_turns}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# SCORE BREAKDOWN
# ----------------------------
st.markdown('<div class="section-title">📊 Candidate Performance Breakdown</div>', unsafe_allow_html=True)

score_df = pd.DataFrame({
    "Metric": [
        "Communication",
        "Positivity",
        "Confidence",
        "Response Depth"
    ],
    "Score": [
        float(data["communication_score"].iloc[0]),
        float(data["positivity_score"].iloc[0]),
        float(data["confidence_score"].iloc[0]),
        float(data["response_depth_score"].iloc[0])
    ]
})

c1, c2 = st.columns([1, 1.4])

with c1:
    st.dataframe(score_df, use_container_width=True, hide_index=True)

with c2:
    fig_breakdown = px.bar(
        score_df,
        x="Metric",
        y="Score",
        color="Metric",
        text="Score",
        title="Performance Components"
    )
    fig_breakdown.update_traces(textposition="outside")
    fig_breakdown.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font_color="white",
        showlegend=False,
        title_font_size=20
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

# ----------------------------
# SESSION DATA TABLE
# ----------------------------
st.markdown('<div class="section-title">📋 Session Data</div>', unsafe_allow_html=True)

display_df = data[[
    "timestamp", "segment_start", "segment_end", "speaker",
    "interaction", "topic", "text", "sentiment_score", "sentiment"
]].copy()

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ----------------------------
# CHARTS
# ----------------------------
left, right = st.columns(2)

with left:
    fig_sent = px.line(
        data,
        x=data.index,
        y="sentiment_score",
        markers=True,
        title="📈 Sentiment Score by Segment"
    )
    fig_sent.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font_color="white",
        xaxis_title="Segment Index",
        yaxis_title="Sentiment Score"
    )
    st.plotly_chart(fig_sent, use_container_width=True)

with right:
    speaker_counts = data["speaker"].value_counts().reset_index()
    speaker_counts.columns = ["Speaker", "Count"]

    fig_speaker = px.pie(
        speaker_counts,
        names="Speaker",
        values="Count",
        title="👥 Speaker Distribution",
        hole=0.45
    )
    fig_speaker.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig_speaker, use_container_width=True)

# ----------------------------
# INTERACTION + TOPIC
# ----------------------------
l2, r2 = st.columns(2)

with l2:
    interaction_counts = data["interaction"].value_counts().reset_index()
    interaction_counts.columns = ["Interaction", "Count"]

    fig_interaction = px.bar(
        interaction_counts,
        x="Interaction",
        y="Count",
        color="Interaction",
        title="❓ Interaction Type Distribution",
        text="Count"
    )
    fig_interaction.update_traces(textposition="outside")
    fig_interaction.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font_color="white",
        showlegend=False
    )
    st.plotly_chart(fig_interaction, use_container_width=True)

with r2:
    topic_counts = data["topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]

    fig_topic = px.bar(
        topic_counts,
        x="Topic",
        y="Count",
        color="Topic",
        title="📚 Topic Distribution",
        text="Count"
    )
    fig_topic.update_traces(textposition="outside")
    fig_topic.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font_color="white",
        showlegend=False
    )
    st.plotly_chart(fig_topic, use_container_width=True)

# ----------------------------
# SPEAKER-WISE SENTIMENT
# ----------------------------
st.markdown('<div class="section-title">🧠 Speaker-wise Sentiment</div>', unsafe_allow_html=True)

speaker_sentiment = data.groupby("speaker")["sentiment_score"].mean().reset_index()

s1, s2 = st.columns([1, 1.2])

with s1:
    st.dataframe(speaker_sentiment, use_container_width=True, hide_index=True)

with s2:
    fig_sw = px.bar(
        speaker_sentiment,
        x="speaker",
        y="sentiment_score",
        color="speaker",
        text="sentiment_score",
        title="Average Sentiment by Speaker"
    )
    fig_sw.update_traces(textposition="outside")
    fig_sw.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font_color="white",
        showlegend=False
    )
    st.plotly_chart(fig_sw, use_container_width=True)

# ----------------------------
# CONVERSATION FEED
# ----------------------------
st.markdown('<div class="section-title">💬 Conversation Feed</div>', unsafe_allow_html=True)

for _, row in data.iterrows():
    if row["sentiment"] == "Positive":
        pill = '<span class="positive-pill">Positive</span>'
    elif row["sentiment"] == "Negative":
        pill = '<span class="negative-pill">Negative</span>'
    else:
        pill = '<span class="neutral-pill">Neutral</span>'

    st.markdown(
        f"""
        <div class="insight-box">
            <b>{row['speaker']}</b> • {row['interaction']} • {row['topic']} &nbsp; {pill}
            <br><br>
            {row['text']}
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# FINAL INSIGHTS
# ----------------------------
st.markdown('<div class="section-title">📝 Final Insights</div>', unsafe_allow_html=True)

positive_count = (data["sentiment"] == "Positive").sum()
negative_count = (data["sentiment"] == "Negative").sum()
neutral_count = (data["sentiment"] == "Neutral").sum()

interviewer_turns = (data["speaker"] == "Interviewer").sum()
candidate_turns = (data["speaker"] == "Candidate").sum()

i1, i2 = st.columns(2)

with i1:
    st.markdown(f"""
    <div class="insight-box">
        <b>Sentiment Summary</b><br>
        Positive segments: <b>{positive_count}</b><br>
        Negative segments: <b>{negative_count}</b><br>
        Neutral segments: <b>{neutral_count}</b>
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown(f"""
    <div class="insight-box">
        <b>Participation Summary</b><br>
        Interviewer turns: <b>{interviewer_turns}</b><br>
        Candidate turns: <b>{candidate_turns}</b>
    </div>
    """, unsafe_allow_html=True)

if candidate_score >= 75:
    st.success("✅ Candidate performance appears strong.")
elif candidate_score >= 50:
    st.warning("⚠️ Candidate performance appears moderate.")
else:
    st.error("❌ Candidate performance appears weak.")

if engagement_score >= 80:
    st.success("✅ Interview engagement is high.")
elif engagement_score >= 50:
    st.warning("⚠️ Interview engagement is moderate.")
else:
    st.error("❌ Interview engagement is low.")