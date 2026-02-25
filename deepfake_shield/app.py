import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt

from video_utils import extract_frames, analyze_frames
from audio_utils import extract_audio, analyze_audio, extract_transcript

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Deepfake Shield",
    page_icon="🛡",
    layout="wide"
)

# ---------------- CUSTOM DARK THEME ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3, h4 {
    color: white;
}
.stMetric {
    background-color: #1C1F26;
    padding: 15px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    font-size: 26px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align: center; color: #00BFFF;'>
🛡 AI Deepfake & Synthetic Media Shield
</h1>
<p style='text-align: center; color: #BBBBBB;'>
Multi-Modal Synthetic Media Risk Intelligence Platform
</p>
""", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "mov", "avi"])

if uploaded_file is not None:

    # Save uploaded file
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Video uploaded successfully!")
    st.video(file_path)

    with st.spinner("Analyzing Synthetic Media..."):

        # -------- VIDEO ANALYSIS --------
        num_frames, frame_paths = extract_frames(file_path)

        if num_frames == 0:
            st.error("No frames extracted.")
            st.stop()

        video_score, frame_scores = analyze_frames(frame_paths)

        # -------- AUDIO ANALYSIS --------
        audio_path = extract_audio(file_path)
        audio_score = analyze_audio(audio_path)

        # -------- CONTEXT ANALYSIS --------
        transcript = extract_transcript(audio_path)

        high_risk_keywords = [
            "transfer money", "urgent", "bank account",
            "otp", "election", "emergency", "confidential"
        ]

        context_risk = 0
        for keyword in high_risk_keywords:
            if transcript and keyword in transcript:
                context_risk += 0.1

        context_risk = min(context_risk, 0.3)

        # -------- FINAL FUSION --------
        final_score = (0.5 * video_score) + (0.3 * audio_score) + (0.2 * context_risk)

    # ---------------- DASHBOARD ----------------
    st.subheader("📊 Risk Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Video Risk", f"{round(video_score*100,2)}%")
        st.metric("Audio Risk", f"{round(audio_score*100,2)}%")

    with col2:
        st.metric("Context Risk", f"{round(context_risk*100,2)}%")
        st.metric("Final Risk", f"{round(final_score*100,2)}%")

    # -------- RISK BADGE --------
    if final_score < 0.3:
        risk_color = "#00FF99"
        risk_label = "LOW RISK"
    elif final_score < 0.6:
        risk_color = "#FFA500"
        risk_label = "MODERATE RISK"
    else:
        risk_color = "#FF4B4B"
        risk_label = "HIGH RISK"

    st.markdown(f"""
    <div style='padding:20px; border-radius:12px; background-color:#1C1F26; text-align:center;'>
        <h2 style='color:{risk_color};'>Overall Risk: {risk_label}</h2>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- GRAPH ----------------
    st.subheader("📈 Frame-Level Deepfake Probability")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(np.arange(len(frame_scores)), frame_scores, linewidth=2)
    ax.set_facecolor("#1C1F26")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Fake Probability")
    ax.set_title("Frame-Level Deepfake Probability")

    st.pyplot(fig)

    # ---------------- SUSPICIOUS FRAMES ----------------
    st.subheader("🚨 Top Suspicious Frames")

    threshold = 0.7
    shown = 0

    for i, score in enumerate(frame_scores):
        if score > threshold and shown < 5:
            st.image(frame_paths[i], caption=f"Frame {i} | Score: {round(score,2)}")
            shown += 1

    if shown == 0:
        st.info("No highly suspicious frames detected.")

    # ---------------- TRANSCRIPT ----------------
    if transcript:
        st.subheader("🗣 Extracted Speech")
        st.write(transcript)

    st.markdown("""
    <hr style="border:1px solid #333;">
    <p style='text-align:center; color:#888;'>
    AI Deepfake Shield | Hackathon 2026
    </p>
    """, unsafe_allow_html=True)