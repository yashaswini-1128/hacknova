import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt

from video_utils import extract_frames, analyze_frames
from audio_utils import extract_audio, analyze_audio, extract_transcript

st.set_page_config(
    page_title="AI Deepfake Shield",
    page_icon="🛡",
    layout="wide"
)

st.title("🛡 AI Deepfake & Synthetic Media Shield")
st.markdown("### Multi-Modal Synthetic Media Risk Intelligence System")

uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "mov", "avi"])

if uploaded_file is not None:

    # Save uploaded video
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Video uploaded successfully!")
    st.video(file_path)

    # ---------------- VIDEO ANALYSIS ----------------
    st.subheader("🎥 Video Analysis")

    num_frames, frame_paths = extract_frames(file_path)

    if num_frames == 0:
        st.error("No frames extracted from video.")
        st.stop()

    video_score, frame_scores = analyze_frames(frame_paths)

    # ---------------- AUDIO ANALYSIS ----------------
    st.subheader("🎧 Audio Analysis")

    audio_path = extract_audio(file_path)
    audio_score = analyze_audio(audio_path)

    # ---------------- CONTEXT ANALYSIS ----------------
    st.subheader("📝 Context Analysis")

    transcript = extract_transcript(audio_path)

    high_risk_keywords = [
        "transfer money", "urgent", "bank account",
        "otp", "election", "emergency", "confidential"
    ]

    context_risk = 0
    for keyword in high_risk_keywords:
        if keyword in transcript:
            context_risk += 0.1

    context_risk = min(context_risk, 0.3)

    # ---------------- FINAL RISK FUSION ----------------
    final_score = (0.5 * video_score) + (0.3 * audio_score) + (0.2 * context_risk)

    # ---------------- DISPLAY RESULTS ----------------
    st.subheader("📊 Risk Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Video Risk", f"{round(video_score*100,2)}%")

    with col2:
        st.metric("Audio Risk", f"{round(audio_score*100,2)}%")

    with col3:
        st.metric("Context Risk", f"{round(context_risk*100,2)}%")

    with col4:
        st.metric("Final Risk", f"{round(final_score*100,2)}%")

    if final_score < 0.3:
        st.success("Overall Risk: Low")
    elif final_score < 0.6:
        st.warning("Overall Risk: Moderate")
    else:
        st.error("Overall Risk: High")

    # ---------------- FRAME GRAPH ----------------
    st.subheader("📈 Frame-Level Deepfake Probability")

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(frame_scores)), frame_scores)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Fake Probability")

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

    # ---------------- TRANSCRIPT DISPLAY ----------------
    if transcript:
        st.subheader("🗣 Extracted Speech")
        st.write(transcript)