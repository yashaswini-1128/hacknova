import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt

from video_utils import extract_frames, analyze_frames
from audio_utils import extract_audio, analyze_audio, extract_transcript

st.title("🛡 AI Deepfake & Synthetic Media Shield")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:

    # Create uploads folder
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Video uploaded successfully!")
    st.video(file_path)

    # Extract frames
    st.write("DEBUG — Frame paths created?")
    st.write(frame_paths)
    try:
      num_frames, frame_paths = extract_frames(file_path)
    except Exception as e:
      st.error(f"Frame extraction failed: {e}")
    frame_paths = []
    num_frames = 0
    st.success(f"Extracted {num_frames} frames successfully!")

    # Analyze frames
    video_score, frame_scores = analyze_frames(frame_paths)

    st.subheader("🔍 Deepfake Risk Analysis")
    st.write(f"Video Deepfake Probability: {round(video_score * 100, 2)}%")

    if video_score < 0.3:
        st.success("Risk Level: Low")
    elif video_score < 0.6:
        st.warning("Risk Level: Moderate")
    else:
        st.error("Risk Level: High")

    # Graph
    st.subheader("📊 Frame-Level Deepfake Probability")

    frame_numbers = np.arange(len(frame_scores))

    fig, ax = plt.subplots()
    ax.plot(frame_numbers, frame_scores)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Fake Probability")
    ax.set_title("Deepfake Probability Across Frames")

    st.pyplot(fig)

    # Suspicious frames
    st.subheader("🚨 Highly Suspicious Frames")

    threshold = 0.7
    suspicious_found = False

    for i, score in enumerate(frame_scores[:5]):  # limit to 5 frames
        if score > threshold:
            st.image(frame_paths[i], caption=f"Frame {i} - Score: {round(score,2)}")
            suspicious_found = True

    if not suspicious_found:
        st.write("No highly suspicious frames detected.")

        # ---------------- AUDIO ANALYSIS ----------------
    st.subheader("🎧 Audio Deepfake Analysis")

    audio_path = extract_audio(file_path)
    audio_score = analyze_audio(audio_path)

    st.write(f"Audio Deepfake Probability: {round(audio_score * 100, 2)}%")

    if audio_score < 0.3:
        st.success("Audio Risk: Low")
    elif audio_score < 0.6:
        st.warning("Audio Risk: Moderate")
    else:
        st.error("Audio Risk: High")

    # ---------------- FINAL MULTI-MODAL SCORE ----------------
    st.subheader("🧠 Final Multi-Modal Risk Intelligence")

    final_score = (0.5 * video_score) + (0.3 * audio_score) + (0.2 *context_risk)

    st.write(f"Final Deepfake Risk Score: {round(final_score * 100, 2)}%")

    if final_score < 0.3:
        st.success("Overall Risk: Low")
    elif final_score < 0.6:
        st.warning("Overall Risk: Moderate")
    else:
        st.error("Overall Risk: High")

       # ---------------- CONTEXT ANALYSIS ----------------
    st.subheader("📝 Context Risk Analysis")

    transcript = extract_transcript(audio_path)

    if transcript:
        st.write("Transcript:")
        st.write(transcript)
    else:
        st.write("No clear speech detected.")

    # Define high-risk keywords
    high_risk_keywords = [
        "transfer money",
        "urgent",
        "bank account",
        "otp",
        "election",
        "emergency",
        "confidential"
    ]

    context_risk = 0

    for keyword in high_risk_keywords:
        if keyword in transcript:
            context_risk += 0.1

    context_risk = min(context_risk, 0.3)

    st.write(f"Context Risk Score: {round(context_risk * 100, 2)}%")

    @st.cache_resource
    def load_video_model():
      from video_utils import train_dummy_video_model
      return train_dummy_video_model()