import librosa
import numpy as np
import random
import moviepy
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_audio(video_path, output_audio="temp_audio.wav"):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio, verbose=False, logger=None)
        return output_audio
    except:
        return None


def analyze_audio(audio_path):
    try:
        if audio_path is None:
            return 0.0

        y, sr = librosa.load(audio_path, duration=10)

        # Extract features (for real ML later)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Hackathon simulation score
        audio_score = random.uniform(0.2, 0.9)

        return audio_score

    except:
        return 0.0

import speech_recognition as sr

def extract_transcript(audio_path):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text.lower()
    except:
        return ""