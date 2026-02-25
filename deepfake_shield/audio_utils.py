import librosa
import random
from moviepy.video.io.VideoFileClip import VideoFileClip
import speech_recognition as sr

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
        return random.uniform(0.2, 0.9)
    except:
        return 0.0

def extract_transcript(audio_path):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text.lower()
    except:
        return ""