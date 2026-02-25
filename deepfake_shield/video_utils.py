import cv2
import os
import random

def extract_frames(video_path, output_folder="frames", frame_skip=30):
    if os.path.exists(output_folder):
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))
    else:
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    frame_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count, frame_paths


def analyze_frames(frame_paths):
    if len(frame_paths) == 0:
        return 0.0, []

    model = train_dummy_video_model()

    features = extract_video_features(frame_paths)

    if len(features) == 0:
        return 0.0, []

    probabilities = model.predict_proba(features)[:, 1]
    video_score = np.mean(probabilities)

    return video_score, probabilities

import numpy as np
from sklearn.ensemble import RandomForestClassifier

def extract_video_features(frame_paths):
    features = []

    for path in frame_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        mean = np.mean(img)
        std = np.std(img)
        features.append([mean, std])

    return np.array(features)

def train_dummy_video_model():
    # Create synthetic training data
    X = np.random.rand(200, 2)
    y = np.random.randint(0, 2, 200)

    model = RandomForestClassifier()
    model.fit(X, y)

    return model