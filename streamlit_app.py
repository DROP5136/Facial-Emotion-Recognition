import streamlit as st
import cv2
from fer import FER
import numpy as np
from PIL import Image
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🧠 Facial Emotion Recognition (FER) with FER Library")

detector = FER(mtcnn=True)
emotion_colors = {
    "angry": "#E74C3C", "disgust": "#2ECC71", "fear": "#3498DB",
    "happy": "#F1C40F", "sad": "#9B59B6", "surprise": "#E67E22", "neutral": "#95A5A6"
}

uploaded_file = st.file_uploader("Upload an image or take a photo", type=["jpg", "jpeg", "png"])
# camera_input = st.camera_input("Take a selfie (Optional)")  # Optional camera

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    result = detector.detect_emotions(image_np)

    if result:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        emotion_sequence = []
        confidence_sequence = []

        for face in result:
            (x, y, w, h) = face["box"]
            top_emotion = max(face["emotions"], key=face["emotions"].get)
            confidence = face["emotions"][top_emotion]

            emotion_sequence.append(top_emotion)
            confidence_sequence.append(confidence * 100)

        # Create DataFrame
        df = pd.DataFrame({
            "Emotion": emotion_sequence,
            "Confidence (%)": confidence_sequence,
            "Face": list(range(1, len(emotion_sequence) + 1))
        })

        st.subheader("Detected Emotions")
        st.dataframe(df)

        # Plot confidence
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x="Emotion", y="Confidence (%)", data=df, palette=emotion_colors)
        plt.title("Emotion Confidence per Face")
        st.pyplot(fig)

    else:
        st.warning("No face or emotion detected.")
else:
    st.info("Please upload an image to begin.")
