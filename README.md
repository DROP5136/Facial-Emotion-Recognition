# Facial Emotion Recognition System

A real-time facial emotion recognition system that captures video through a webcam, detects human faces, and classifies emotions such as **Happy**, **Sad**, **Angry**, **Surprised**, **Fear**, **Disgust**, and **Neutral**. The system displays the dominant emotion along with a percentage confidence score.

## Features

- Real-time facial emotion detection using a webcam.
- Displays detected emotion with percentage confidence.
- Visualizes emotion trends over time using Seaborn and Matplotlib.
- Clean and color-coded output for better clarity.
- Uses pre-trained deep learning models through the `FER` library.

## Emotions Detected

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral

## Tech Stack

- Python  
- OpenCV  
- FER (Facial Expression Recognition)  
- Matplotlib  
- Seaborn  
- Pandas  

## Demo

### Real-time Emotion Detection

![Webcam Detection](sample_output/demo_image.jpg)

### Emotion Confidence Plot

![Emotion Plot](sample_output/graph_plot.png)

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/facial-emotion-recognition.git
cd facial-emotion-recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the script
python emotion_recognition.py

# 4. Quit the program by pressing 'q' in the webcam window
