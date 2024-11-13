import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import Image
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define the alphabet (labels for the classes)

alphabet =  ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y','Z']# Replace with actual labels

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Streamlit app title
st.title("American Sign Language Detection")

# Webcam input or image upload
option = st.selectbox("Select Input Method", ("Webcam", "Upload Image"))

def preprocess_image(image):
    image = image.resize((64, 64))
    image_array = np.array(image)
    if image_array.shape[-1] != 3:
        image_array = np.stack([image_array] * 3, axis=-1)
    image_array = image_array.astype('float32') / 255.0
    return np.expand_dims(image_array, axis=0)

# Preprocess landmarks to use only x and y coordinates (first 42 values)
def process_frame(image):
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                # Select only x and y coordinates
                landmark_list = [landmark.x for landmark in hand_landmarks.landmark] + \
                                [landmark.y for landmark in hand_landmarks.landmark]
                df = pd.DataFrame([landmark_list])
                predictions = model.predict(df, verbose=0)
                predicted_class = np.argmax(predictions, axis=1)[0]
                label = alphabet[predicted_class]
                cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        return image

if option == "Webcam":
    st.write("Open the webcam to detect ASL.")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Error: Could not read frame from webcam.")
            break
        frame = process_frame(frame)
        FRAME_WINDOW.image(frame, channels="BGR")
else:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        st.write(f"Predicted ASL Character: {alphabet[predicted_class]}")
