import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier

# Load the MobileNet model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)

# Define a class to hold training data
class TrainingData:
    def __init__(self):
        self.features = []
        self.labels = []

    def add_data(self, features, label):
        self.features.append(features)
        self.labels.append(label)

# Initialize the training data holder
training_data = TrainingData()

# Define a function to capture frames from the webcam
def capture_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error("Failed to capture image.")
        return None
    return frame

# Define a function to preprocess images and extract features
def preprocess_and_extract_features(image):
    img_resized = cv2.resize(image, (224, 224))
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
    features = model.predict(img_resized)
    return features.flatten()

def main():
    st.title("Teachable Machine with Streamlit")

    label = st.text_input("Enter label for training data:")
    if st.button('Capture Image'):
        if not label:
            st.error("Label not provided.")
        else:
            frame = capture_frame()
            if frame is not None:
                features = preprocess_and_extract_features(frame)
                training_data.add_data(features, label)
                st.image(frame, channels="BGR")
                st.success("Captured image and added training data for label: " + label)

    if st.button('Train Classifier'):
        if training_data.features and training_data.labels:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(training_data.features, training_data.labels)
            st.session_state.knn_classifier = knn
            st.success("Classifier trained!")

    if 'knn_classifier' in st.session_state:
        st.header("Real-Time Prediction")
        frame = capture_frame()
        if frame is not None:
            features = preprocess_and_extract_features(frame)
            knn_classifier = st.session_state.knn_classifier
            prediction = knn_classifier.predict([features])
            label = f"Prediction: {prediction[0]}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            st.image(frame, channels="BGR")
        else:
            st.error("Failed to capture image for prediction.")

if __name__ == "__main__":
    main()
