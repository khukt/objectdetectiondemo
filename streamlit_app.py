import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import av
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import cv2

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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.knn_classifier = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (224, 224))
        img_resized = np.expand_dims(img_resized, axis=0)
        img_resized = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
        
        # Extract features
        features = model.predict(img_resized)
        features = features.flatten()

        # Predict with KNN classifier if it exists
        if self.knn_classifier:
            prediction = self.knn_classifier.predict([features])
            label = f"Prediction: {prediction[0]}"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            label = "No classifier trained yet."
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Teachable Machine with Streamlit")

    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

    label = st.text_input("Enter label for training data:")
    if st.button('Add Training Data'):
        if webrtc_ctx.video_transformer and label:
            video_transformer = webrtc_ctx.video_transformer
            img = webrtc_ctx.video_frame.to_ndarray(format="bgr24")
            img_resized = cv2.resize(img, (224, 224))
            img_resized = np.expand_dims(img_resized, axis=0)
            img_resized = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
            
            features = model.predict(img_resized)
            features = features.flatten()

            training_data.add_data(features, label)
            st.success("Training data added!")

    if st.button('Train Classifier'):
        if webrtc_ctx.video_transformer and training_data.features:
            video_transformer = webrtc_ctx.video_transformer
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(training_data.features, training_data.labels)
            video_transformer.knn_classifier = knn
            st.success("Classifier trained!")

if __name__ == "__main__":
    main()
