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
        self.current_frame = None
        self.capturing = False
        self.capture_label = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.current_frame = img

        # Resize and preprocess the image
        img_resized = cv2.resize(img, (224, 224))
        img_resized = np.expand_dims(img_resized, axis=0)
        img_resized = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
        
        # Extract features using MobileNet
        features = model.predict(img_resized)
        features = features.flatten()

        # Capture features if capturing is enabled
        if self.capturing and self.capture_label:
            training_data.add_data(features, self.capture_label)

        # Predict with KNN classifier if it exists
        if self.knn_classifier:
            prediction = self.knn_classifier.predict([features])
            label = f"Prediction: {prediction[0]}"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Teachable Machine with Streamlit")

    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.video_transformer:
        video_transformer = webrtc_ctx.video_transformer
        label = st.text_input("Enter label for training data:")
        
        if st.button('Start Capturing'):
            if webrtc_ctx.state.playing and label:
                video_transformer.capturing = True
                video_transformer.capture_label = label
                st.success("Started capturing images for label: " + label)

        if st.button('Stop Capturing'):
            if webrtc_ctx.state.playing:
                video_transformer.capturing = False
                video_transformer.capture_label = None
                st.success("Stopped capturing images.")

        if st.button('Train Classifier'):
            if training_data.features and training_data.labels:
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(training_data.features, training_data.labels)
                video_transformer.knn_classifier = knn
                st.success("Classifier trained!")

if __name__ == "__main__":
    main()
