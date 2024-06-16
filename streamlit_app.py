import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import av
import tensorflow as tf
import numpy as np

# Load the MobileNet model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.knn_classifier = None
        self.features = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = tf.image.resize(img, (224, 224))
        img_resized = np.expand_dims(img_resized, axis=0)
        img_resized = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
        
        # Extract features
        features = model.predict(img_resized)
        features = features.flatten()

        # Predict with KNN classifier if it exists
        if self.knn_classifier:
            prediction = self.knn_classifier.predict([features])
            label = f"Prediction: {prediction[0]}"
            st.write(label)
        else:
            label = "No classifier trained yet."
        
        # Return the original frame with the label
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Teachable Machine with Streamlit")
    
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

    if st.button('Train Classifier'):
        if webrtc_ctx.video_transformer:
            # Example: Train a simple KNN classifier with random data
            from sklearn.neighbors import KNeighborsClassifier
            video_transformer = webrtc_ctx.video_transformer
            # Assuming we collect features and labels somehow
            features = np.random.rand(10, 1280)  # Dummy features
            labels = np.random.choice(['class1', 'class2'], 10)  # Dummy labels
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(features, labels)
            video_transformer.knn_classifier = knn
            st.success("Classifier trained!")

if __name__ == "__main__":
    main()
