import streamlit as st

st.title("Teachable Machine with Streamlit and TensorFlow.js")
st.write("Upload an image to classify it using a pre-trained MobileNet model with TensorFlow.js.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Embed JavaScript code to use TensorFlow.js
    st.write("""
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet"></script>
        <script>
            async function classifyImage() {
                const img = document.getElementById('uploaded-image');
                const model = await mobilenet.load();
                const predictions = await model.classify(img);
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '';
                predictions.forEach(prediction => {
                    const p = document.createElement('p');
                    p.innerText = `${prediction.className}: ${prediction.probability.toFixed(4)}`;
                    resultDiv.appendChild(p);
                });
            }
            
            document.addEventListener('DOMContentLoaded', () => {
                const img = document.getElementById('uploaded-image');
                if (img) {
                    classifyImage();
                }
            });
        </script>
        <img id="uploaded-image" src="data:image/jpeg;base64,{}" style="display: none;" onload="classifyImage()"/>
        <div id="result"></div>
    """.format(uploaded_file.read().decode("utf-8")), unsafe_allow_html=True)
