import streamlit as st
import streamlit.components.v1 as components

st.title("Teachable Machine with Streamlit")
st.write("This demo uses TensorFlow.js to create a KNN classifier for webcam images.")

html_code = """
<!DOCTYPE html>
<html>
  <head>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/knn-classifier"></script>
  </head>
  <body>
    <h2>Webcam Image Classification</h2>
    <video id="webcam" width="224" height="224" autoplay></video>
    <br>
    <button id="add-example">Add Example</button>
    <button id="predict">Predict</button>
    <div id="result"></div>
    <script>
      const webcamElement = document.getElementById('webcam');
      const classifier = knnClassifier.create();
      let net;

      async function setupWebcam() {
        return new Promise((resolve, reject) => {
          const navigatorAny = navigator;
          navigator.getUserMedia = navigator.getUserMedia ||
              navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
              navigatorAny.msGetUserMedia;
          if (navigator.getUserMedia) {
            navigator.getUserMedia(
                {video: true},
                stream => {
                  webcamElement.srcObject = stream;
                  webcamElement.addEventListener('loadeddata',  () => resolve(), false);
                },
                error => reject());
          } else {
            reject();
          }
        });
      }

      async function app() {
        console.log('Loading mobilenet..');

        // Load the model.
        net = await mobilenet.load();
        console.log('Sucessfully loaded model');

        await setupWebcam();

        // Add example
        const addExample = async classId => {
          const activation = net.infer(webcamElement, true);
          classifier.addExample(activation, classId);
        };

        // Predict
        const predict = async () => {
          if (classifier.getNumClasses() > 0) {
            const activation = net.infer(webcamElement, 'conv_preds');
            const result = await classifier.predictClass(activation);
            document.getElementById('result').innerText = `
              Prediction: ${result.label}\n
              Probability: ${result.confidences[result.label]}
            `;
          }
        };

        document.getElementById('add-example').addEventListener('click', () => addExample(0));
        document.getElementById('predict').addEventListener('click', () => predict());
      }

      app();
    </script>
  </body>
</html>
"""

components.html(html_code, height=600)
