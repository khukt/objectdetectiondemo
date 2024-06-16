let net;
let classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');

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
                    webcamElement.addEventListener('loadeddata', () => resolve(), false);
                },
                error => reject());
        } else {
            reject();
        }
    });
}

async function init() {
    console.log('Loading mobilenet..');
    net = await mobilenet.load();
    console.log('Successfully loaded model');

    await setupWebcam();
}

async function train() {
    const activation = net.infer(webcamElement, 'conv_preds');
    classifier.addExample(activation, prompt('Enter label:'));
}

async function predict() {
    const activation = net.infer(webcamElement, 'conv_preds');
    const result = await classifier.predictClass(activation);
    alert(`Prediction: ${result.label}`);
}
