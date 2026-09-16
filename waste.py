import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model = tf.keras.models.load_model("waste_classifier.keras")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]

    if prediction < 0.5:
        label = "BIODEGRADABLE"
        confidence = (1 - prediction) * 100
    else:
        label = "NON-BIODEGRADABLE"
        confidence = prediction * 100

    return jsonify({
        "prediction": label,
        "confidence": round(float(confidence), 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)