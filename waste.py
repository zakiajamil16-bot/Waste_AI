import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

# Load lightweight LiteRT model
interpreter = Interpreter(
    model_path="waste_classifier.tflite",
    num_threads=1
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # OpenCV uses BGR, but the model was trained with RGB images
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))

    # Convert to the format expected by the model
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    # MobileNetV2 preprocessing: 0-255 -> -1 to 1
    img = (img / 127.5) - 1.0

    # Run LiteRT inference
    interpreter.set_tensor(
        input_details[0]["index"],
        img
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )[0][0]

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
