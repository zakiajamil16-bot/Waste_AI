
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

# Load the trained AI model
interpreter = Interpreter(
    model_path="waste_classifier_updated.tflite",
    num_threads=1
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Shared system state
esp_triggered = False
latest_result = {
    "ready": False,
    "label": "",
    "confidence": 0
}


@app.route("/")
def index():
    return render_template("index.html")


# ESP32 calls this when its sensor detects waste
@app.route("/esp-trigger", methods=["POST"])
def esp_trigger():
    global esp_triggered, latest_result

    esp_triggered = True
    latest_result = {
        "ready": False,
        "label": "",
        "confidence": 0
    }

    return jsonify({
        "status": "trigger_received"
    }), 200


# Phone browser checks whether the ESP32 detected waste
@app.route("/check-trigger", methods=["GET"])
def check_trigger():
    global esp_triggered

    if esp_triggered:
        esp_triggered = False
        return jsonify({"triggered": True})

    return jsonify({"triggered": False})


# Classify the captured image using the trained model
@app.route("/predict", methods=["POST"])
def predict():
    global latest_result

    if "file" not in request.files:
        return jsonify({"error": "No image sent"}), 400

    file = request.files["file"]

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        # Preprocess image for MobileNetV2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)

        # Run the AI model
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()

        prediction = float(
            interpreter.get_tensor(output_details[0]["index"])[0][0]
        )

        # Convert model output into a class
        if prediction < 0.5:
            label = "BIODEGRADABLE"
            confidence = (1 - prediction) * 100
        else:
            label = "NON-BIODEGRADABLE"
            confidence = prediction * 100

        latest_result = {
            "ready": True,
            "label": label,
            "confidence": round(float(confidence), 2)
        }

        return jsonify(latest_result)

    except Exception as error:
        app.logger.exception("Image prediction failed")
        return jsonify({
            "error": "Prediction failed"
        }), 500


# ESP32 checks this endpoint to retrieve the latest prediction
@app.route("/get-result", methods=["GET"])
def get_result():
    return jsonify(latest_result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

