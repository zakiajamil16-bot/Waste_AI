
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

interpreter = Interpreter(
    model_path="waste_classifier.tflite",
    num_threads=1
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app_state = {
    "capture_requested": False,
    "latest_result": None
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/esp-trigger", methods=["POST"])
def esp_trigger():
    app_state["capture_requested"] = True
    app_state["latest_result"] = None

    return jsonify({
        "status": "success",
        "message": "Capture requested"
    })


@app.route("/check-status", methods=["GET"])
def check_status():
    return jsonify({
        "capture_requested": app_state["capture_requested"]
    })


@app.route("/get-result", methods=["GET"])
def get_result():
    if app_state["latest_result"] is not None:
        return jsonify({
            "ready": True,
            "result": app_state["latest_result"]
        })

    return jsonify({"ready": False})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = (img / 127.5) - 1.0

    interpreter.set_tensor(input_details[0]["index"], img)
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

    result_data = {
        "prediction": label,
        "confidence": round(float(confidence), 2)
    }

    app_state["latest_result"] = result_data
    app_state["capture_requested"] = False

    return jsonify(result_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

