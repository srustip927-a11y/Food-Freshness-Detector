from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import io
import numpy as np
import cv2

app = Flask(__name__)

# -----------------------------
# Load YOLO model
# -----------------------------

yolo_model = YOLO("yolo11n.pt")

# -----------------------------
# Load freshness model
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

freshness_model = models.resnet18(weights=None)

freshness_model.fc = nn.Linear(
    freshness_model.fc.in_features,
    2
)

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "freshness_model.pth"
)

checkpoint = torch.load(
    model_path,
    map_location=device
)

freshness_model.load_state_dict(
    checkpoint["model"]
)

classes = checkpoint["classes"]

freshness_model = freshness_model.to(device)
freshness_model.eval()

# -----------------------------
# Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Home page
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction API
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image received"
        }), 400

    file = request.files["image"]

    try:
        image = Image.open(
            io.BytesIO(file.read())
        ).convert("RGB")

    except Exception:
        return jsonify({
            "error": "Invalid image"
        }), 400

    # Convert PIL image to OpenCV format
    frame = np.array(image)

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    # -----------------------------
    # YOLO detection
    # -----------------------------

    results = yolo_model(
        frame,
        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < 0.50:
                continue

            class_id = int(box.cls[0])

            food_name = yolo_model.names[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            food_crop = frame[
                y1:y2,
                x1:x2
            ]

            if food_crop.size == 0:
                continue

            # -----------------------------
            # Freshness prediction
            # -----------------------------

            rgb_crop = cv2.cvtColor(
                food_crop,
                cv2.COLOR_BGR2RGB
            )

            food_image = Image.fromarray(
                rgb_crop
            )

            input_tensor = transform(
                food_image
            )

            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():

                output = freshness_model(
                    input_tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                freshness_confidence, predicted = torch.max(
                    probabilities,
                    1
                )

            freshness_label = classes[
                predicted.item()
            ]

            freshness_percentage = (
                freshness_confidence.item() * 100
            )

            detections.append({
                "food": food_name,
                "freshness": freshness_label,
                "confidence": round(
                    freshness_percentage,
                    2
                )
            })

    return jsonify({
        "detections": detections
    })


# -----------------------------
# Start Flask server
# -----------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )