import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import os

# ==========================================
# 1. DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. LOAD YOLO FOOD DETECTION MODEL
# ==========================================

yolo_model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully.")

# ==========================================
# 3. LOAD FRESHNESS MODEL
# ==========================================

freshness_model = models.resnet18(weights=None)

freshness_model.fc = nn.Linear(
    freshness_model.fc.in_features,
    2
)

# Model is in the main project folder
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

print("Freshness model loaded successfully.")

# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 5. OPEN CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()

print("Camera opened successfully.")
print("Show food in front of the camera.")
print("Press Q to quit.")

# ==========================================
# 6. CAMERA LOOP
# ==========================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read camera frame.")
        break

    # --------------------------------------
    # YOLO detection
    # --------------------------------------

    results = yolo_model(
        frame,
        verbose=False
    )

    for result in results:

        boxes = result.boxes

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < 0.50:
                continue

            class_id = int(box.cls[0])

            food_name = yolo_model.names[class_id]

            # --------------------------------------
            # Get bounding box
            # --------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # --------------------------------------
            # Crop detected food
            # --------------------------------------

            food_crop = frame[
                y1:y2,
                x1:x2
            ]

            if food_crop.size == 0:
                continue

            # --------------------------------------
            # Freshness prediction
            # --------------------------------------

            rgb_crop = cv2.cvtColor(
                food_crop,
                cv2.COLOR_BGR2RGB
            )

            pil_image = Image.fromarray(
                rgb_crop
            )

            input_tensor = transform(
                pil_image
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

            # --------------------------------------
            # Display text
            # --------------------------------------

            text = (
                f"{food_name}: "
                f"{freshness_label.upper()} "
                f"{freshness_percentage:.1f}%"
            )

            # --------------------------------------
            # Draw bounding box
            # --------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # --------------------------------------
            # Draw result
            # --------------------------------------

            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # ==========================================
    # SHOW CAMERA
    # ==========================================

    cv2.imshow(
        "Smart Food Freshness Detector",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# 7. CLOSE CAMERA
# ==========================================

camera.release()

cv2.destroyAllWindows()

print("Program stopped.")