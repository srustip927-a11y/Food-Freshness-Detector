import cv2
from PIL import Image
from transformers import pipeline

print("Loading Food-101 model...")

food_model = pipeline(
    "image-classification",
    model="aspis/swin-finetuned-food101"
)

print("Food-101 model loaded successfully!")
print("Opening camera...")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera could not be opened")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    # Convert OpenCV image to PIL image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)

    # Food prediction
    results = food_model(image)

    best = results[0]

    label = best["label"]
    confidence = best["score"] * 100

    text = f"{label}: {confidence:.1f}%"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("Food Recognition - Food 101", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()