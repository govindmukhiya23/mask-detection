import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "../models/mask_detector.h5"

# Load face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load trained mask detection model
model = load_model(MODEL_PATH)

# -------------------------------------------------
# PREDICT MASK OR NO MASK
# -------------------------------------------------
def predict_mask(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    pred = model.predict(face_img)[0]

    return np.argmax(pred), max(pred)


# -------------------------------------------------
# WEBCAM LIVE DETECTION
# -------------------------------------------------
cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    # Draw bounding boxes & predictions
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        label, confidence = predict_mask(face_img)

        if label == 0:
            text = f"Mask ({confidence*100:.2f}%)"
            color = (0, 255, 0)
        else:
            text = f"No Mask ({confidence*100:.2f}%)"
            color = (0, 0, 255)

        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    cv2.imshow("Mask Detector", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam closed.")
