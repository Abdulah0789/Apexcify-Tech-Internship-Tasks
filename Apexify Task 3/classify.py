import cv2
import numpy as np
from keras.models import load_model

# Load model and labels
model = load_model("keras_model.h5", compile=False)
labels = open("labels.txt", "r").readlines()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    img = (img / 127.5) - 1  # Normalize

    prediction = model.predict(img)
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence = prediction[0][index]

    cv2.putText(frame, f"{class_name} ({confidence:.2f})",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)

    cv2.imshow("Teachable Machine Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
