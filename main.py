import cv2
import numpy as np
from picamera2 import Picamera2

# Initialize the Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={'size': (640, 480)}))
picam2.start()

# Load the Haar Cascade once
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

while True:
    # Capture frame from the camera
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Resize the frame
    img = cv2.resize(frame, (500, 500))

    # Copy for results
    result = img.copy()

    # Detect faces
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Stack original and result image vertically
    stacked_image = np.vstack((img, result))

    # Show the result
    cv2.imshow("Face Detection", stacked_image)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.close()
