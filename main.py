import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# Initialize the Raspberry Pi camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
raw_capture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warm up
time.sleep(0.1)

# Define the lower and upper boundaries of the "yellow" color in HSV color space
# You may need to adjust these values depending on lighting conditions
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    # Grab the raw NumPy array representing the image
    image = frame.array
    
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply a series of erosions and dilations to remove small blobs
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Find contours in the mask
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Handle different OpenCV versions
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    # Only proceed if at least one contour was found
    if len(contours) > 0:
        # Find the largest contour in the mask
        c = max(contours, key=cv2.contourArea)
        
        # Compute the minimum enclosing circle and centroid
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        
        # Only proceed if the radius meets a minimum size
        if radius > 10:
            # Draw the circle and centroid on the frame
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # Display the coordinates
            text = "x: {:.1f}, y: {:.1f}".format(x, y)
            cv2.putText(image, text, (int(x) - 40, int(y) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Show the frame
    cv2.imshow("Frame", image)
    cv2.imshow("Mask", mask)
    
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)
    
    # If the 'q' key is pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()