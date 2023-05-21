import cv2 # Video Capture
import time # Time Delayer
import numpy as np # Array Modifier
from keras.models import model_from_json # Deep learning Framework
# Simulate Keyboard and Mouse input
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
keyboard = KeyboardController()
mouse = MouseController()
# Determines how close you need to be, to be detected
min_face_size = 100
# Load the pre-trained model
model = model_from_json(open("/Users/shashishmac/PythonProjects/FacialR/Model/facial_expression_model_structure.json", "r").read())
model.load_weights('/Users/shashishmac/PythonProjects/FacialR/Model/facial_expression_model_weights.h5')

min_face_size = 150


# Define the emotions
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using a Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier('/Users/shashishmac/PythonProjects/FacialR/Model/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each detected face
    for (x, y, w, h) in faces:
        if w < min_face_size or h < min_face_size:
            continue
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) from the grayscale image
        roi_gray = gray[y:y + h, x:x + w]

        # Resize the ROI to match the input size of the model
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype('float32')
        roi /= 255
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Make a prediction using the pre-trained model
        prediction = model.predict(roi)[0]

        # Find the emotion with the highest probability
        max_index = np.argmax(prediction)
        emotion = emotions[max_index]
        if emotion == 'happy':
            keyboard.press(Key.space)
            keyboard.release(Key.space)
            time.sleep(1.5)

        elif emotion == 'surprise':
            keyboard.press(Key.right)
            keyboard.release(Key.right)

        if emotion == 'angry':
            mouse.scroll(0, -10)

        # Display the emotion on the image
        cv2.putText(frame, emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame with detected faces and emotions
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()