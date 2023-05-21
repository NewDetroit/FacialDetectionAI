import cv2 # Video Capture
import time # Time Delayer
import numpy as np # Array Modifier
from keras.models import model_from_json # Deep learning Framework
from mtcnn import MTCNN # Face Detection AKA MultiTask Cascaded Convolutional Neural Network
# Simulate Keyboard and Mouse input
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
keyboard = KeyboardController()
mouse = MouseController()
# Determines how close you need to be, to be detected
min_face_size = 150

# Load the model
model = model_from_json(open("/FacialR/Model/facial_expression_model_structure.json", "r").read())
model.load_weights('/FacialR/Model/facial_expression_model_weights.h5')

# Define the emotions
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Initialize MTCNN for face detection
detector = MTCNN()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Resize the frame to improve face detection performance
    frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the resized frame to grayscale
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using MTCNN
    faces = detector.detect_faces(frame_resized)

    # For each detected face
    for face in faces:
        x, y, w, h = face['box']
        # Only using people who are close up


        # Draw a rectangle around the face
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) from the grayscale image
        roi_gray = gray[y:y + h, x:x + w]

        # Resize the ROI to match the input size of the model
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype('float32')
        roi /= 255
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Make a prediction using the model
        prediction = model.predict(roi)[0]

        # Find the emotion with the highest probability
        max_index = np.argmax(prediction)
        emotion = emotions[max_index]

        # Keyboard Commands based on Facial cues
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
        cv2.putText(frame_resized, emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame with detected faces and emotions
    cv2.imshow('Emotion Detection', frame_resized)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
