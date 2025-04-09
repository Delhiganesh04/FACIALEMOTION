import serial
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np

# Initialize serial communication with Arduino
arduino = serial.Serial('COM7', 9600)  # Replace 'COM3' with the correct port

# Load the pre-trained emotion detection model
classifier = load_model('Emotion_vgg.h5')
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Send the detected emotion to Arduino
            if label == 'Happy':
                arduino.write(b'H')  # Send 'H' for Happy
            elif label == 'Sad':
                arduino.write(b'S')  # Send 'S' for Sad
            elif label == 'Angry':
                arduino.write(b'A')  # Send 'A' for Angry
            elif label == 'Surprise':
                arduino.write(b'U')  # Send 'U' for Surprise
            else:
                arduino.write(b'N')  # Send 'N' for Neutral
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
