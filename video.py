import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(29) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

