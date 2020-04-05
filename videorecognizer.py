import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep

def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./training"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("training/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

def unknown_image_encoded(img):
    face = fr.load_image_file("training/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding

cap = cv2.VideoCapture(0)

faces = get_encoded_faces()
faces_encoded = list(faces.values())
known_face_names = list(faces.keys())

while(True):
    ret, frame = cap.read()
    cv2.imwrite("./master/frame.jpg",frame)
    img = cv2.imread("./master/frame.jpg", 1)

    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    appreaces = []
    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
    
        appreaces.append(name)
        face_names.append(name)
    
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
            cv2.rectangle(frame, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left -20, bottom + 15), font, .8, (255, 255, 255), 2)

    cv2.imshow('frame',frame)
    print(appreaces)

    if cv2.waitKey(29) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

