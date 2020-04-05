import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import time

tempPath = 'temp'
allowedFormat = ["jpg","jpeg","png"]

def get_face(pathName):
    face = fr.load_image_file(pathName)
    return fr.face_encodings(face)

def gather():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./" + tempPath):
        for f in fnames:
            ext = f.split(".")[1]
            if ext.lower() in allowedFormat:
                start = time.time()
                
                encoding = get_face(tempPath + "/" + f)            
                encoded[f.split(".")[0]] = encoding
                end = time.time() - start
                
                print ("%s [%6.2f seconds]" % (f,end))

    return encoded

def recognize(): 
    images = gather()
    
    for im in images: 
        face_locations = face_recognition.face_locations(img)
        unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

timeStart = time.time()
listFace = recognize()
timeEnd = time.time() - timeStart
print("Total processing time : %6.2f Seconds" % (timeEnd))
#print(listFace)