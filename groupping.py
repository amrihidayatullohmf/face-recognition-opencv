import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import time
from PIL import Image
import PIL, sys

masterPath = 'master'
resultPath = 'results'
tempPath = 'temp'

masterCounter = 0
masterImages = {}
allowedFormat = ["jpg","jpeg","png"]

def gather():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./" + tempPath):
        for f in fnames:
            ext = f.split(".")[1]
            if ext.lower() in allowedFormat:
                encoded[f] = tempPath + "/" + f

    return encoded

def get_encoded_faces(excludeName):
    encoded = {}
    
    for dirpath, dnames, fnames in os.walk("./"+masterPath):
        for f in fnames:
            if f == excludeName:
                continue
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file(masterPath+"/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

def unknown_image_encoded(img):
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding

def get_image(pathName):
    face = fr.load_image_file(pathName)
    return fr.face_locations(face)


def recognizeIt(pathName,fileName):
    timeStart = time.time()
    faces = get_encoded_faces(fileName)
    names = fileName.split('.')

    if len(faces) == 0:
        newName = names[0]+'_1.'+names[1]
        os.rename(r'./'+masterPath+'/'+fileName,r'./'+masterPath+'/'+newName)
        timeEnd = time.time() - timeStart

        print('Identifying %s in %2.3f seconds' % (fileName,timeEnd))

        return True

    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())
    img = cv2.imread(pathName, 1)

    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        #newName = names[0]+'_1.'+names[1]

        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            newNames = known_face_names[best_match_index].split('_')[1]
            countName = 0
            for nm in known_face_names: 
                checkName = nm.split("_")[1]
                if checkName == newNames: 
                    countName += 1
        
            newName = "master_"+newNames+"_"+str(countName)+"."+names[1]
            break

    os.rename(r'./'+masterPath+'/'+fileName,r'./'+masterPath+'/'+newName)
    timeEnd = time.time() - timeStart

    print('Identifying %s in %2.3f seconds' % (fileName,timeEnd))

    return True

def cropIt(pathName):
    global masterCounter
    faceLocations = get_image(pathName)
    imageObject = Image.open(pathName)
    cropPath = ''

    for (top,right,bottom,left) in faceLocations:
        masterCounter += 1
        timeStart = time.time()
        
        cropPosition = (
            left - 40,
            top - 40,
            right + 40, 
            bottom + 40           
        )

        cropName = 'master_'+str(masterCounter)+'.png'
        cropPath = './'+masterPath+'/'+cropName
        cropped = imageObject.crop(cropPosition)
        cropped.save(cropPath,'png')

        timeEnd = time.time() - timeStart
    
        print('Cropping %s tile %2i : [top:%5i, right:%5i, bottom:%5i, left:%5i] in %2.3f seconds' % (pathName,masterCounter,top,right,bottom,left,timeEnd))

        #recognizeIt(cropPath,cropName)

def runIt():
    listImages = gather()
    images = list(listImages.values())
    for img in images:
        cropIt(img)
   


runIt()
#faces = cropIt('./temp/3.JPG')
