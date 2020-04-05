import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import time
from PIL import Image
import PIL, sys
from shutil import copyfile

masterPath = 'master'
resultPath = 'results'
tempPath = 'temp'

tempCounter = 0
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

def get_encoded_faces():
    encoded = {}
    
    for dirpath, dnames, fnames in os.walk("./"+masterPath):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                if f.split("_")[0] == 'master':
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

def create_directory(dirName):
    try:
        os.mkdir(dirName)
    except OSError:
        return False
    else:
        return True

def recognizeIt(pathName,cropObjects):
    global masterCounter
    timeStart = time.time()

    for obj in cropObjects:
        faces = get_encoded_faces()

        if len(faces) == 0:
            masterCounter += 1
            os.rename(r''+obj,r'./'+masterPath+'/master_'+str(masterCounter)+'_1.png')
            if create_directory("./"+resultPath+'/'+str(masterCounter)):
                copyfile(pathName, "./"+resultPath+'/'+str(masterCounter)+'/'+pathName.split('/')[-1])


        else:
            faces_encoded = list(faces.values())
            known_face_names = list(faces.keys())
            img = cv2.imread(obj, 1)

            face_locations = face_recognition.face_locations(img)
            unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

            for face_encoding in unknown_face_encodings:
                matches = face_recognition.compare_faces(faces_encoded, face_encoding)

                face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    localeCounter = 1
                    existingMasterIndex = known_face_names[best_match_index].split("_")[1]

                    for nm in known_face_names:
                        if nm.split("_")[1] == existingMasterIndex:
                            localeCounter += 1

                    os.rename(r''+obj,r'./'+masterPath+'/master_'+str(existingMasterIndex)+'_'+str(localeCounter)+'.png')
                    copyfile(pathName, "./"+resultPath+'/'+str(existingMasterIndex)+'/'+pathName.split('/')[-1])
                else:
                    masterCounter += 1
                    os.rename(r''+obj,r'./'+masterPath+'/master_'+str(masterCounter)+'_1.png')
                    create_directory("./"+resultPath+'/'+str(masterCounter))
                    if create_directory("./"+resultPath+'/'+str(masterCounter)):
                        copyfile(pathName, "./"+resultPath+'/'+str(masterCounter)+'/'+pathName.split('/')[-1])

    timeEnd = time.time() - timeStart
    print("Analyze image %s [Total Faces : %3i] in %2.6s" % (pathName, len(cropObjects), timeEnd))

def cropIt(pathName):
    global tempCounter
    faceLocations = get_image(pathName)
    imageObject = Image.open(pathName)
    cropObj = []

    for (top,right,bottom,left) in faceLocations:
        timeStart = time.time()
        
        cropPosition = (
            left - 20,
            top - 20,
            right + 20, 
            bottom + 20           
        )

        cropped = imageObject.crop(cropPosition)
        cropPath = './'+masterPath+'/temp_'+str(tempCounter)+'.png'
        cropped.save(cropPath,'png')
        cropObj.append(cropPath)

        tempCounter += 1

        timeEnd = time.time() - timeStart
    
        print('Cropping %s tile %2i : [top:%5i, right:%5i, bottom:%5i, left:%5i] in %2.3f seconds' % (pathName,masterCounter,top,right,bottom,left,timeEnd))

    #print(cropObj)
    recognizeIt(pathName,cropObj)

def runIt():
    listImages = gather()
    images = list(listImages.values())
    for img in images:
        cropIt(img)
   
runIt()
#faces = cropIt('./temp/3.JPG')
