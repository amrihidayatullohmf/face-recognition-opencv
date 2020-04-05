import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import time
from PIL import Image
import PIL, sys

cropPath = './cropping/'

def get_image(pathName):
    face = fr.load_image_file(pathName)
    return fr.face_locations(face)

def cropIt(pathName):
    faceLocations = get_image(pathName)
    imageObject = Image.open(pathName)
    counter = 0
    img = cv2.imread(pathName, 1)

    for (top,right,bottom,left) in faceLocations:
        counter += 1
        timeStart = time.time()

        cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
        
        cropPosition = (
            left,
            top,
            right,
            bottom            
        )
        cropped = imageObject.crop(cropPosition)
        cropped.save(cropPath+'_crop_'+str(counter)+'.png','png')

        timeEnd = time.time() - timeStart
    
        print('Cropping tile %2i : [top:%5i, right:%5i, bottom:%5i, left:%5i] in %2.3f seconds' % (counter,top,right,bottom,left,timeEnd))

    print('Display')
    cv2.imshow('Video', img)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names 

faces = cropIt('./temp/3.JPG')
