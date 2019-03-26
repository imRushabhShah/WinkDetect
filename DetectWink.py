import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys

def detectWink(frame, location, ROI, cascade):
    eyes = cascade.detectMultiScale(
        ROI, 1.15, 3, 0|cv2.CASCADE_SCALE_IMAGE, (10, 20)) 
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
    return len(eyes) == 1    # number of eyes is one

def detect(frame, faceCascade, eyesCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # possible frame pre-processing:
    # gray_frame = cv2.equalizeHist(gray_frame)
    # gray_frame = cv2.medianBlur(gray_frame, 5)

    scaleFactor = 1.15 # range is from 1 to ..
    minNeighbors = 3   # range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0 or 0|cv2.CASCADE_SCALE_IMAGE 
    minSize = (30,30) # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        gray_frame, 
        scaleFactor, 
        minNeighbors, 
        flag, 
        minSize)

    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = gray_frame[y:y+h, x:x+w]
        if detectWink(frame, (x, y), faceROI, eyesCascade):
            detected += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    return detected


def run_on_folder(cascade1, cascade2, folder):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]

    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCount

def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False
    
    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_eye.xml')

    if(len(sys.argv) == 2): # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, eye_cascade, folderName)
        print("Total of ", detections, "detections")
    else: # no arguments
        runonVideo(face_cascade, eye_cascade)

