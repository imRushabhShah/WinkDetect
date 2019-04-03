import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys
# import PIL
# from PIL import Image
#

def detectWink(frame, location, ROI, cascade,right=False):
#     ROI = cv2.equalizeHist(ROI)
    x_r,y_r=ROI.shape
    a = int(x_r*0.2)
    eyes = cascade.detectMultiScale(ROI, 1.04, 15, 0|cv2.CASCADE_SCALE_IMAGE, (5, 10))
#     eyes = cascade.detectMultiScale(ROI, 1.04, 20, 0|cv2.CASCADE_SCALE_IMAGE, (5, 10))
#     eyes = cascade.detectMultiScale(ROI, 1.04, 20, 0|cv2.CASCADE_SCALE_IMAGE, (a, 2*a))
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        if right:
            
            # print(location,ROI.shape)
            x_r = int(x_r*7/8)
            cv2.rectangle(frame, (x+x_r,y), (x+w+x_r,y+h), (0, 255, 255), 2)
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 255), 2)
    return len(eyes)>0    # number of eyes is one

def detectWink_train(frame, location, ROI, cascade,scale,mn):
    # ROI = cv2.equalizeHist(ROI)
    # x,y = ROI.shape
    # ROI = ROI[:][:int(4*x/7)]
    # x,y = ROI.shape
    # ROI = cv2.equalizeHist(ROI)

    eyes = cascade.detectMultiScale(
        ROI, scale, mn, 0|cv2.CASCADE_SCALE_IMAGE, (10, 20))
#     eyes = cascade.detectMultiScale(
#         ROI, 1.15, 3, 0|cv2.CASCADE_SCALE_IMAGE, (10, 20))
    # for e in eyes:
    #     e[0] += location[0]
    #     e[1] += location[1]
    #     x, y, w, h = e[0], e[1], e[2], e[3]
    #
    #     cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
    return len(eyes)>0    # number of eyes is one

def detect(frame, faceCascade, eyesCascade):
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # possible frame pre-processing:
    # gray_frame = cv2.equalizeHist(gray_frame)
    gray_frame = cv2.medianBlur(gray_frame, 5)

    scaleFactor = 1.02 # range is from 1 to ..
    minNeighbors = 10 # range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0 or 0|cv2.CASCADE_SCALE_IMAGE
    minSize = (50,50) # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor,
        minNeighbors,
        flag,
        minSize)
#     faces = face_cascade.detectMultiScale(gray_frame, 1.2, 5)
    detected = 0
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCount=0
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_frame = cv2.convertScaleAbs(gray_frame,1.2,5)
    
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        # showlive = True
        # if showlive:
        #     cv2.imshow("face", gray_frame[y:y+h, x:x+w])
        #     if cv2.waitKey(0):
        #         showlive = False


        faceROI_L = gray_frame[y:int(y+(h*4/7)), x:x+int(w*4/7)]
        faceROI_R = gray_frame[y:int(y+(h*4/7)), x+int(w*3/7):(x+w)]
        eyeCount = 0
        eyeL = detectWink(frame, (x, y), faceROI_L, eyesCascade)
        if eyeL:
             cv2.rectangle(frame, (x,y), (x+int(w*4/7),int(y+(h*4/7))), (0, 0, 255), 2)
        eyeR = detectWink(frame, (x, y), faceROI_R, eyesCascade,right=True)
        if eyeR:
            cv2.rectangle(frame, (x+int(w*3/7),y), ((x+w),int(y+(h*4/7))), (0, 0, 255), 2)
        eyeCount = eyeL + eyeR
        faceCount +=1
        # print("for face ",faceCount," Number of eyes = ",eyeCount)
        if eyeCount == 1:
            detected += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
            cv2.putText(frame,'Wink detected',(x,y), font, 0.5,(255,255,255),2,cv2.LINE_AA)

        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    return detected

def detect_train(frame, faceCascade, eyesCascade,sF,mnF,sE,mnE):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # possible frame pre-processing:
    # gray_frame = cv2.equalizeHist(gray_frame)
    gray_frame = cv2.medianBlur(gray_frame, 5)

    scaleFactor = 1.02 # range is from 1 to ..
    minNeighbors = 10 # range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0 or 0|cv2.CASCADE_SCALE_IMAGE
    minSize = (50,50) # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor,
        minNeighbors,
        flag,
        minSize)
    detected = 0
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCount=0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]

        faceROI_L = gray_frame[y:int(y+(h*4/7)), x:x+int(w*4/7)]
        faceROI_R = gray_frame[y:int(y+(h*4/7)), x+int(w*3/7):(x+w)]
        eyeCount = 0
        eyeL = detectWink_train(frame, (x, y), faceROI_L, eyesCascade,sE,mnE)
        # if eyeL:
        #      cv2.rectangle(frame, (x,y), (x+int(w*4/7),int(y+(h*4/7))), (0, 0, 255), 2)
        eyeR = detectWink_train(frame, (x, y), faceROI_R, eyesCascade,sE,mnE)
        # if eyeR:
        #     cv2.rectangle(frame, (x+int(w*3/7),y), ((x+w),int(y+(h*4/7))), (0, 0, 255), 2)
        eyeCount = eyeL + eyeR
        faceCount +=1
        #
        # faceROI = gray_frame[y:y+h, x:x+w]
        # eyeCount = detectWink_train(frame, (x, y), faceROI, eyesCascade,sE,mnE)
        # faceCount +=1
        # # print("for face ",faceCount," Number of eyes = ",eyeCount)
        if eyeCount == 1:
            detected += 1
            # cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        # else:
            # cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    return detected


def run_on_folder(cascade1, cascade2, folder, showImg=True):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]

    windowName = None
    totalCount = 0
    for f in files:
        # print('\n\n',f)
        img = cv2.imread(f)
        # r = 500.0/img.shape[1]
        # dim =(500,int(img.shape[0]*r))
        # img = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            # print("detected = ",lCnt,'\n')
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            if showImg:
                cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(windowName, img)
                cv2.waitKey(0)
    return totalCount

def run_on_folder_train(cascade1, cascade2, folder, sF,mnF,sE,mnE):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]

    windowName = None
    totalCount = 0
    for f in files:
        # print('\n\n',f)
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            lCnt = detect_train(img, cascade1, cascade2,sF,mnF,sE,mnE)
            # print("detected = ",lCnt)
            totalCount += lCnt
            # if windowName != None:
                # cv2.destroyWindow(windowName)
            # windowName = f
            # if showImg:
            #     cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            #     cv2.imshow(windowName, img)
            #     cv2.waitKey(0)
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
    if len(sys.argv) != 1 and len(sys.argv) != 2 and len(sys.argv) != 3:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_frontalface_default.xml')

    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
    #                               + 'haarcascade_frontalface_alt2.xml')


    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_eye.xml')

    if(len(sys.argv) == 3):
        # custom train model
        folderName = sys.argv[1]
        a=1.02
        b=10
        for c in np.arange(1.05,1.9,0.05):
            for d in range(1,12):
                detections = run_on_folder_train(face_cascade, eye_cascade, folderName,a,b,c,d)
                with open('log2.txt','a') as file:
                    file.write(str(str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(detections)+"\n"))
                print(a,b,c,d,"Total of ", detections, "detections")

    elif(len(sys.argv) == 2): # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, eye_cascade, folderName)
        print("Total of ", detections, "detections")
    else: # no arguments
        runonVideo(face_cascade, eye_cascade)
