import cv2
import numpy as np
from cnn import *

prediction = ''
run = 1
# Read Video
cap=cv2.VideoCapture("vid_samples/vid.1.mp4")
# cap=cv2.VideoCapture("rtsp://user:user123!@202.51.112.66:2215/Streaming/Channels/101/")
while run:

    ### Create object detector using background subtractor mog2
    object_detector = cv2.createBackgroundSubtractorMOG2(history=200)
    success, frame = cap.read()
    while(cap.isOpened()):
        success, frame = cap.read()

        if not success:
            continue
        
        frame = cv2.resize(frame, (800,400))

        roi = frame[200: 400, 130: 800]
        
        mask = object_detector.apply(roi)
        blur = cv2.GaussianBlur(mask,(15,15),0)
        h,mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        kernel = np.ones((2,2),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            #calculate area and remove small element
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                if y > 10 and y < 190:
                    img_pred = cv2.cvtColor(roi[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    prediction = cnn.predict(img_pred)
                    cv2.putText(roi, prediction, (x, y+h-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (244,0,0))

                cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 244, 0), 2)

        cv2.imshow('roi', roi)
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)

        key = cv2.waitKey(60)
        if key == ord('p'):
            cv2.waitKey(-1) #wait until any key is pressed
        elif(key == 27):
            run = 0
            break

cap.release()
cv2.destroyAllWindows()