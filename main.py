import numpy as np
import pickle
from cv2 import cv2 
import to_xlsx
from cnn import CNN
from tracker import EuclideanDistTracker

count = [0, 0, 0]
prediction = ''
classified = 0
tracker = EuclideanDistTracker()

f = open('saved_model/model-2.pckl', 'rb')
model = pickle.load(f)
f.close()
cnn = CNN(model=model)


### Read Video
cap=cv2.VideoCapture("vid_samples/vid.1.mp4")
# cap=cv2.VideoCapture("rtsp://user:user123!@202.51.112.66:2215/Streaming/Channels/101/")

object_detector = cv2.createBackgroundSubtractorMOG2()
_, frame = cap.read()
l1 = 85
l2 = 130
# lend = 800
while(cap.isOpened()):
    # video to frame
    _, frame = cap.read()
    frame = cv2.resize(frame, (800,400))

    # frame to roi
    roi = frame[230: 400, 60: 800]
    # cv2.line(roi, (0, l1), (730,  l1), (128, 0, 128), 1)
    # cv2.line(roi, (0, l2), (730,  l2), (128, 0, 128), 1)
    
    # # preprocessing citra
    mask = object_detector.apply(roi)
    blur = cv2.GaussianBlur(mask,(5,5),0)
    h,mask = cv2.threshold(blur,0,250,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((2,2),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # segmentasi citra
    contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    obj_detections = []
    for cnt in contours:
        #calculate area and remove small element
        area = cv2.contourArea(cnt)
        if area > 550:
            x, y, w, h = cv2.boundingRect(cnt)
            cy = (y + y + h) // 2
            if cy > l1 and cy < l2:
                # tracking / indexing objek citra dengan penanda
                obj_detections.append([x,y,w,h])
        
    tracker_obj = tracker.update(obj_detections)

    for bid in tracker_obj:
        x,y,w,h,ids = bid

        if classified < ids:
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            
            img = cv2.cvtColor(roi[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            prediction = cnn.predict(img)
            cv2.line(roi, (0, (l1+((l2-l1)//2))), (800, (l1+((l2-l1)//2))), (255, 0, 0), 2)
            cv2.putText(roi, prediction, (x, y+h-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (244,0,0))

            if prediction == 'sepeda motor':
                count[0] += 1
            elif prediction == 'sepeda':
                count[1] += 1
            elif prediction == 'mobil penumpang':
                count[2] += 1
            cv2.circle(roi, (cx, cy), 4, (0, 244, 0), 2)
            cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 244, 0), 2)

            classified += 1
            
    cv2.putText(frame, 'sepeda motor: '+str(count[0]), (500, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
    cv2.putText(frame, 'sepeda: '+str(count[1]), (500, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
    cv2.putText(frame, 'mobil penumpang: '+str(count[2]), (500, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))

    cv2.imshow('frame', frame)
    cv2.imshow('roi', roi)
    # cv2.imshow('mask', mask)
    to_xlsx.save_to_xlsx(count)

    key = cv2.waitKey(60)
    if key == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed
    elif(key == 27):
        break

cap.release()
cv2.destroyAllWindows()
