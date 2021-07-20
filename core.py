import numpy as np
import pickle
from cv2 import cv2 
import to_xlsx
from cnn import CNN
from tracker import EuclideanDistTracker
import time

class Core:
    def __init__(self):
        self.classified = 0
        self.prediction = ''
        self.tracker = EuclideanDistTracker()

        self.vehicles = []
        self.vindex = -1
        self.mytime = 0

        ### init config
        self.init_config()
        self.object_detector = cv2.createBackgroundSubtractorMOG2(history=200)
    
    def init_config(self, video_source='vid_samples/vid.2.mp4', model='saved_model/model-2.pckl', roi_x1 = 60, roi_x2 = 800, roi_y1 = 230, roi_y2 = 400, l1=85, l2=130):
        
        self.video_source = video_source
        self.cap=cv2.VideoCapture(video_source)

        f = open(model, 'rb')
        self.model = pickle.load(f)
        f.close()

        l = model.split('/')
        model_name = l.pop()
        self.model_name = model_name = model_name.split('.')[0]
        folder_model = '/'.join(l) + '/'
        f = open(folder_model+'nk-'+model_name+'.pckl', 'rb')
        self.nama_kendaraan = pickle.load(f)
        f.close()
        self.count = [0] * len(self.nama_kendaraan)


        self.cnn = CNN(model=self.model, saved_model=folder_model, model_name=model_name)

        self.roi_x1 = roi_x1        
        self.roi_x2 = roi_x2        
        self.roi_y1 = roi_y1 
        self.roi_y2 = roi_y2 
        self.l1 = l1
        self.l2 = l2

    def set_config(self, video_source, model, roi_x1, roi_x2, roi_y1, roi_y2, l1, l2):
        
        if self.video_source != video_source:
            self.video_source = video_source
            self.cap=cv2.VideoCapture(video_source)

        l = model.split('/')
        model_name = l.pop()
        self.model_name = model_name = model_name.split('.')[0]
        folder_model = '/'.join(l) + '/'
        f = open(folder_model+'nk-'+model_name+'.pckl', 'rb')
        self.nama_kendaraan = pickle.load(f)
        f.close()
        self.count = [0] * len(self.nama_kendaraan)


        if self.model != model:
            f = open(model, 'rb')
            self.model = pickle.load(f)
            f.close()
            self.cnn = CNN(model=self.model, saved_model=folder_model, model_name=model_name)

        

        if self.roi_x1 != roi_x1:
            self.roi_x1 = roi_x1
        
        if self.roi_x2 != roi_x2:
            self.roi_x2 = roi_x2
        
        if self.roi_y1 != roi_y1:
            self.roi_y1 = roi_y1
        
        if self.roi_y2 != roi_y2:
            self.roi_y2 = roi_y2
        
        if self.l1 != l1:
            self.l1 = l1
                
        if self.l2 != l2:
            self.l2 = l2

    def main(self):        
        l1 = self.l1
        l2 = self.l2
        _ret, frame = self.cap.read()

        # if ret in GUI:
        if(_ret == True and self.cap.isOpened()):

            # video to frame
            frame = cv2.resize(frame, (800,400))
            # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # frame to roi

            roi_x1 = self.roi_x1
            roi_x2 = self.roi_x2
            roi_y1 = self.roi_y1
            roi_y2 = self.roi_y2

            # roi = frame[230:400, 60:800]
            roi = frame[roi_y1: roi_y2, roi_x1: roi_x2]

            # # line roi
            # cv2.line(frame, (roi_x1, roi_y1), (roi_x2,  roi_y1), (255, 0, 0), 2) # horizontal 1
            # cv2.line(frame, (roi_x1, roi_y2), (roi_x2,  roi_y2), (255, 0, 0), 2) # horizontal 2
            # cv2.line(frame, (roi_x1, roi_y1), (roi_x1, roi_y2), (255, 0, 0), 2) # vertical 1
            # cv2.line(frame, (roi_x2, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2) # vertical 2

            # # line classification area
            # cv2.line(roi, (0, l1), (roi_x2,  l1), (0, 255, 0), 1)
            # cv2.line(roi, (0, l2), (roi_x2,  l2), (0, 255, 0), 1)
            
            # preprocessing citra
            mask = self.object_detector.apply(roi)
            blur = cv2.GaussianBlur(mask,(5,5),0)
            h,mask = cv2.threshold(blur,0,254,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

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
                
            tracker_obj = self.tracker.update(obj_detections)

            named_tuple = time.localtime()
            # time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
            filename = self.model_name + '-' + time.strftime("%d-%m-%Y", named_tuple)
            timeq = time.strftime("%H:00:00", named_tuple)
            if self.mytime != timeq:
                self.mytime = timeq
                self.count = [0] * len(self.nama_kendaraan)
                self.vindex += 1
                self.count.append(self.mytime)
                self.vehicles.append(self.count)


            for bid in tracker_obj:
                x,y,w,h,ids = bid

                if self.classified < ids:
                    cx = (x + x + w) // 2
                    cy = (y + y + h) // 2
                    
                    img = cv2.cvtColor(roi[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    self.prediction = self.cnn.predict(img)
                    cv2.line(roi, (0, (l1+((l2-l1)//2))), (roi_x2, (l1+((l2-l1)//2))), (0, 0, 0), 2)
                    cv2.putText(roi, self.prediction, (x, y+h-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (244,0,0))

                    self.count[self.nama_kendaraan.index(self.prediction)] += 1

                    cv2.circle(roi, (cx, cy), 4, (0, 0, 255), 2)
                    cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 0, 255), 2)

                    self.vehicles[self.vindex] = self.count
      
                    self.classified += 1
            
            for ind in range(len(self.nama_kendaraan)):
                posh = 20*(ind+1)
                cv2.putText(frame, self.nama_kendaraan[ind]+': '+str(self.count[ind]), (500, posh), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))

            # cv2.imshow('frame', frame)
            # cv2.imshow('roi', roi)
            # cv2.imshow('mask', mask)
            to_xlsx.save_to_xlsx(self.count, self.nama_kendaraan, filename)

            # key = cv2.waitKey(60)
            # if key == ord('p'):
            #     cv2.waitKey(-1) #wait until any key is pressed

            # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        return (_ret, frame)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
