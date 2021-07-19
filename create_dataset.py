import numpy as np
from cv2 import cv2 
from tracker import EuclideanDistTracker

class CreateDataset:
    def __init__(self):
        self.count = 0
        self.classified = 0
        self.tracker = EuclideanDistTracker()
        self.output_folder = 'generate_dataset'
        ### init config
        self.init_config()
        self.object_detector = cv2.createBackgroundSubtractorMOG2(history=200)
    
    def init_config(self, video_source='vid_samples/vid.2.mp4', output_folder = 'generate_dataset', roi_x1 = 60, roi_x2 = 800, roi_y1 = 230, roi_y2 = 400, l1=85, l2=130):
        
        self.video_source = video_source
        self.cap=cv2.VideoCapture(video_source)

        self.output_folder = output_folder

        self.roi_x1 = roi_x1        
        self.roi_x2 = roi_x2        
        self.roi_y1 = roi_y1 
        self.roi_y2 = roi_y2 
        self.l1 = l1
        self.l2 = l2

    def set_config(self, video_source, output_folder, roi_x1, roi_x2, roi_y1, roi_y2, l1, l2):
        
        if self.video_source != video_source:
            self.video_source = video_source
            self.cap=cv2.VideoCapture(video_source)

        if self.output_folder != output_folder:
            self.output_folder = output_folder

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
            frame = cv2.resize(frame, (800,400))

            roi_x1 = self.roi_x1
            roi_x2 = self.roi_x2
            roi_y1 = self.roi_y1
            roi_y2 = self.roi_y2

            roi = frame[roi_y1: roi_y2, roi_x1: roi_x2]

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

            for bid in tracker_obj:
                x,y,w,h,ids = bid

                if self.classified < ids:
                    cx = (x + x + w) // 2
                    cy = (y + y + h) // 2
                    
                    self.count += 1
                    # cv2.imshow('Cropped Bounding Rect', roi[y:y+h, x:x+w])
                    cv2.imwrite(self.output_folder+'/img'+'-'+str(ids)+'.png', roi[y:y+h, x:x+w])
                    # cv2.line(roi, (0, (l1+((l2-l1)//2))), (roi_x2, (l1+((l2-l1)//2))), (0, 0, 0), 2)
                    self.classified += 1
        
        return (_ret, frame)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
