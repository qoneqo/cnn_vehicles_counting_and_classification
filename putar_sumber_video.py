from cv2 import cv2 

class PutarSumberVideo:
    def __init__(self):
        ### init config
        self.init_config()
    
    def init_config(self, video_source='vid_samples/vid.2.mp4'):
        self.video_source = video_source
        self.cap=cv2.VideoCapture(video_source)

    def set_config(self, video_source):
        if self.video_source != video_source:
            self.video_source = video_source
            self.cap=cv2.VideoCapture(video_source)

    def main(self):        
        _ret, frame = self.cap.read()

        # if ret in GUI:
        if(_ret == True and self.cap.isOpened()):
            frame = cv2.resize(frame, (800,400))

        return (_ret, frame)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
