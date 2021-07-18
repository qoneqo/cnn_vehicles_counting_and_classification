import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import time
from core import Core 

class GUICounting:
    def __init__(self, main_frame):
        self.main_frame = main_frame
        self.roi_x1 = 60
        self.roi_x2 = 800
        self.roi_y1 = 230
        self.roi_y2 = 400
         
        self.l1 = 85
        self.l2 = 130

        self.gui_frame_inp_11()
        self.gui_source_vid()

        self.gui_frame_inp_12()
        self.gui_pilih_model()

        self.gui_frame_inp_21()
        self.gui_set_roi()

        self.gui_frame_inp_22()
        self.gui_area_classification()

        self.gui_counting()
        self.gui_button_1()
        # After it is called once, the update method will be automatically called every delay milliseconds
    
    ### FRAME 1.1 - INPUT SOURCE VIDEO
    def gui_frame_inp_11(self):
        self.frame_inp_1 = tk.Frame(self.main_frame)
        self.frame_inp_1.pack(fill=tk.X)
        self.frame_inp_11 = tk.LabelFrame(self.frame_inp_1, text='Source Video')
        self.frame_inp_11.pack(fill=tk.X, side=tk.LEFT, padx=(0, 20))

    def gui_source_vid(self):
        self.gui_label_source_vid()
        self.gui_entry_source_vid()

    def gui_label_source_vid(self):
        label = tk.Label(self.frame_inp_11, text="Input source video or RSTP Cred link: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_source_vid(self):
        self.entry_source_vid = entry_source_vid = tk.Entry(self.frame_inp_11)
        entry_source_vid.bind('<KeyRelease>', self.ev_entry_source_vid)
        entry_source_vid.pack(side=tk.LEFT, padx=10)

    def ev_entry_source_vid(self, event):
        print(self.entry_source_vid.get())

    ### FRAME 1.2 - PILIH MODEL
    def gui_frame_inp_12(self):
        self.frame_inp_12 = tk.LabelFrame(self.frame_inp_1, text='Model')
        self.frame_inp_12.pack(fill=tk.X, side=tk.LEFT)

    def gui_pilih_model(self):
        self.gui_label_pilih_model()
        self.gui_entry_pilih_model()

    def gui_label_pilih_model(self):
        label = tk.Label(self.frame_inp_12, text="Pilih Model: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_pilih_model(self):
        entry_source_vid = tk.Entry(self.frame_inp_12)
        entry_source_vid.pack(side=tk.LEFT, padx=10)

    ### FRAME 2.1 - INPUT ROI
    def gui_frame_inp_21(self):
        self.frame_inp_2 = tk.Frame(self.main_frame)
        self.frame_inp_2.pack(fill=tk.X, pady=(10, 15))

        self.frame_inp_21 = tk.LabelFrame(self.frame_inp_2, text='Set Region of Interest (ROI)')
        self.frame_inp_21.pack(fill=tk.X, side=tk.LEFT, padx=(0, 20))

        self.frame_inp_21_1 = tk.Frame(self.frame_inp_21)
        self.frame_inp_21_1.pack(fill=tk.X)
        self.frame_inp_21_2 = tk.Frame(self.frame_inp_21)
        self.frame_inp_21_2.pack(fill=tk.X)
    
    def gui_set_roi(self):
        self.gui_label_roi_x1()
        self.gui_entry_roi_x1()
        self.gui_label_roi_y1()
        self.gui_entry_roi_y1()
        self.gui_label_roi_x2()
        self.gui_entry_roi_x2()
        self.gui_label_roi_y2()
        self.gui_entry_roi_y2()

    def gui_label_roi_x1(self):
        label = tk.Label(self.frame_inp_21_1, text="X1: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_roi_x1(self):
        self.entry_roi_x1 = entry_roi_x1 = tk.Entry(self.frame_inp_21_1, width=20)
        entry_roi_x1.bind('<KeyRelease>', self.ev_entry_roi_x1)
        entry_roi_x1.pack(side=tk.LEFT, padx=10)

    def ev_entry_roi_x1(self, event):
        self.roi_x1 = self.entry_roi_x1.get()
        self.gui_counting()

    def gui_label_roi_y1(self):
        label = tk.Label(self.frame_inp_21_1, text="Y1: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_roi_y1(self):
        entry_roi_x1 = tk.Entry(self.frame_inp_21_1, width=20)
        entry_roi_x1.pack(side=tk.LEFT, padx=10)

    def gui_label_roi_x2(self):
        label = tk.Label(self.frame_inp_21_2, text="X2: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_roi_x2(self):
        entry_roi_x2 = tk.Entry(self.frame_inp_21_2, width=20)
        entry_roi_x2.pack(side=tk.LEFT, padx=10)

    def gui_label_roi_y2(self):
        label = tk.Label(self.frame_inp_21_2, text="Y2: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_roi_y2(self):
        entry_roi_x1 = tk.Entry(self.frame_inp_21_2, width=20)
        entry_roi_x1.pack(side=tk.LEFT, padx=10)

    ### FRAME 2.2 - INPUT AREA CLASSIFICATION

    def gui_frame_inp_22(self):        
        self.frame_inp_22 = tk.LabelFrame(self.frame_inp_2, text='Set Area Classification')
        self.frame_inp_22.pack(fill=tk.X, side=tk.LEFT)
        
        self.frame_inp_22_1 = tk.Frame(self.frame_inp_22)
        self.frame_inp_22_1.pack(fill=tk.X)
        self.frame_inp_22_2 = tk.Frame(self.frame_inp_22)
        self.frame_inp_22_2.pack(fill=tk.X)

    def gui_area_classification(self):
        self.gui_label_l1()
        self.gui_entry_l1()
        self.gui_label_l2()
        self.gui_entry_l2()

    def gui_label_l1(self):
        label = tk.Label(self.frame_inp_22_1, text="L1: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_l1(self):
        entry_l1 = tk.Entry(self.frame_inp_22_1, width=20)
        entry_l1.pack(side=tk.LEFT, padx=10)

    def gui_label_l2(self):
        label = tk.Label(self.frame_inp_22_2, text="L2: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_l2(self):
        entry_l2 = tk.Entry(self.frame_inp_22_2, width=20)
        entry_l2.pack(side=tk.LEFT, padx=10)

    ### BUTTON PLAY

    def gui_button_1(self):
        button_1 = tk.Button(self.main_frame, text='Play Video', width=25, command=self.run_gui_counting)
        button_1.pack()

    def gui_counting(self):
        video_source = "vid_samples/vid.2.mp4"
        self.core = Core(video_source)

        # open video source 
        self.vid = self.core.main()

        # Create a canvas that can fit the above video source size        
        self.canvas = canvas = tk.Canvas(self.main_frame, width = 800, height = 400)
        canvas.config(highlightbackground="black")
        canvas.pack()
        self.delay = 30

        ret, frame = self.core.main()        

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            
            self.draw_line()

    def draw_line(self):
        # line roi
        self.canvas.create_line(self.roi_x1, self.roi_y1, self.roi_x2,  self.roi_y1, fill="red", width=2) # horizontal 1
        self.canvas.create_line(self.roi_x1, self.roi_y2, self.roi_x2,  self.roi_y2, fill="red", width=2) # horizontal 2
        self.canvas.create_line(self.roi_x1, self.roi_y1, self.roi_x1, self.roi_y2, fill="red", width=2) # vertical 1
        self.canvas.create_line(self.roi_x2, self.roi_y1, self.roi_x2, self.roi_y2, fill="red", width=2) # vertical 2

        # line classification area
        self.canvas.create_line(self.roi_x1, self.roi_y1 + self.l1, self.roi_x2, self.roi_y1 + self.l1, fill="green", width=2)
        self.canvas.create_line(self.roi_x1, self.roi_y1 + self.l2, self.roi_x2,  self.roi_y1 + self.l2, fill="green", width=2)

    def run_gui_counting(self):
        # Get a frame from the video source
        ret, frame = self.core.main()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.main_frame.after(self.delay, self.run_gui_counting)
        else:
            del self.core
            self.canvas.destroy()
