import tkinter as tk
import PIL.Image, PIL.ImageTk
from create_dataset import CreateDataset 


class GUICreateDataset:
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
        self.gui_pilih_folder_output()

        self.gui_frame_inp_21()
        self.gui_set_roi()

        self.gui_frame_inp_22()
        self.gui_area_classification()

        self.create_dataset = CreateDataset()
        
        # open video source
        self.vid = self.create_dataset.main()
        
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 30
        self.pause = False

        self.gui_frame_3()
        self.gui_counting()
        self.gui_button()
    
    ### FRAME 1.1 - INPUT SOURCE VIDEO
    def gui_frame_inp_11(self):
        self.frame_inp_1 = tk.Frame(self.main_frame)
        self.frame_inp_1.pack(fill=tk.X)
        self.frame_inp_11 = tk.LabelFrame(self.frame_inp_1, text='Source Video')
        self.frame_inp_11.pack(fill=tk.X, side=tk.LEFT, padx=(0, 20))

    def gui_source_vid(self):
        self.gui_label_source_vid()
        self.gui_entry_source_vid()
        self.gui_btn_source_vid()

    def gui_label_source_vid(self):
        label = tk.Label(self.frame_inp_11, text="Input source video or RTSP Cred link: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_source_vid(self):
        self.entry_source_vid = entry_source_vid = tk.Entry(self.frame_inp_11)
        entry_source_vid.pack(side=tk.LEFT, padx=10)
        self.source_vid = default_source = 'vid_samples/vid.2.mp4'    

        entry_source_vid.insert(0, str(self.source_vid))
        entry_source_vid.bind('<KeyRelease>', self.ev_entry_source_vid)

    def ev_entry_source_vid(self, event):
        self.source_vid = self.entry_source_vid.get()
        self.refresh_gui_counting(reload=True)

    def gui_btn_source_vid(self):        
        self.btn_source_vid = btn_source_vid = tk.Button(self.frame_inp_11, text='Browse', command=self.ev_btn_source_vid)
        btn_source_vid.pack(side=tk.BOTTOM, padx=(0, 10))

    def ev_btn_source_vid(self):
        file_name = tk.filedialog.askopenfilename(parent=self.frame_inp_11, filetypes=[('Video File','*')], title='Plih Folder Output')
        self.entry_source_vid.delete(0, tk.END)
        self.entry_source_vid.insert(0, file_name)
        self.entry_source_vid.xview(len(file_name))
        self.source_vid = file_name
        self.refresh_gui_counting(reload=True)


    ### FRAME 1.2 - PILIH FOLDER OUTPUT
    def gui_frame_inp_12(self):
        self.frame_inp_12 = tk.LabelFrame(self.frame_inp_1, text='Folder Output')
        self.frame_inp_12.pack(fill=tk.X, side=tk.LEFT)

    def gui_pilih_folder_output(self):
        self.gui_label_pilih_folder_output()
        self.gui_entry_pilih_folder_output()
        self.gui_btn_pilih_folder_output()

    def gui_label_pilih_folder_output(self):
        label = tk.Label(self.frame_inp_12, text="Pilih Folder Output: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_pilih_folder_output(self):
        self.entry_pilih_folder_output = entry_pilih_folder_output = tk.Entry(self.frame_inp_12)
        entry_pilih_folder_output.pack(side=tk.LEFT, padx=10)

        self.pilih_folder_output = default_folder_output = 'generate_dataset/'
        entry_pilih_folder_output.insert(0, default_folder_output)

        entry_pilih_folder_output.bind('<KeyRelease>', self.ev_entry_pilih_folder_output)

    def ev_entry_pilih_folder_output(self, event):
        self.pilih_folder_output = self.entry_pilih_folder_output.get()
        self.refresh_gui_counting(reload=True)

    def gui_btn_pilih_folder_output(self):        
        self.btn_pilih_folder_output = btn_pilih_folder_output = tk.Button(self.frame_inp_12, text='Browse', command=self.ev_btn_pilih_folder_output)
        btn_pilih_folder_output.pack(side=tk.LEFT, padx=(0, 10))

    def ev_btn_pilih_folder_output(self):
        file_name = tk.filedialog.askdirectory(parent=self.frame_inp_12, title='Plih folder output', initialdir='generate_dataset')
        self.entry_pilih_folder_output.delete(0, tk.END)
        self.entry_pilih_folder_output.insert(0, file_name)
        self.entry_pilih_folder_output.xview(len(file_name))
        self.pilih_folder_output = file_name
        self.refresh_gui_counting(reload=True)

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
        entry_roi_x1.pack(side=tk.LEFT, padx=10)
        entry_roi_x1.insert(0, str(self.roi_x1))
        entry_roi_x1.bind('<KeyRelease>', self.ev_entry_roi_x1)

    def ev_entry_roi_x1(self, event):
        self.roi_x1 = int(self.entry_roi_x1.get())
        self.refresh_gui_counting()        
        
    def gui_label_roi_y1(self):
        label = tk.Label(self.frame_inp_21_1, text="Y1: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_roi_y1(self):
        self.entry_roi_y1 = entry_roi_y1 = tk.Entry(self.frame_inp_21_1, width=20)
        entry_roi_y1.pack(side=tk.LEFT, padx=10)
        entry_roi_y1.insert(0, str(self.roi_y1))
        entry_roi_y1.bind('<KeyRelease>', self.ev_entry_roi_y1)

    def ev_entry_roi_y1(self, event):
        self.roi_y1 = int(self.entry_roi_y1.get())
        self.refresh_gui_counting()

    def gui_label_roi_x2(self):
        label = tk.Label(self.frame_inp_21_2, text="X2: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_roi_x2(self):
        self.entry_roi_x2 = entry_roi_x2 = tk.Entry(self.frame_inp_21_2, width=20)
        entry_roi_x2.pack(side=tk.LEFT, padx=10)
        entry_roi_x2.insert(0, str(self.roi_x2))
        entry_roi_x2.bind('<KeyRelease>', self.ev_entry_roi_x2)

    def ev_entry_roi_x2(self, event):
        self.roi_x2 = int(self.entry_roi_x2.get())
        self.refresh_gui_counting()

    def gui_label_roi_y2(self):
        label = tk.Label(self.frame_inp_21_2, text="Y2: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_roi_y2(self):
        self.entry_roi_y2 = entry_roi_y2 = tk.Entry(self.frame_inp_21_2, width=20)
        entry_roi_y2.pack(side=tk.LEFT, padx=10)
        entry_roi_y2.insert(0, str(self.roi_y2))
        entry_roi_y2.bind('<KeyRelease>', self.ev_entry_roi_y2)

    def ev_entry_roi_y2(self, event):
        self.roi_y2 = int(self.entry_roi_y2.get())
        self.refresh_gui_counting()

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
        self.entry_l1 = entry_l1 = tk.Entry(self.frame_inp_22_1, width=20)
        entry_l1.pack(side=tk.LEFT, padx=10)
        entry_l1.insert(0, str(self.l1))
        entry_l1.bind('<KeyRelease>', self.ev_entry_l1)

    def ev_entry_l1(self, event):
        self.l1 = int(self.entry_l1.get())
        self.refresh_gui_counting()

    def gui_label_l2(self):
        label = tk.Label(self.frame_inp_22_2, text="L2: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_l2(self):
        self.entry_l2 = entry_l2 = tk.Entry(self.frame_inp_22_2, width=20)
        entry_l2.pack(side=tk.LEFT, padx=10)
        entry_l2.insert(0, str(self.l2))
        entry_l2.bind('<KeyRelease>', self.ev_entry_l2)

    def ev_entry_l2(self, event):
        self.l2 = int(self.entry_l2.get())
        self.refresh_gui_counting()

    ### FRAME 3
    def gui_frame_3(self):
        self.frame_3 = tk.Frame(self.main_frame)
        self.frame_3.pack(fill=tk.X)

    def gui_counting(self, reload=True):

        # Create a canvas that can fit the above video source size
        self.canvas = canvas = tk.Canvas(self.frame_3, width = 800, height = 400)
        canvas.config(highlightbackground="black")
        canvas.pack()

        if reload == True:
            self.ret_counting, self.frame_counting = self.create_dataset.main()

        if self.ret_counting:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame_counting))
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
        self.canvas.create_line(self.roi_x1, self.roi_y1 + (self.l1+((self.l2-self.l1)//2)), self.roi_x2, self.roi_y1 + (self.l1+((self.l2-self.l1)//2)), fill="black", width=2)

    def run_gui_counting(self):
        ret, frame = self.create_dataset.main()
        if ret and not self.pause:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.frame_3.after(self.delay, self.run_gui_counting)
        else:
            self.pause = False
        #     del self.create_dataset
        #     self.canvas.destroy()
    
    ### BUTTON PLAY
    def gui_button(self):
        self.frame_3_btn = tk.Frame(self.frame_3)
        self.frame_3_btn.pack(fill=tk.X, pady=(10, 15))

        self.button_1 = button_1 = tk.Button(self.frame_3_btn, text='Play', width=25, command=self.play_gui_counting)
        button_1.pack(side=tk.LEFT, padx=(0, 10))
        
        self.button_3 = button_3 = tk.Button(self.frame_3_btn, text='Pause / Resume', width=25, command=self.pause_gui_counting)
        button_3.pack(side=tk.LEFT, padx=(0, 10))
        
        self.button_4 = button_4 = tk.Button(self.frame_3_btn, text='Stop', width=25, command=self.stop_gui_counting)
        button_4.pack(side=tk.LEFT, padx=(0, 10))
        
        self.button_2 = button_2 = tk.Button(self.frame_3_btn, text='Next Frame', width=25, command=lambda: self.refresh_gui_counting(reload=True))
        button_2.pack(side=tk.LEFT, padx=(0, 10))

    ### Reload untuk configurasi line
    def refresh_gui_counting(self, reload=False):
        # self.canvas.delete('all')
        self.canvas.destroy()
        self.button_1.destroy()
        self.frame_3.destroy()
        self.set_create_dataset_config()
        self.gui_frame_3()
        self.gui_counting(reload=reload)
        self.gui_button()
    
    def play_gui_counting(self):
        self.refresh_gui_counting(reload=True)
        self.run_gui_counting()

    def pause_gui_counting(self):
        if self.pause == False:
            self.pause = True
        else:
            self.pause = False
            self.play_gui_counting()

    def stop_gui_counting(self):
        self.pause_gui_counting()
        self.create_dataset = CreateDataset()
        self.refresh_gui_counting(reload=True)

    def set_create_dataset_config(self):
        self.create_dataset.set_config(video_source=self.source_vid, output_folder=self.pilih_folder_output, roi_x1=self.roi_x1, roi_x2=self.roi_x2, roi_y1=self.roi_y1, roi_y2=self.roi_y2, l1=self.l1, l2=self.l2)
