import tkinter as tk
import PIL.Image, PIL.ImageTk
from putar_sumber_video import PutarSumberVideo 

class GUIPutarSumberVideo:
    def __init__(self, main_frame):
        self.main_frame = main_frame
        self.putar_sumber_video = PutarSumberVideo()

        self.gui_frame_inp_11()
        self.gui_source_vid()
        
        self.delay = 30
        self.pause = False

        self.gui_frame_3()
        self.gui_video()
        self.gui_button()
    
    ### FRAME 1.1 - INPUT SOURCE VIDEO
    def gui_frame_inp_11(self):
        self.frame_inp_1 = tk.Frame(self.main_frame)
        self.frame_inp_1.pack(pady=(0, 20))
        self.frame_inp_11 = tk.LabelFrame(self.frame_inp_1, text='Source Video')
        self.frame_inp_11.pack(padx=(0, 20))

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
        self.refresh_gui_video(reload=True)

    def gui_btn_source_vid(self):        
        self.btn_source_vid = btn_source_vid = tk.Button(self.frame_inp_11, text='Browse', command=self.ev_btn_source_vid)
        btn_source_vid.pack(side=tk.BOTTOM, padx=(0, 10))

    def ev_btn_source_vid(self):
        file_name = tk.filedialog.askopenfilename(parent=self.frame_inp_11, filetypes=[('Video File','*')], title='Plih Folder Output')
        self.entry_source_vid.delete(0, tk.END)
        self.entry_source_vid.insert(0, file_name)
        self.entry_source_vid.xview(len(file_name))
        self.source_vid = file_name
        self.refresh_gui_video(reload=True)

    ### FRAME 3
    def gui_frame_3(self):
        self.frame_3 = tk.Frame(self.main_frame)
        self.frame_3.pack(fill=tk.X)

    def gui_video(self, reload=True):

        # Create a canvas that can fit the above video source size
        self.canvas = canvas = tk.Canvas(self.frame_3, width = 800, height = 400)
        canvas.config(highlightbackground="black")
        canvas.pack()

        if reload == True:
            self.ret_video, self.frame_video = self.putar_sumber_video.main()

        if self.ret_video:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame_video))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)

    def run_gui_video(self):
        ret, frame = self.putar_sumber_video.main()
        if ret and not self.pause:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.frame_3.after(self.delay, self.run_gui_video)
        else:
            self.pause = False
    
    ### BUTTON PLAY
    def gui_button(self):
        self.frame_3_btn = tk.Frame(self.frame_3)
        self.frame_3_btn.pack(fill=tk.X, pady=(10, 15))

        self.button_1 = button_1 = tk.Button(self.frame_3_btn, text='Play', width=25, command=self.play_gui_video)
        button_1.pack(side=tk.LEFT, padx=(0, 10))
        
        self.button_3 = button_3 = tk.Button(self.frame_3_btn, text='Pause / Resume', width=25, command=self.pause_gui_video)
        button_3.pack(side=tk.LEFT, padx=(0, 10))
        
        self.button_4 = button_4 = tk.Button(self.frame_3_btn, text='Stop', width=25, command=self.stop_gui_video)
        button_4.pack(side=tk.LEFT, padx=(0, 10))
        
        self.button_2 = button_2 = tk.Button(self.frame_3_btn, text='Next Frame', width=25, command=lambda: self.refresh_gui_video(reload=True))
        button_2.pack(side=tk.LEFT, padx=(0, 10))

    ### Reload untuk configurasi line
    def refresh_gui_video(self, reload=False):
        # self.canvas.delete('all')
        self.canvas.destroy()
        self.frame_3.destroy()
        self.set_putar_sumber_video_config()
        self.gui_frame_3()
        self.gui_video(reload=reload)
        self.gui_button()
    
    def play_gui_video(self):
        self.refresh_gui_video(reload=True)
        self.run_gui_video()

    def pause_gui_video(self):
        if self.pause == False:
            self.pause = True
        else:
            self.pause = False
            self.play_gui_video()

    def stop_gui_video(self):
        self.pause_gui_video()
        self.putar_sumber_video = PutarSumberVideo()
        self.refresh_gui_video(reload=True)

    def set_putar_sumber_video_config(self):
        self.putar_sumber_video.set_config(video_source=self.source_vid)
