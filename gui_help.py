import tkinter as tk

class GUIHelp:
    def __init__(self, main_frame):
        self.main_frame = main_frame
        self.root = tk.Tk()
        self.gui_frame_inp_1()
        self.gui_about()
        self.root.mainloop()

    ### FRAME 1.1 - INPUT HELP
    def gui_frame_inp_1(self):
        self.frame_inp_1 = tk.Frame(self.root)
        self.frame_inp_1.pack(fill=tk.X, pady=(0, 20))
        
    def gui_about(self):
        label = tk.Label(self.frame_inp_1, text="Pengembangan Sistem Klasifikasi dan Perhitungan Kendaraan di Dinas Perhubungan Provinsi DKI Jakarta ")
        label.pack(side=tk.LEFT, padx=10)