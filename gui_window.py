import tkinter as tk
from gui_counting import *
from gui_create_dataset import *
from gui_create_model import *
from gui_putar_sumber_video import *
from gui_help import *

class GUIWindow:
    def __init__(self):
        self.window = window = self.create_window()
        # run self method create menu
        self.main_frame = main_frame = tk.Frame(window)
        main_frame.pack()
        self.menu(window)
        self.reload_main_frame(GUICounting, 'Sistem Klasifikasi dan Perhitungan Kendaraan')
        window.mainloop()

    def reload_main_frame(self, class_gui, window_title):
        for child in self.main_frame.winfo_children():
            child.destroy()
        class_gui(self.main_frame)
        self.window.title(window_title)

    def create_window(self):
        window = tk.Tk()
        window.title('Sistem Klasifikasi dan Perhitungan Kendaraan')
        window.geometry("1000x600")
        return window
    
    def about(self):
        root = tk.Tk()
        root.title('About')
        root.geometry("200x200")
        label = tk.Label(root, text="Pengembangan Sistem Klasifikasi dan Perhitungan Kendaraan di Dinas Perhubungan Provinsi DKI Jakarta menggunakan Convolutional Neural Network", wraplength=170)
        label.pack(side=tk.LEFT, padx=10)
        root.mainloop()

    def menu(self, root_window):
        rootWindow = root_window
        menu = tk.Menu()
        rootWindow.config(menu=menu)

        filemenu = tk.Menu(menu)
        menu.add_cascade(label='Main', menu=filemenu)
        filemenu.add_command(label='Klasifikasi dan Hitung Kendaraan', command=lambda: self.reload_main_frame(GUICounting, 'Sistem Klasifikasi dan Perhitungan Kendaraan'))
        filemenu.add_command(label='Putar Sumber Video', command=lambda: self.reload_main_frame(GUIPutarSumberVideo, 'Putar Sumber Video'))
        filemenu.add_separator()
        filemenu.add_command(label='Exit', command=rootWindow.quit)

        datasetmenu = tk.Menu(menu)
        menu.add_cascade(label='Dataset', menu=datasetmenu)
        datasetmenu.add_command(label='Buat Dataset', command=lambda: self.reload_main_frame(GUICreateDataset, 'Buat Dataset'))

        modelmenu = tk.Menu(menu)
        menu.add_cascade(label='Model', menu=modelmenu)
        modelmenu.add_command(label='Buat Model', command=lambda: self.reload_main_frame(GUICreateModel, 'Buat Model'))

        helpmenu = tk.Menu(menu)
        menu.add_cascade(label='Help', menu=helpmenu)
        helpmenu.add_command(label='About', command=self.about)
