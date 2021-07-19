import tkinter as tk
from gui_counting import *
from gui_create_dataset import *

class GUIWindow:
    def __init__(self):
        self.window = window = self.create_window()
        # run self method create menu
        self.main_frame = main_frame = tk.Frame(window)
        main_frame.pack()
        self.menu(window)
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

    def menu(self, root_window):
        rootWindow = root_window
        menu = tk.Menu()
        rootWindow.config(menu=menu)

        filemenu = tk.Menu(menu)
        menu.add_cascade(label='Menu', menu=filemenu)
        filemenu.add_command(label='Monitoring Lalu Lintas')
        filemenu.add_command(label='Counting Kendaraan', command=lambda: self.reload_main_frame(GUICounting, 'Sistem Klasifikasi dan Perhitungan Kendaraan'))
        filemenu.add_separator()
        filemenu.add_command(label='Exit', command=rootWindow.quit)

        datasetmenu = tk.Menu(menu)
        menu.add_cascade(label='Dataset', menu=datasetmenu)
        datasetmenu.add_command(label='Buat Dataset', command=lambda: self.reload_main_frame(GUICreateDataset, 'Buat Dataset'))

        modelmenu = tk.Menu(menu)
        menu.add_cascade(label='Model', menu=modelmenu)
        modelmenu.add_command(label='Buat Model')
        modelmenu.add_command(label='Hapus Model')

        helpmenu = tk.Menu(menu)
        menu.add_cascade(label='Help', menu=helpmenu)
        helpmenu.add_command(label='About')

GUIWindow()
