import tkinter as tk
from tkinter import ttk
import math
from train_n_test_cnn import TrainNTestCNN
import time

"""
Nama Model
Pilih Folder Output Model
Pilih Folder Data Latih
Pilih Folder Data Uji
Nama Kendaraan Max 6 Kendaraan
"""

class GUICreateModel:
    def __init__(self, main_frame):
        self.main_frame = main_frame
        self.gui_frame_inp_11()
        self.gui_nama_model()

        self.gui_frame_inp_12()
        self.gui_pilih_folder_output()

        self.gui_frame_inp_21()
        self.gui_pilih_folder_latih()

        self.gui_frame_inp_22()
        self.gui_pilih_folder_uji()

        self.gui_frame_inp_31()
        self.jumlah_kendaraan = 3
        # self.gui_jumlah_kendaraan()

        # separator = ttk.Separator(self.frame_inp_3, orient='horizontal')
        # separator.pack(side='top', fill='x')
        
        self.gui_frame_inp_41()
        self.gui_jenis_kendaraan()

        self.gui_frame_inp_61()
        self.gui_epochs()
        
        self.gui_button()
        self.gui_label_finish()

    ### FRAME 1.1 - INPUT NAMA MODEL
    def gui_frame_inp_11(self):
        self.frame_inp_1 = tk.Frame(self.main_frame)
        self.frame_inp_1.pack(fill=tk.X, pady=(0, 20))
        self.frame_inp_11 = tk.LabelFrame(self.frame_inp_1, text='Nama Model')
        self.frame_inp_11.pack(fill=tk.X, side=tk.LEFT, padx=(0, 20))

    def gui_nama_model(self):
        self.gui_label_nama_model()
        self.gui_entry_nama_model()

    def gui_label_nama_model(self):
        label = tk.Label(self.frame_inp_11, text="Input Nama Model: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_nama_model(self):
        self.entry_nama_model = entry_nama_model = tk.Entry(self.frame_inp_11)
        entry_nama_model.pack(side=tk.LEFT, padx=10)
        self.nama_model = default_source = 'model-5'    

        entry_nama_model.insert(0, str(self.nama_model))
        entry_nama_model.bind('<KeyRelease>', self.ev_entry_nama_model)

    def ev_entry_nama_model(self, event):
        self.nama_model = self.entry_nama_model.get()

    ### FRAME 1.2 - PILIH FOLDER OUTPUT MODEL
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

        self.pilih_folder_output = default_folder_output = 'saved_model/'
        entry_pilih_folder_output.insert(0, default_folder_output)

        entry_pilih_folder_output.bind('<KeyRelease>', self.ev_entry_pilih_folder_output)

    def ev_entry_pilih_folder_output(self, event):
        self.pilih_folder_output = self.entry_pilih_folder_output.get()

    def gui_btn_pilih_folder_output(self):        
        self.btn_pilih_folder_output = btn_pilih_folder_output = tk.Button(self.frame_inp_12, text='Browse', command=self.ev_btn_pilih_folder_output)
        btn_pilih_folder_output.pack(side=tk.LEFT, padx=(0, 10))

    def ev_btn_pilih_folder_output(self):
        file_name = tk.filedialog.askdirectory(parent=self.frame_inp_12, title='Plih folder output', initialdir='saved_model/')
        self.entry_pilih_folder_output.delete(0, tk.END)
        self.entry_pilih_folder_output.insert(0, file_name)
        self.entry_pilih_folder_output.xview(len(file_name))
        self.pilih_folder_output = file_name


    ### FRAME 2.1 - PILIH FOLDER DATA LATIH
    def gui_frame_inp_21(self):
        self.frame_inp_2 = tk.Frame(self.main_frame)
        self.frame_inp_2.pack(fill=tk.X, pady=(0, 20))
        self.frame_inp_21 = tk.LabelFrame(self.frame_inp_2, text='Folder Data Latih')
        self.frame_inp_21.pack(fill=tk.X, side=tk.LEFT, padx=(0, 20))

    def gui_pilih_folder_latih(self):
        self.gui_label_pilih_folder_latih()
        self.gui_entry_pilih_folder_latih()
        self.gui_btn_pilih_folder_latih()

    def gui_label_pilih_folder_latih(self):
        label = tk.Label(self.frame_inp_21, text="Pilih Folder Data Latih: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_pilih_folder_latih(self):
        self.entry_pilih_folder_latih = entry_pilih_folder_latih = tk.Entry(self.frame_inp_21)
        entry_pilih_folder_latih.pack(side=tk.LEFT, padx=10)

        self.pilih_folder_latih = default_folder_latih = 'dataset/train/'
        entry_pilih_folder_latih.insert(0, default_folder_latih)

        entry_pilih_folder_latih.bind('<KeyRelease>', self.ev_entry_pilih_folder_latih)

    def ev_entry_pilih_folder_latih(self, event):
        self.pilih_folder_latih = self.entry_pilih_folder_latih.get()

    def gui_btn_pilih_folder_latih(self):        
        self.btn_pilih_folder_latih = btn_pilih_folder_latih = tk.Button(self.frame_inp_21, text='Browse', command=self.ev_btn_pilih_folder_latih)
        btn_pilih_folder_latih.pack(side=tk.LEFT, padx=(0, 10))

    def ev_btn_pilih_folder_latih(self):
        file_name = tk.filedialog.askdirectory(parent=self.frame_inp_21, title='Plih folder data latih', initialdir='dataset/train/')
        self.entry_pilih_folder_latih.delete(0, tk.END)
        self.entry_pilih_folder_latih.insert(0, file_name)
        self.entry_pilih_folder_latih.xview(len(file_name))
        self.pilih_folder_latih = file_name


    ### FRAME 2.2 - PILIH FOLDER DATA UJI
    def gui_frame_inp_22(self):
        self.frame_inp_22 = tk.LabelFrame(self.frame_inp_2, text='Folder Data Uji')
        self.frame_inp_22.pack(fill=tk.X, side=tk.LEFT)

    def gui_pilih_folder_uji(self):
        self.gui_label_pilih_folder_uji()
        self.gui_entry_pilih_folder_uji()
        self.gui_btn_pilih_folder_uji()

    def gui_label_pilih_folder_uji(self):
        label = tk.Label(self.frame_inp_22, text="Pilih Folder Data Uji: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_pilih_folder_uji(self):
        self.entry_pilih_folder_uji = entry_pilih_folder_uji = tk.Entry(self.frame_inp_22)
        entry_pilih_folder_uji.pack(side=tk.LEFT, padx=10)

        self.pilih_folder_uji = default_folder_uji = 'dataset/test/'
        entry_pilih_folder_uji.insert(0, default_folder_uji)

        entry_pilih_folder_uji.bind('<KeyRelease>', self.ev_entry_pilih_folder_uji)

    def ev_entry_pilih_folder_uji(self, event):
        self.pilih_folder_uji = self.entry_pilih_folder_uji.get()

    def gui_btn_pilih_folder_uji(self):        
        self.btn_pilih_folder_uji = btn_pilih_folder_uji = tk.Button(self.frame_inp_22, text='Browse', command=self.ev_btn_pilih_folder_uji)
        btn_pilih_folder_uji.pack(side=tk.LEFT, padx=(0, 10))

    def ev_btn_pilih_folder_uji(self):
        file_name = tk.filedialog.askdirectory(parent=self.frame_inp_22, title='Plih folder data uji', initialdir='dataset/test/')
        self.entry_pilih_folder_uji.delete(0, tk.END)
        self.entry_pilih_folder_uji.insert(0, file_name)
        self.entry_pilih_folder_uji.xview(len(file_name))
        self.pilih_folder_uji = file_name

    ### FRAME 3.1 - INPUT JUMLAH KENDARAAN
    def gui_frame_inp_31(self):
        # self.frame_inp_3 = tk.LabelFrame(self.main_frame, text='Input Jumlah & Jenis Kendaraan')
        self.frame_inp_3 = tk.LabelFrame(self.main_frame, text='Input Jenis Kendaraan')
        self.frame_inp_3.pack(fill=tk.X, pady=(0, 20))
        self.frame_inp_31 = tk.Frame(self.frame_inp_3)
        self.frame_inp_31.pack(fill=tk.X, padx=(0, 20))
        # self.frame_inp_31.pack(fill=tk.X, padx=(0, 20), pady=10)

    def gui_jumlah_kendaraan(self):
        self.gui_label_jumlah_kendaraan()
        self.gui_entry_jumlah_kendaraan()
        self.label_warn_jumlah_kendaraan = label = tk.Label(self.frame_inp_31, text="Minimal Jumlah Kendaraan = 2 dan Maksimal Jumlah Kendaraan = 10 !", fg="yellow")
        label.pack(side=tk.LEFT, padx=10)

    def gui_label_jumlah_kendaraan(self):
        label = tk.Label(self.frame_inp_31, text="Input Jumlah Kendaraan: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_jumlah_kendaraan(self):
        self.entry_jumlah_kendaraan = entry_jumlah_kendaraan = tk.Entry(self.frame_inp_31)
        entry_jumlah_kendaraan.pack(side=tk.LEFT, padx=10)
        self.jumlah_kendaraan = default_source = '3'    

        entry_jumlah_kendaraan.insert(0, str(self.jumlah_kendaraan))
        entry_jumlah_kendaraan.bind('<KeyRelease>', self.ev_entry_jumlah_kendaraan)

    def ev_entry_jumlah_kendaraan(self, event):
        self.jumlah_kendaraan = self.entry_jumlah_kendaraan.get()
        
        if int(self.jumlah_kendaraan) > 10:
            self.jumlah_kendaraan = '10'
            self.entry_jumlah_kendaraan.delete(0, tk.END)
            self.entry_jumlah_kendaraan.insert(0, str(self.jumlah_kendaraan))
        elif int(self.jumlah_kendaraan) < 2:
            self.jumlah_kendaraan = '2'
            self.entry_jumlah_kendaraan.delete(0, tk.END)
            self.entry_jumlah_kendaraan.insert(0, str(self.jumlah_kendaraan))

        self.frame_inp_4.destroy()
        self.gui_frame_inp_41()
        self.gui_jenis_kendaraan()

        ### FRAME 4.1 - INPUT JENIS KENDARAAN
    def gui_frame_inp_41(self):
        self.frame_inp_4 = tk.Frame(self.frame_inp_3)
        self.frame_inp_4.pack(fill=tk.X)
        
        self.frame_inp_4x = [None] * int(self.jumlah_kendaraan)
        self.entry_jenis_kendaraan = [None] * int(self.jumlah_kendaraan)
        self.jenis_kendaraan = [None] * int(self.jumlah_kendaraan)

        for i in range(math.ceil(int(self.jumlah_kendaraan) / 2)):
            self.frame_inp_4x[i] = tk.Frame(self.frame_inp_4)
            self.frame_inp_4x[i].pack(fill=tk.X, padx=(0, 20), pady=10)

    def gui_jenis_kendaraan(self):
        for i in range(int(self.jumlah_kendaraan)):
            self.gui_label_jenis_kendaraan(len=i)
            self.gui_entry_jenis_kendaraan(len=i)

    def gui_label_jenis_kendaraan(self, len=0):
        len_frame = math.floor(len / 2)
        label = tk.Label(self.frame_inp_4x[len_frame], text="Input Jenis Kendaraan "+str(len+1)+": ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_jenis_kendaraan(self, len=0):
        len_frame = math.floor(len / 2)
        self.jenis_kendaraan[len] = 'Kendaraan '+str(len+1)
        
        self.entry_jenis_kendaraan[len] = tk.Entry(self.frame_inp_4x[len_frame])
        self.entry_jenis_kendaraan[len].pack(side=tk.LEFT, padx=10)

        self.entry_jenis_kendaraan[len].insert(0, str(self.jenis_kendaraan[len]))
        self.entry_jenis_kendaraan[len].bind('<KeyRelease>', self.ev_entry_jenis_kendaraan)

    def ev_entry_jenis_kendaraan(self, event):
        for i in range(len(self.entry_jenis_kendaraan)):
            self.jenis_kendaraan[i] = self.entry_jenis_kendaraan[i].get()

### FRAME EPOCHS
    def gui_frame_inp_61(self):
        self.frame_inp_6 = tk.Frame(self.main_frame)
        self.frame_inp_6.pack(fill=tk.X, pady=(0, 20))
        self.frame_inp_61 = tk.LabelFrame(self.frame_inp_6, text='Epochs')
        self.frame_inp_61.pack(fill=tk.X, side=tk.LEFT, padx=(0, 20))

    def gui_epochs(self):
        self.gui_label_epochs()
        self.gui_entry_epochs()

    def gui_label_epochs(self):
        label = tk.Label(self.frame_inp_61, text="Input Jumlah Epochs: ")
        label.pack(side=tk.LEFT, padx=10)

    def gui_entry_epochs(self):
        self.entry_epochs = entry_epochs = tk.Entry(self.frame_inp_61)
        entry_epochs.pack(side=tk.LEFT, padx=10)
        self.epochs = default_source = '20'    

        entry_epochs.insert(0, str(self.epochs))
        entry_epochs.bind('<KeyRelease>', self.ev_entry_epochs)

    def ev_entry_epochs(self, event):
        self.epochs = self.entry_epochs.get()

    ### BUTTON LATIH DAN UJI

    def gui_button(self):
        self.frame_5 = tk.Frame(self.main_frame)
        self.frame_5.pack(fill=tk.X, pady=(10, 15))

        self.button_1 = button_1 = tk.Button(self.frame_5, text='Latih Model', width=25, command=self.train)
        button_1.pack(side=tk.LEFT, padx=(0, 10))
        
        self.button_2 = button_2 = tk.Button(self.frame_5, text='Uji Model', width=25, command=self.test)
        button_2.pack(side=tk.LEFT, padx=(0, 10))
    
    def gui_label_finish(self):
        self.frame_6 = tk.Frame(self.main_frame)
        self.frame_6.pack(fill=tk.X, pady=(10, 15))
        self.label_finish = label_finish = tk.Label(self.frame_6, text='', font=("Arial", 12))
        label_finish.pack(side=tk.LEFT, padx=10)

    def train(self):
        cnn = TrainNTestCNN(model_name=self.nama_model, model_folder=self.pilih_folder_output, train_folder=self.pilih_folder_latih, test_folder=self.pilih_folder_uji, jumlah_kendaraan=int(self.jumlah_kendaraan), nama_kendaraan=self.jenis_kendaraan)
        ret = cnn.train(epochs=int(self.epochs))
        self.label_finish.config(text=ret, fg='green2')

    def test(self):
        cnn = TrainNTestCNN(model_name=self.nama_model, model_folder=self.pilih_folder_output, train_folder=self.pilih_folder_latih, test_folder=self.pilih_folder_uji, jumlah_kendaraan=int(self.jumlah_kendaraan), nama_kendaraan=self.jenis_kendaraan)
        model_src = self.pilih_folder_output+'/'+self.nama_model+'.pckl'
        ret = cnn.test(model_src=model_src)
        self.label_finish.config(text=ret, fg='green2')
