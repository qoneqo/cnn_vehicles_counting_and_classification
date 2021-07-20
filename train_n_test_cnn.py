from cnn import CNN
import pickle
import os


# nk = ['sepeda motor', 'sepeda', 'mobil penumpang']
# f = open('saved_model/nk-model-2.pckl', 'wb')
# pickle.dump(nk, f)
# f.close()

class TrainNTestCNN:
    def __init__(self, model_name, model_folder, train_folder, test_folder, jumlah_kendaraan, nama_kendaraan):
        self.model_name = model_name
        self.model_folder = model_folder
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.jumlah_kendaraan = jumlah_kendaraan
        self.nama_kendaraan = nama_kendaraan

    def set_architecture(self):
        self.architecture = [
            {'layer_type': 'conv',      'filter_len': 2,        'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
            {'layer_type': 'max_pool',  'pool_size': 2,    'stride': 2, 'padding': 0},
            
            {'layer_type': 'conv',      'filter_len': 6,        'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
            {'layer_type': 'max_pool',  'pool_size': 2,    'stride': 2, 'padding': 0},

            {'layer_type': 'conv',      'filter_len': 8,        'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
            {'layer_type': 'max_pool',  'pool_size': 2,    'stride': 2, 'padding': 0},

            {'layer_type': 'flatten'},
            
            {'layer_type': 'dense',     'input_size': 512,   'output_size': 256,   'activation': 'relu'},
            {'layer_type': 'dense',     'input_size': 256,   'output_size': 27,   'activation': 'relu'},
            {'layer_type': 'dense',     'input_size': 27,   'output_size': self.jumlah_kendaraan,   'activation': 'softmax'},
        ]

    def train(self, epochs=20):
        self.set_architecture()
        cnn = CNN(architecture=self.architecture, train_data=self.train_folder, saved_model=self.model_folder, model_name=self.model_name, nama_kendaraan=self.nama_kendaraan)
        ret = cnn.train(epochs=epochs)
        return ret

    def test(self, model_src):
        f = open(model_src, 'rb')
        model = pickle.load(f)
        f.close()
        cnn = CNN(model=model, test_data=self.test_folder, nama_kendaraan=self.nama_kendaraan)
        ret = cnn.test()
        return ret