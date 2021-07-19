import os
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import pickle
import math
from tqdm import tqdm
from random import shuffle
from conv_layer import ConvLayer
from max_pool_layer import MaxPoolLayer
from dense_layer import DenseLayer
from adam_optim import AdamOptim

class CNN:
    def __init__(self, architecture = False, model = False, train_data = 'dataset/train/', test_data = 'dataset/test/', saved_model='saved_model/', model_name='model', nama_kendaraan = []):
        self.architecture = architecture
        self.i_adam = 0
        self.train_data = train_data
        self.test_data = test_data
        self.saved_model = saved_model
        self.model_name = model_name + '.pckl'
        # nama_kendaraan  = ['sepeda motor', 'sepeda', 'mobil', 'bus']
        self.nama_kendaraan = nama_kendaraan

        if model == False:
            self.obj = self.init_obj_layer()
        else:
            self.obj = model

    def init_obj_layer(self):
        obj_layer = []
        for i, arch in enumerate(self.architecture):
            layer_type = arch['layer_type']
            if layer_type == 'conv':
                obj = ConvLayer(filter_len=arch['filter_len'], kernel_size=arch['kernel_size'], stride=arch['stride'], padding=arch['padding'], activation=arch['activation'])

            elif layer_type == 'max_pool':
                obj = MaxPoolLayer(pool_size=arch['pool_size'], padding=arch['padding'], stride=arch['stride'])

            elif layer_type == 'flatten':
                obj = 'flatten'

            elif layer_type == 'dense':
                obj = DenseLayer(inp_size=arch['input_size'], output_size=arch['output_size'], activation=arch['activation'])
            
            obj_layer.append(obj)
        return obj_layer

    def desc_predict(self, num):
        return self.nama_kendaraan[num]

    def one_hot_label(self, img):
        label = img.split('.')[0]

        arr_hot_label = [0]*len(self.nama_kendaraan)
        index_arr = self.nama_kendaraan.index(label)
        arr_hot_label[index_arr] = 1

        ohl = np.array(arr_hot_label)
        ohl = np.reshape(ohl, (ohl.shape[0], 1))
        return ohl
        
    def train_data_with_label(self):
        train_images = []
        for i in os.listdir(self.train_data):
            path = os.path.join(self.train_data, i)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
            img = cv2.resize(img, (64, 64))
            train_images.append([np.array([img]), self.one_hot_label(i)])
        shuffle(train_images)
        return train_images

    def test_data_with_label(self):    
        test_images = []
        for i in os.listdir(self.test_data):
            path = os.path.join(self.test_data, i)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
            img = cv2.resize(img, (64, 64))
            test_images.append([np.array([img]), self.one_hot_label(i)])
        return test_images

    def shuffle_data(self):
        training_images = self.train_data_with_label()
        tr_img_data = np.array([i[0] for i in training_images])
        tr_lbl_data = np.array([i[1] for i in training_images])
        return tr_img_data, tr_lbl_data

    def update(self):
        params = []
        params_d = []
        obj_selected = []
        for obj in self.obj:
            if obj.__class__.__name__ == 'DenseLayer':
                params.append(obj.weight)
                params.append(obj.bias)
                dw, db = obj.grad_function(obj.d1z)
                params_d.append(dw)
                params_d.append(db)
                obj_select = {'param': obj, 'param_name': 'weight'}
                obj_selected.append(obj_select)
                obj_select = {'param': obj, 'param_name': 'bias'}
                obj_selected.append(obj_select)

            elif obj.__class__.__name__ == 'ConvLayer':
                params.append(obj.filters)
                dk = obj.grad_function(obj.d1z)
                params_d.append(dk)
                obj_select = {'param': obj, 'param_name': 'filters'}
                obj_selected.append(obj_select)
        adam = AdamOptim(params_len=len(params))
        params_updated = adam.update(params.copy(), params_d.copy(), self.i_adam)
        
        for ob, par in zip(obj_selected, params_updated):
            setattr(ob['param'], ob['param_name'], par)

    def cross_entropy(self, actual, predicted):
        predicted = predicted.clip(min=1e-8,max=None)
        loss2 = np.where(actual==1, -np.sum(actual*np.log(predicted)), 0)
        # loss = -np.sum(actual*np.log(predicted))
        loss = np.sum(loss2)
        return loss, loss2

    def backward_cross_entropy(self, predicted, actual):
        predicted = predicted.clip(min=1e-8,max=None)
        # dy = predicted - actual
        dy = np.where(actual==1,-1/predicted, 0)
        return dy

    def train(self, epochs = 1):
        input, output = self.shuffle_data()
        true_values  = []
        predictions = []
        losses = []
        tloss = 0
        taccuracy = 0

        history = [[], []]

        for ep in tqdm(range(epochs), desc='Epochs', leave=False):
            true_values  = []
            predictions = []
            losses = []

            input, output = self.shuffle_data()
            tqdmit = tqdm(range(output.shape[0]), desc='Iteration', leave='False')
            self.i_adam = 0
            for it in tqdmit:
                self.i_adam += 1
                inp = np.array(input[it],copy=True, dtype=np.float64)
                
                # print('Forward Prop...')
                for i, obj in enumerate(self.obj):
                    if obj == 'flatten':
                        inp = np.reshape(inp, (math.prod(inp.shape), 1))
                    else:
                        obj.set_inp(inp)
                        inp = obj.forward()
                
                loss, loss2 = self.cross_entropy(output[it], inp)
                
                dz = self.backward_cross_entropy(inp, output[it])


                ### count total loss
                losses.append(loss)
                tloss = sum(losses)/len(losses)
                ### count total accuracy
                true_values.append(np.argmax(output[it].flatten()))
                predictions.append(np.argmax(inp.flatten()))

                N = len(true_values)
                taccuracy = (np.array(true_values) == np.array(predictions)).sum() / N
                ###

                # tqdmit.set_description('Loss: '+  str(tloss)+ ' Accuracy: '+ str(accuracy))
                print('\n\n')
                print('=====================================================')
                print('loss: ', loss, ' accuracy: ', taccuracy)
                print('actual: ', self.desc_predict(np.argmax(output[it].flatten())), ' predicted: ', self.desc_predict(np.argmax(inp.flatten())))
                              
                for i, obj in reversed(list(enumerate(self.obj))):
                    if obj == 'flatten':
                        dz = np.reshape(dz, self.obj[i-1].output.shape)
                    else:
                        obj.d1z = dz
                        if i == 0:
                            break
                        dz = obj.backward(dz)

                # print('Update Params...')
                self.update()

            # Saving the objects:
            f = open(self.saved_model+self.model_name, 'wb')
            pickle.dump(self.obj, f)
            f.close()

            file_object = open('history.txt', 'a')
            # Append 'hello' at the end of file
            file_object.write('Loss: '+str(tloss))
            file_object.write('\tAccuracy: '+str(taccuracy))
            file_object.write('\n')
            # Close the file
            file_object.close()
        print('Pelatihan Model Selesai!')
        return 'Pelatihan Model Selesai!'

    def test(self):        
        true_values  = []
        predictions = []
        losses = []
        tloss = 0
        taccuracy = 0

        conf_matrix = ['Conf Matrix']+ [0]*len(self.nama_kendaraan)    
        conf_matrix[0] = ['Conf Matrix']+ self.nama_kendaraan
        for nk in range(len(self.nama_kendaraan)):
            conf_matrix[nk+1] = [self.nama_kendaraan[nk]]+ [0]*len(self.nama_kendaraan)
        
        data = self.test_data_with_label()
        input = np.array([i[0] for i in data])
        output = np.array([i[1] for i in data])

        tqdmit = tqdm(range(output.shape[0]), desc='Iteration', leave='False')
        for it in tqdmit:
            inp = np.array(input[it],copy=True, dtype=np.float64)
            # print('Forward Prop...')
            for i, obj in enumerate(self.obj):
                if obj == 'flatten':
                    inp = np.reshape(inp, (math.prod(inp.shape), 1))
                else:
                    obj.set_inp(inp)
                    inp = obj.forward()

            loss, loss2 = self.cross_entropy(output[it], inp)
            dz = self.backward_cross_entropy(inp, output[it])
                        
            ### count total loss
            losses.append(loss)
            tloss = sum(losses)/len(losses)
            ### count total accuracy
            true_values.append(np.argmax(output[it].flatten()))
            predictions.append(np.argmax(inp.flatten()))

            N = len(true_values)
            taccuracy = (np.array(true_values) == np.array(predictions)).sum() / N
            ### count conf matrix
            conf_matrix[np.argmax(output[it].flatten())+1][np.argmax(inp.flatten())+1] += 1

            # tqdmit.set_description('Loss: '+  str(tloss)+ ' Accuracy: '+ str(accuracy))
            print('\n\n')
            print('=====================================================')
            print('loss: ', loss, ' accuracy: ', taccuracy)
            print('actual: ', self.desc_predict(np.argmax(output[it].flatten())), ' predicted: ', self.desc_predict(np.argmax(inp.flatten())))

        print('\n\nLoss: '+str(tloss))
        print('\tAccuracy: '+str(taccuracy))
        print('\n')
        print(conf_matrix)
        print('Pengujian Model Selesai!')
        return 'Pengujian Model Selesai!'

    def predict(self, img):
        img = cv2.resize(img, (64, 64))
        inp = np.array([img / 255])

        for i, obj in enumerate(self.obj):
            if obj == 'flatten':
                inp = np.reshape(inp, (math.prod(inp.shape), 1))
            else:
                obj.set_inp(inp)
                inp = obj.forward()

        return self.desc_predict(np.argmax(inp.flatten()))

    def visualize(self, epochs = None, loss  = None, accuracy = None, label='Training'):
        loss = [2.924855142739167, 2.509765277930861, 1.8966760156228928, 1.5804372695196718, 1.438682270865215, 1.086320492557087, 1.226886596305873, 1.1932683191127285, 0.9674412680396102, 0.8487963364575677, 0.7887939870358096, 0.7274058810240308, 0.7484065413697386, 0.7082279100840955, 0.6494763333093001, 0.5748012295403986, 0.7498089374597057, 0.6660268658015341, 0.5598664905876192, 0.542726378364782, 0.591944444783053]
        accuracy = [0.5155875299760192, 0.697841726618705, 0.7649880095923262, 0.8129496402877698, 0.8177458033573142, 0.8776978417266187, 0.8800959232613909, 0.9040767386091128, 0.9112709832134293, 0.9136690647482014, 0.9280575539568345, 0.9184652278177458, 0.9256594724220624, 0.9448441247002398, 0.9424460431654677, 0.9400479616306955, 0.9280575539568345, 0.9400479616306955, 0.9376498800959233, 0.9520383693045563, 0.9424460431654677]
        epochs = range(len(loss))
        
        plt.plot(epochs, loss, 'red', label=label+' loss')
        plt.plot(epochs, accuracy, 'green', label=label+'accuracy')
        plt.title(label+' loss and accuracy')
        plt.xlabel('Epochs')
        plt.xticks(epochs)
        plt.legend()
        plt.show()

# mobil penumpang: 139
# sepeda motor: 139
# sepeda : 139
