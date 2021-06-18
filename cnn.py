from conv_layer import *
from max_pool_layer import *
from dense_layer import *
from adam_optim import *
import pickle
import os
from tqdm import tqdm
from random import shuffle
from cv2 import cv2

train_data = 'dataset/train'
test_data = 'dataset/test'

def desc_predict(num):
    label = ''
    if num == 0:
        label = 'sepeda motor'    
    elif num == 1:
        label = 'sepeda' 
    elif num == 2:
        label = 'mobil penumpang'
    elif num == 3:
        label = 'mobil barang'
    return label

def one_hot_label(img):
    label = img.split('.')[0]
    ohl = np.array([0, 0, 0, 0])
    if label == 'sepeda-motor':
        ohl = np.array([1, 0, 0, 0])
    elif label == 'sepeda':
        ohl = np.array([0, 1, 0, 0])
    elif label == 'mobil-penumpang':
        ohl = np.array([0, 0, 1, 0])
    elif label == 'mobil-barang':
        ohl = np.array([0, 0, 0, 1])
    ohl = np.reshape(ohl, (ohl.shape[0], 1))
    return ohl
    
def train_data_with_label():
    train_images = []
    for i in os.listdir(train_data):
        path = os.path.join(train_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
        img = cv2.resize(img, (64, 64))
        train_images.append([np.array([img]), one_hot_label(i)])
    shuffle(train_images)
    return train_images

def test_data_with_label():    
    test_images = []
    for i in os.listdir(test_data):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
        img = cv2.resize(img, (64, 64))
        test_images.append([np.array([img]), one_hot_label(i)])
    shuffle(test_images)
    return test_images
    
training_images = train_data_with_label()
testing_images = test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images])
tr_lbl_data = np.array([i[1] for i in training_images])
tst_img_data = np.array([i[0] for i in testing_images])
tst_lbl_data = np.array([i[1] for i in testing_images])


class CNN:
    def __init__(self, architecture, inp, model = False):
        self.architecture = architecture
        self.inp = inp
        self.d = []
        self.true_values = []
        self.predictions = []
        self.i_adam = 0
        
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
        self.i_adam += 1
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


    def train(self, out, epochs = 1):
        for ep in tqdm(range(epochs), desc='Epochs', leave=False):
            self.true_values = []
            self.predictions = []
            self.tloss = []
            
            tqdmit = tqdm(range(out.shape[0]), desc='Iteration', leave='False')
            for it in tqdmit:
                inp = np.array(self.inp[it],copy=True, dtype=np.float128)
                
                # print('Forward Prop...')
                for i, obj in enumerate(self.obj):
                    if obj == 'flatten':
                        inp = np.reshape(inp, (math.prod(inp.shape), 1))
                        self.d = np.random.rand(inp.shape[0], inp.shape[1])
                    else:
                        obj.set_inp(inp)
                        inp = obj.forward()
                
                loss, loss2 = cnn.cross_entropy(out[it], inp)
                dz = cnn.backward_cross_entropy(inp, out[it])
                self.tloss.append(loss)
                tloss = sum(self.tloss)/len(self.tloss)

                ### count accuracy
                self.true_values.append(np.argmax(out[it].flatten()))
                self.predictions.append(np.argmax(inp.flatten()))

                true_values = np.array(self.true_values, copy=True)
                predictions = np.array(self.predictions, copy=True)

                N = len(true_values)
                accuracy = (true_values == predictions).sum() / N

                # tqdmit.set_description('Loss: '+  str(tloss)+ ' Accuracy: '+ str(accuracy))
                print('\n\n')
                print('=====================================================')
                print('loss: ', loss, ' accuracy: ', accuracy)
                print('actual: ', desc_predict(np.argmax(out[it].flatten())), ' predicted: ', desc_predict(np.argmax(inp.flatten())))
                # if ep == epochs-1 and it == out.shape[0]-1:
                #     break
                # print('Backward Prop...')
                for i, obj in reversed(list(enumerate(self.obj))):
                    if obj == 'flatten':
                        dz = np.reshape(dz, self.obj[i-1].output.shape)
                    else:
                        obj.d1z = dz
                        if i == 0:
                            break
                        dz = obj.backward(dz)
                # print(self.obj[4].filters[1][1])

                # print('Update Params...')
                self.update()

            # Saving the objects:
            f = open('model-2.pckl', 'wb')
            pickle.dump(self.obj, f)
            f.close()

            # print(self.obj[4].filters[1][1])

    def test(self, out, epochs = 1):
        self.true_values = []
        self.predictions = []
        self.tloss = []
        
        tqdmit = tqdm(range(out.shape[0]), desc='Iteration', leave='False')
        for it in tqdmit:
            inp = np.array(self.inp[it],copy=True, dtype=np.float128)
            
            # print('Forward Prop...')
            for i, obj in enumerate(self.obj):
                if obj == 'flatten':
                    inp = np.reshape(inp, (math.prod(inp.shape), 1))
                    self.d = np.random.rand(inp.shape[0], inp.shape[1])
                else:
                    obj.set_inp(inp)
                    inp = obj.forward()
            
            
            loss, loss2 = cnn.cross_entropy(out[it], inp)
            dz = cnn.backward_cross_entropy(inp, out[it])
            self.tloss.append(loss)
            tloss = sum(self.tloss)/len(self.tloss)

            ### count accuracy
            self.true_values.append(np.argmax(out[it].flatten()))
            self.predictions.append(np.argmax(inp.flatten()))

            true_values = np.array(self.true_values, copy=True)
            predictions = np.array(self.predictions, copy=True)

            N = len(true_values)
            accuracy = (true_values == predictions).sum() / N

            # tqdmit.set_description('Loss: '+  str(tloss)+ ' Accuracy: '+ str(accuracy))
            print('\n\n')
            print('=====================================================')
            print('loss: ', loss, ' accuracy: ', accuracy)
            print('actual: ', desc_predict(np.argmax(out[it].flatten())), ' predicted: ', desc_predict(np.argmax(inp.flatten())))
            

    def predict(self, img):
        img = cv2.resize(img, (64, 64))
        inp = np.array([img / 255])

        for i, obj in enumerate(self.obj):
            if obj == 'flatten':
                inp = np.reshape(inp, (math.prod(inp.shape), 1))
                self.d = np.random.rand(inp.shape[0], inp.shape[1])
            else:
                obj.set_inp(inp)
                inp = obj.forward()

        return desc_predict(np.argmax(inp.flatten()))
                
        


f = open('model-2.pckl', 'rb')
model = pickle.load(f)
f.close()

# cnn = CNN(False, np.array(tst_img_data), model)
# cnn.test(np.array(tst_lbl_data))



architecture = [
    {'layer_type': 'conv',      'filter_len': 2,        'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_size': 2,    'stride': 2, 'padding': 0},
    
    {'layer_type': 'conv',      'filter_len': 8,        'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_size': 2,    'stride': 2, 'padding': 0},

    {'layer_type': 'conv',      'filter_len': 8,        'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_size': 2,    'stride': 2, 'padding': 0},

    {'layer_type': 'flatten'},
    
    {'layer_type': 'dense',     'input_size': 512,   'output_size': 256,   'activation': 'relu'},
    {'layer_type': 'dense',     'input_size': 256,   'output_size': 27,   'activation': 'relu'},
    {'layer_type': 'dense',     'input_size': 27,   'output_size': 4,   'activation': 'softmax'},
]

# x = w,x,y,z
# y = x,y,z
img1 = cv2.imread('dataset2/sepeda-motor.png', cv2.IMREAD_GRAYSCALE) / 255
img1 = cv2.resize(img1, (64, 64))
img2 = cv2.imread('dataset2/sepeda.png', cv2.IMREAD_GRAYSCALE) / 255
img2 = cv2.resize(img2, (64, 64))
img3 = cv2.imread('dataset2/mobil-penumpang.png', cv2.IMREAD_GRAYSCALE) / 255
img3 = cv2.resize(img3, (64, 64))
img4 = cv2.imread('dataset2/mobil-barang.png', cv2.IMREAD_GRAYSCALE) / 255
img4 = cv2.resize(img4, (64, 64))
x = np.array([
    np.array([img1]),
    np.array([img2]),
    np.array([img3]),
    np.array([img4]),
])
y = np.array([
    [[1], [0], [0], [0]],
    [[0], [1], [0], [0]],
    [[0], [0], [1], [0]],
    [[0], [0], [0], [1]]
])     
# cnn = CNN(architecture, np.array(x))
# cnn.train(np.array(y), epochs=50)


cnn = CNN(architecture, np.array(tr_img_data))
cnn.train(np.array(tr_lbl_data), epochs=50)

# mobil barang: 28
# mobil penumpang: 147
# sepeda motor: 139
# sepeda : 139
