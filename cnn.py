from conv_layer import *
from max_pool_layer import *
from dense_layer import *
from adam_optim import *
from cv2 import cv2
import pickle
from tqdm import tqdm
from random import shuffle
import os

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
        self.i_adam = 0
        self.d = []
        self.true_values = []
        self.predictions = []
        
        if model == False:
            self.obj = self.init_obj_layer()
        else:
            self.obj = model
        
    def init_obj_layer(self):
        obj_layer = []
        for i, arch in enumerate(self.architecture):
            layer_type = arch['layer_type']
            if layer_type == 'conv':
                obj = ConvLayer(filter_len=arch['filter_len'], kernel_size=arch['kernel_size'], stride=arch['stride'], activation=arch['activation'])

            elif layer_type == 'max_pool':
                obj = MaxPoolLayer(pool_size=arch['pool_size'], stride=arch['stride'])

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
        params = adam.update(self.i_adam, params.copy(), params_d.copy())
        
        for ob, par in zip(obj_selected, params):
            setattr(ob['param'], ob['param_name'], par)
            
    def cross_entropy(self, actual, predicted):
        predicted[predicted == 0] = 1e-8
        output = -actual * np.log(predicted)
        # loss = -np.sum(actual * np.log(predicted))
        loss = np.sum(output)
        return loss, output
    
    def backward_cross_entropy(self, softmax_output, hot_vector):
        dy = softmax_output - hot_vector
        return dy 


    def train(self, out, epochs = 1):

        self.true_values = []
        self.predictions = []

        for ep in tqdm(range(epochs), desc='Epochs', leave=False):
            for it in tqdm(range(out.shape[0]), desc='Iteration', leave='False'):
                inp = np.array(self.inp[it],copy=True)
                
                # print('Forward Prop...')
                for i, obj in enumerate(self.obj):
                    if obj == 'flatten':
                        inp = np.reshape(inp, (math.prod(inp.shape), 1))
                        self.d = np.random.rand(inp.shape[0], inp.shape[1])
                    else:
                        obj.set_inp(inp)
                        inp = obj.forward()
                
                
                loss, ce_z = cnn.cross_entropy(out[it], inp)
                dz = cnn.backward_cross_entropy(inp, out[it])
                
                ### count accuracy
                self.true_values.append(np.argmax(out[it].flatten()))
                self.predictions.append(np.argmax(inp.flatten()))

                true_values = np.array(self.true_values, copy=True)
                predictions = np.array(self.predictions, copy=True)

                N = len(true_values)
                accuracy = (true_values == predictions).sum() / N

                print('\n\n')
                print('=====================================================')
                print('loss: ', loss, ' accuracy: ', accuracy)
                print('actual: ', np.argmax(out[it].flatten()), ' predicted: ', np.argmax(inp.flatten()))
                if ep == epochs-1 and it == out.shape[0]-1:
                    break

                # print('Backward Prop...')
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
        f = open('model-5.pckl', 'wb')
        pickle.dump(self.obj, f)
        f.close()



architecture = [
    {'layer_type': 'conv',      'filter_len': 8,        'kernel_size': 3, 'stride': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_size': 3,    'stride': 3},
    
    {'layer_type': 'conv',      'filter_len': 8,        'kernel_size': 3, 'stride': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_size': 3,    'stride': 3},
    
    {'layer_type': 'conv',      'filter_len': 8,        'kernel_size': 3, 'stride': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_size': 3,    'stride': 3},
    
    {'layer_type': 'flatten'},
    
    {'layer_type': 'dense',     'input_size': 32,  'output_size': 12,  'activation': 'relu'},
    {'layer_type': 'dense',     'input_size': 12,   'output_size': 4,   'activation': 'softmax'},
]

# x = cv2.imread('mobil-penumpang.6.png', cv2.IMREAD_GRAYSCALE) / 255
# x = cv2.resize(x, (64, 64))
# x = np.array([x])
# y = np.array([
#     [
#         [0],
#         [0],
#         [1],
#         [0],
#     ]
# ])

# cnn = CNN(architecture, np.array([x]))
# predicted = cnn.train(np.array(y), epochs=100)


f = open('model-4.pckl', 'rb')
obj = pickle.load(f)
f.close()

# x = w,x,y,z
# y = x,y,z
cnn = CNN(architecture, np.array(tr_img_data))
predicted = cnn.train(np.array(tr_lbl_data), epochs=15)


# mobil barang: 28
# mobil penumpang: 147
# sepeda motor: 139
# sepeda : 139

