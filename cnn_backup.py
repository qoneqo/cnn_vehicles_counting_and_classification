from conv_layer import *
from max_pool_layer import *
from dense_layer import *
from adam_optim import *
from cv2 import cv2
import pickle
from tqdm import tqdm

class CNN:
    def __init__(self, architecture, inp, model = False):
        self.architecture = architecture
        self.inp = inp
        if model == False:
            self.obj = self.init_obj_layer()
        else:
            self.obj = model
        
    def init_obj_layer(self):
        obj_layer = []
        for i, arch in enumerate(self.architecture):
            layer_type = arch['layer_type']
            if layer_type == 'conv':
                obj = ConvLayer(kernel_len=arch['kernel_len'], kernel_dim=arch['kernel_dim'], stride=arch['stride'], activation=arch['activation'])

            elif layer_type == 'max_pool':
                obj = MaxPoolLayer(pool_dim=arch['pool_dim'], stride=arch['stride'])

            elif layer_type == 'flatten':
                obj = 'flatten'

            elif layer_type == 'dense':
                obj = DenseLayer(inp_size=arch['input_size'], output_size=arch['output_size'], activation=arch['activation'])
            
            obj_layer.append(obj)
        return obj_layer
        

    def cross_entropy(self, actual, predicted):
        output = -actual * np.log(predicted)
        # loss = -np.sum(actual * np.log(predicted))
        loss = np.sum(output)
        return loss, output
    
    def backward_cross_entropy(self, softmax_output, hot_vector):
        dy = softmax_output - hot_vector
        return dy 

    def train(self, out, epochs = 1):
        for ep in tqdm(range(epochs), desc='Epochs'):
            inp = self.inp
            self.dz = []
            
            print('Forward Prop...')
            for i, obj in tqdm(enumerate(self.obj), desc='Forward Progress'):
                if obj == 'flatten':
                    inp = np.reshape(inp, (math.prod(inp.shape), 1))
                else:
                    obj.set_inp(inp)
                    inp = obj.forward()
            
            loss, ce_z = cnn.cross_entropy(out, inp)
            dz = cnn.backward_cross_entropy(inp, out)
            print('==========================================================')
            print('loss: ',loss)
            print('==========================================================')

            print('Backward Prop...')
            for i, obj in reversed(list(enumerate(self.obj))):
                if obj == 'flatten':
                    dz = np.reshape(dz, self.obj[i-1].output.shape)
                else:
                    obj.d1z = dz
                    dz = obj.backward(dz)

            print('Update Params...')
            w1 = self.obj[4].weight
            b1 = self.obj[4].bias
            w2 = self.obj[3].weight
            b2 = self.obj[3].bias
            k1 = self.obj[0].kernel
            adam = AdamOptim()

            dw1, db1 = self.obj[4].grad_function(self.obj[4].d1z)
            dw2, db2 = self.obj[3].grad_function(self.obj[3].d1z)
            dk1 = self.obj[0].grad_function(self.obj[0].d1z)

            self.obj[4].weight, self.obj[4].bias, self.obj[3].weight, self.obj[3].bias, self.obj[0].kernel = adam.update(1, w1, b1, dw1, db1, w2, b2, dw2, db2, k1, dk1)
        # Saving the objects:
        f = open('params.pckl', 'wb')
        pickle.dump(self.obj, f)
        f.close()



x = np.array(
    [
        [[2, 5, 3],
        [2, 1, 2],
        [4, 2, 3]],

        [[1, 3, 3],
        [2, 1, 2],
        [2, 2, 3]],

        [[1, 3, 3],
        [2, 1, 2],
        [2, 2, 3]],
    ]
) 

y = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
])

architecture = [
    {'layer_type': 'conv',      'kernel_len': 2,        'kernel_dim': (2,2), 'stride': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_dim': (2,2),    'stride': 2},
    {'layer_type': 'flatten'},
    {'layer_type': 'dense',     'input_size': 18,  'output_size': 10,  'activation': 'relu'},
    {'layer_type': 'dense',     'input_size': 10,   'output_size': 5,   'activation': 'softmax'},
]

x = cv2.imread('mobil-penumpang.6.png', cv2.IMREAD_GRAYSCALE) / 255
x = cv2.resize(x, (5, 5))
x = np.array([x])


# f = open('params.pckl', 'rb')
# obj = pickle.load(f)
# f.close()

cnn = CNN(architecture, x)
cnn.train(y, epochs=5)

