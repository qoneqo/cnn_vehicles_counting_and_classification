from conv_layer import *
from max_pool_layer import *
from dense_layer import *
from adam_optim import *
from cv2 import cv2
from tqdm import tqdm

class CNN:
    def __init__(self, architecture, inp):
        self.architecture = architecture
        self.inp = inp
        self.obj = self.init_obj_layer()
        
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

    def train(self, out, epochs = 15):
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
            t = 1 

            dw1, db1 = self.obj[4].grad_function(self.obj[4].d1z)
            dw2, db2 = self.obj[3].grad_function(self.obj[3].d1z)
            dk1 = self.obj[0].grad_function(self.obj[0].d1z)

            self.obj[4].weight, self.obj[4].bias, self.obj[3].weight, self.obj[3].bias, self.obj[0].kernel = adam.update(t, w1, b1, dw1, db1, w2, b2, dw2, db2, k1, dk1)


                
                        
                

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
    [1],
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
cnn = CNN(architecture, x)
cnn.train(y)



# ### conv operation example
# c1 = ConvLayer(x, kernel_dim=(2,2), kernel_len=2, padding=1)
# c1_z = c1.forward()
# ### max pool operation example
# p1 = MaxPoolLayer(c1_z, pool_dim = (2,2), padding = 0, stride = 2)
# p1_z = p1.forward()
# ### reshape operation example
# z = np.reshape(p1_z, (math.prod(p1_z.shape), 1))
# ### dense operation example
# f1 = DenseLayer(z, inp_size = z.shape[0], output_size = 55)
# f1_z = f1.forward()
# f2 = DenseLayer(f1_z, inp_size = 55, output_size = 5, activation='softmax')
# f2_z = f2.forward()

# cnn = CNN()
# actual = np.array([
#     [0],
#     [0],
#     [0],
#     [0],
#     [1],
# ])
# loss, ce_z = cnn.cross_entropy(actual, f2_z)

# ce_dz = cnn.backward_cross_entropy(f2_z, actual)
# f2_dz = f2.backward(ce_dz)
# f1_dz = f1.backward(f2_dz)

# z = np.reshape(f1_dz, p1_z.shape)
# p1_dz = p1.backward(z)

# c1_dz = c1.backward(p1_dz)
# print(c1_dz)


# epochs = 20
# learning_rate = 0.1

# # train
# for e in range(epochs):
#     error = 0
#     for x, y in zip(x_train, y_train):
#         # forward
#         output = x
#         for layer in network:
#             output = layer.forward(output)

#         # error
#         error += binary_cross_entropy(y, output)

#         # backward
#         grad = binary_cross_entropy_prime(y, output)
#         for layer in reversed(network):
#             grad = layer.backward(grad, learning_rate)

#     error /= len(x_train)
#     print(f"{e + 1}/{epochs}, error={error}")

# # test
# for x, y in zip(x_test, y_test):
#     output = x
#     for layer in network:
#         output = layer.forward(output)
#     print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
