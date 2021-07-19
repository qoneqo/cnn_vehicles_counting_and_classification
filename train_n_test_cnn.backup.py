from cnn import CNN
import pickle
import os

architecture = [
    {'layer_type': 'conv',      'filter_len': 2,        'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_size': 2,    'stride': 2, 'padding': 0},
    
    {'layer_type': 'conv',      'filter_len': 6,        'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_size': 2,    'stride': 2, 'padding': 0},

    {'layer_type': 'conv',      'filter_len': 8,        'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_size': 2,    'stride': 2, 'padding': 0},

    {'layer_type': 'flatten'},
    
    {'layer_type': 'dense',     'input_size': 512,   'output_size': 256,   'activation': 'relu'},
    {'layer_type': 'dense',     'input_size': 256,   'output_size': 27,   'activation': 'relu'},
    {'layer_type': 'dense',     'input_size': 27,   'output_size': 3,   'activation': 'softmax'},
]

f = open('saved_model/model-2.pckl', 'rb')
model = pickle.load(f)
f.close()
cnn = CNN(model=model)
# cnn.train(epochs=20)
cnn.test()
# cnn.visualize()




# x = w,x,y,z
# y = x,y,z

