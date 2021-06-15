import numpy as np
import math

d_layer = chr(ord('A')-1)

class DenseLayer:
    def __init__(self, inp_size, output_size, activation = 'relu'):        
        self.inp_size = inp_size
        self.output_size = output_size
        self.activation = activation
        self.weight = np.random.rand(output_size, inp_size) 
        self.bias = np.random.rand(output_size, 1)
        self.output = []
        self.output_w_activation = []
        self.layer = self.layer_name()
        self.d1z = []
    def set_inp(self, inp):
        self.inp = inp
        

    def layer_name(self):
        global d_layer
        d_layer = chr(ord(d_layer) + 1) 
        return 'D'+d_layer

    def relu(self, inp):
        return np.maximum(0,inp)

    def softmax(self, inp):
        f = np.exp(inp - np.max(inp))  # shift values
        return f / f.sum(axis=0)

    def relu_backward(self, dz):
        dz = np.array(dz, copy = True)
        dz[self.output <= 0] = 0
        return dz

    def softmax_backward(self, dz):
        s = self.output.reshape(-1,1)
        return np.dot((np.diagflat(s) - np.dot(s, s.T)), dz)

    def forward(self):
        self.output = np.dot(self.weight, self.inp) + self.bias
        if self.activation == 'relu':
            activation = self.relu
        else:
            activation = self.softmax
        self.output_w_activation = activation(self.output)
        return self.output_w_activation

    def backward(self, dz):
        if self.activation == 'relu':
            activation = self.relu_backward
        else:
            activation = self.softmax_backward
        dz = activation(dz)

        dz = np.dot(self.weight.T, dz)
        return dz
    
    def grad_function(self, dz):
        dw = np.dot(dz, self.inp.T)
        db = np.sum(dz, axis=1, keepdims=True)
        return dw, db

# dense = DenseLayer(z, inp_size = z.shape[0], output_size = 55)
