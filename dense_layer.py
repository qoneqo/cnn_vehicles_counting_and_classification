import numpy as np
import math

class DenseLayer:
    def __init__(self, inp_size, output_size, activation = 'relu'):        
        self.inp_size = inp_size
        self.output_size = output_size
        self.activation = activation
        self.weight = np.random.randn(output_size, inp_size) * np.sqrt(1/output_size, dtype="float64")
        self.bias = np.full((output_size,1), 0, dtype="float64")
        self.output = []
        self.output_w_activation = []
        self.d1z = []

    def set_inp(self, inp):
        self.inp = inp

    def relu(self, inp):
        return np.maximum(0,inp)

    def softmax(self, inp):
        inp -= np.max(inp)
        z = np.exp(inp) / np.sum(np.exp(inp))
        return z

    def relu_backward(self, dz):
        dz = np.array(dz, copy = True)
        # dz[self.output <= 0] = 0
        dr = np.where(self.output>0, dz, 0)
        return dr

    def softmax_backward(self, dz):
        z = self.output_w_activation
        ds = z * (dz - np.sum(dz * z))
        return ds

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
