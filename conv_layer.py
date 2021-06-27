import numpy as np
import math
from tqdm import tqdm
import random

class ConvLayer:    
    def __init__(self, filter_len, kernel_size, stride = 1, padding  = 1, activation = 'relu'):
        self.filter_len = filter_len
        self.kernel_size = (kernel_size, kernel_size)
        ####
        ## filters = [v][x][y][z]
        ## v = filters, x = input dimension, y = width, z = height
        self.filters = False
        self.stride = stride
        self.activation = activation
        self.padding = padding
        self.output = []
        self.output_w_activation = []
        self.d1z = []

    def set_inp(self, inp):
        self.inp = inp
        self.inp_w_pad = self.set_padding(inp, self.padding)
        if type(self.filters) == bool:
            self.filters = np.random.randn(self.filter_len, inp.shape[0], *self.kernel_size) * np.sqrt(1/(self.kernel_size[0] * self.kernel_size[1]), dtype="float64")

    def set_padding(self, inp, padding):
        if padding > 0:
            output = []
            for i in range(inp.shape[0]):
                output.append(np.pad(inp[i], (padding, padding), 'constant'))
            return np.array(output)
        else:
            return inp
    
    def unpad(self, inp):
        if self.padding != 1:
            return np.array(inp)
        else:
            out = []
            for c in inp:
                c = np.delete(c, -1, 0)
                c = np.delete(c, 0, 0)
                c = np.delete(c, -1, 1)
                c = np.delete(c, 0, 1)
                out.append(c)
            return np.array(out)

    def relu(self, inp):
        return np.maximum(0,inp)

    # cross-correlation
    def forward(self):
        output_size_x = math.floor( (self.inp.shape[1] + (2*self.padding) - self.filters.shape[2]) / self.stride ) + 1
        output_size_y = math.floor( (self.inp.shape[2] + (2*self.padding) - self.filters.shape[3]) / self.stride ) + 1
        output = np.zeros(( self.filters.shape[0], output_size_x, output_size_y))
        inp_w_pad = np.copy(self.inp_w_pad)

        for x in range(self.filters.shape[0]):
            filter = np.array(self.filters[x], copy=True)

            for k in range(output_size_x):
                for l in range(output_size_y):
                    convo = 0

                    for i in range(filter.shape[1]):
                        for j in range(filter.shape[2]):

                            for y in range(self.filters.shape[1]):
                                convo += filter[y][i][j] * self.inp_w_pad[y][k*self.stride+i][l*self.stride+j]
                    
                    output[x][k][l] = convo

        self.output = np.array(output, copy=True)
        output_w_activation = self.relu(output)
        self.output_w_activation = np.array(output_w_activation, copy=True)
        return output_w_activation        

    def backward_relu(self, dz):
        dz = np.array(dz, copy = True)
        # dz[self.output <= 0] = 0
        dr = np.where(self.output>0, dz, 0)
        return dr
    
    def backward_filters(self, da):
        dfilters = np.zeros((self.filters.shape))

        for x in range(da.shape[0]):
            # da.shape[0] === self.filters.shape[0] 

            for y in range(self.inp_w_pad.shape[0]):
                # da.inp_w_pad.shape[0] === self.filters.shape[1]
                inp_w_pad = np.array(self.inp_w_pad, copy=True)

                for k in range(self.filters.shape[2]):
                    for l in range(self.filters.shape[3]):
                        convo = 0
                        for i in range(da.shape[1]):
                            for j in range(da.shape[2]):
                                convo +=  inp_w_pad[y][k * self.stride + i][l * self.stride + j] * da[x][i][j]
                        dfilters[x][y][k][l] = convo
        
        dfilters = np.array(dfilters, copy=True)
        return dfilters
                        
    def backward_input(self, da):
        inp = np.zeros((self.inp_w_pad.shape))
        filters180 = np.rot90(np.array(self.filters, copy=True), 2, axes=(2,3))

        pad_size = filters180.shape[2] - 1
        da_w_pad = np.pad(np.array(da, copy=True), pad_width=((0,0),(pad_size,pad_size),(pad_size,pad_size)), mode='constant')

        irand = random.randint(0, da.shape[0]-1)

        for x in range(self.inp_w_pad.shape[0]):

            for k in range(self.inp_w_pad.shape[1]):
                for l in range(self.inp_w_pad.shape[2]):
                    convo = 0
                    for i in range(filters180.shape[2]):
                        for j in range(filters180.shape[3]):
                            convo += da_w_pad[irand][k * self.stride + i][l * self.stride + j] * filters180[irand][x][i][j]
                    inp[x][k][l] = convo

        if self.padding:
            dinp = self.unpad(inp)

        return dinp

    def backward(self, da):
        dc = self.backward_relu(da)
        di = self.backward_input(dc)
        return di
    
    def grad_function(self, dz):
        dk = self.backward_filters(dz)
        return dk


# conv = ConvLayer(x, kernel_dim=(2,2), kernel_len=2, padding=1)
