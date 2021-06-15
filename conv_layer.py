import numpy as np
import math
from tqdm import tqdm
import random

c_layer = chr(ord('A')-1)

class ConvLayer:    
    def __init__(self, kernel_len, kernel_dim, stride = 1, padding  = 1, activation = 'relu'):
        self.kernel_len = kernel_len
        self.kernel_dim = kernel_dim
        self.stride = stride
        self.activation = activation
        self.padding = padding
        self.layer = self.layer_name()
        self.kernel = np.random.rand(kernel_len, *kernel_dim)
        self.output = []
        self.output_w_activation = []
        self.d1z = []

    def set_inp(self, inp):
        self.inp = inp
        self.inp_w_pad = self.set_padding(inp, self.padding)

    def layer_name(self):
        global c_layer
        c_layer = chr(ord(c_layer) + 1)        
        return 'C'+c_layer
    
    def set_padding(self, inp, padding):
        if padding > 0:
            output = []
            for i in range(inp.shape[0]):
                output.append(np.pad(inp[i], (padding, padding), 'constant'))
            return np.array(output)
        else:
            return inp
    
    def unpad(self, inp):
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
        # y_size = floor(((x+2p-f+1)+1) / s)
        output_size_x = math.floor( (self.inp.shape[1] + (2*self.padding) - self.kernel_dim[0]) / self.stride ) + 1
        output_size_y = math.floor( (self.inp.shape[2] + (2*self.padding) - self.kernel_dim[1]) / self.stride ) + 1
        output_depth = 0
        output = np.zeros(( (self.inp.shape[0] * self.kernel_len), output_size_x, output_size_y), dtype=float)
        kernel = np.copy(self.kernel)
        inp_w_pad = np.copy(self.inp_w_pad)

        ### loop for all images
        for z in range(self.inp.shape[0]): 
            for x in range(self.kernel_len):

                for k in range(output_size_x):
                    for l in range(output_size_y):

                        ### loop for strading
                        convo = 0

                        for i in range(self.kernel_dim[0]):
                            for j in range(self.kernel_dim[1]):

                                ### loop for convolution
                                convo += self.kernel[x][i][j] * self.inp_w_pad[z][k*self.stride+i][l*self.stride+j]

                        output[output_depth][k][l]=convo
                output_depth += 1

        self.output = np.array(output, copy=True)
        output_w_activation = self.relu(output)
        self.output_w_activation = np.array(output_w_activation, copy=True)
        return output_w_activation

    def backward_relu(self, da):
        inp = np.copy(self.output)
        dz = np.array(da, copy = True)
        dz[inp <= 0] = 0
        return dz
    
    def backward_kernel(self, da):    
        inp = np.array(self.inp_w_pad, copy= True)

        nH = math.floor((inp.shape[1] - da.shape[1]) / self.stride + 1)
        nW = math.floor((inp.shape[2] - da.shape[2]) / self.stride + 1)
        output = np.zeros((self.kernel.shape[0], nH, nW), dtype=float)
        
        for z in range(inp.shape[0]):
            for x in range(output.shape[0]): 

                for k in range(nH):
                    for l in range(nW):
                        convo = 0

                        for i in range(da.shape[1]):                
                            for j in range(da.shape[2]):

                                convo +=  inp[z][k * self.stride + i][l * self.stride + j] * da[x][i][j]
                        output[x][k][l]=convo   
        return output    


    def backward_input(self, da):
        kernel = np.array(self.kernel, copy=True)
        nH = math.floor((kernel.shape[1] + da.shape[1]) / self.stride - 1)
        nW = math.floor((kernel.shape[2] + da.shape[2]) / self.stride - 1)
        inp_len = math.floor(da.shape[0] / kernel.shape[0])
        inp = np.zeros((inp_len, nH, nW), dtype=float)

        for x in range(inp_len):
            lnf_rand = random.randint(0, kernel.shape[0]-1)
            kernelx = np.rot90(kernel[lnf_rand], 2)
            kernelx = np.pad(kernelx, ((da.shape[1]-1, da.shape[1]-1),(da.shape[2]-1, da.shape[2]-1)), 'constant')
            for k in range(nH):
                for l in range(nW):
                    convo = 0
                    for i in range(da.shape[1]):
                        for j in range(da.shape[2]):
                            convo +=  da[x * kernel.shape[0] + lnf_rand][i][j] * kernelx[k*self.stride+i][l*self.stride+j]
                    inp[x][(nH-1)-k][(nW-1)-l] = convo
        return inp

    def backward(self, da):
        dc = self.backward_relu(da)
        di = self.backward_input(dc)
        return self.unpad(di)
    
    def grad_function(self, dz):
        dk = self.backward_kernel(dz)
        return dk


# conv = ConvLayer(x, kernel_dim=(2,2), kernel_len=2, padding=1)
