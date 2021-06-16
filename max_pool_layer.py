import numpy as np
import math

m_layer = chr(ord('A')-1)

class MaxPoolLayer:
    def __init__(self, pool_size, padding = 0, stride = 2):        
        self.padding = padding
        self.stride = stride
        self.pool_size = (pool_size, pool_size)
        self.max_pool_index = []
        self.layer = self.layer_name()
        self.output = []
        self.d1z = []

    def set_inp(self, inp):
        self.inp = inp
        self.inp_w_pad = self.set_padding(inp, self.padding)
        self.max_pool_index = []

    def layer_name(self):
        global m_layer
        m_layer = chr(ord(m_layer) + 1) 
        return 'M'+m_layer
    
    def set_padding(self, inp, padding):
        if padding > 0:
            output = []
            for i in range(inp.shape[0]):
                output.append(np.pad(inp[i], (padding, padding), 'constant'))
            return np.array(output)
        else:
            return inp
        
    def forward(self):
        output_size_x = math.floor( (self.inp.shape[1] + (2*self.padding) - self.pool_size[0]) / self.stride ) + 1
        output_size_y = math.floor( (self.inp.shape[2] + (2*self.padding) - self.pool_size[1]) / self.stride ) + 1
        output = np.zeros(( (self.inp.shape[0]), output_size_x, output_size_y), dtype=float)
        max_pool_index = []
        inp = np.copy(self.inp)
        for x in range(self.inp.shape[0]):

            for k in range(output_size_x):
                for l in  range(output_size_y):
                    maks = 0

                    for i in range(self.pool_size[0]) :
                        for j in range(self.pool_size[1]) :
                            val = inp[x][k * self.stride + i][l * self.stride + j]
                            if maks <= val:
                                maks = val
                                max_pool_index = (x, (k * self.stride + i), (l * self.stride + j))
                    output[x][k][l] = maks
                    self.max_pool_index.append(max_pool_index)
        self.output = output
        return output

    def backward(self, dz):
        max_pool_index = self.max_pool_index
        inp = np.copy(self.inp)
        # inp = np.zeros(( self.inp.shape ), dtype=float)
        d_pool_img = dz.flatten()
        for x in range(dz.shape[0]):
            d_pool_img_i = 0
            for c in max_pool_index:
                inp[c[0]][c[1]][c[2]] = d_pool_img[d_pool_img_i]
                d_pool_img_i += 1

        return inp

# pool = MaxPoolLayer(z, pool_size = (2,2), padding = 1, stride = 2)
