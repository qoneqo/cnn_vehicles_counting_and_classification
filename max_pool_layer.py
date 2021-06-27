import numpy as np
import math

class MaxPoolLayer:
    def __init__(self, pool_size, padding = 0, stride = 2):        
        self.padding = padding
        self.stride = stride
        self.pool_size = (pool_size, pool_size)
        self.max_pool_index = []
        self.output = []
        self.d1z = []

    def set_inp(self, inp):
        self.inp = inp
        self.inp_w_pad = self.set_padding(inp, self.padding)
        self.max_pool_index = []

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

    def forward(self):
        output_size_x = math.floor( (self.inp.shape[1] + (2*self.padding) - self.pool_size[0]) / self.stride ) + 1
        output_size_y = math.floor( (self.inp.shape[2] + (2*self.padding) - self.pool_size[1]) / self.stride ) + 1
        output = np.zeros(( (self.inp.shape[0]), output_size_x, output_size_y))
        max_pool_index = []
        inp = np.copy(self.inp_w_pad)
        
        for x in range(self.inp.shape[0]):

            for k in range(output_size_x):
                for l in  range(output_size_y):
                    # maks = 0 error
                    maks = inp[x][k * self.stride][l * self.stride]

                    for i in range(self.pool_size[0]) :
                        for j in range(self.pool_size[1]) :
                            val = inp[x][k * self.stride + i][l * self.stride + j]
                            if maks <= val:
                                maks = val
                                max_pool_index = [x, (k * self.stride + i), (l * self.stride + j)]
                    
                    self.max_pool_index.append(max_pool_index)
                    output[x][k][l] = maks
        self.output = output
        return output

    def backward(self, dz):
        # inp = np.copy(self.inp_w_pad)
        inp = np.zeros(( self.inp_w_pad.shape ))
        d_pool_img = dz.flatten()
        for x in range(dz.shape[0]):
            d_pool_img_i = 0
            for c in self.max_pool_index:
                inp[c[0]][c[1]][c[2]] = d_pool_img[d_pool_img_i]
                d_pool_img_i += 1

        if self.padding:
            inp = self.unpad(inp)

        return inp

# pool = MaxPoolLayer(z, pool_size = (2,2), padding = 1, stride = 2)
