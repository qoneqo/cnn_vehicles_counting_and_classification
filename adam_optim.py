import numpy as np
class AdamOptim():
    def __init__(self, params_len, eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params_len = params_len
        self.m_d = []
        self.v_d = []
        for i in range(params_len):
            self.m_d.append(0)
            self.v_d.append(0)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        
    def update(self, params, params_d, t=1):
        m_d_corr = []
        v_d_corr = []
        for i in range(self.params_len):
            ## momentum beta 1
            self.m_d[i] = self.beta1*self.m_d[i] + (1-self.beta1)*params_d[i]
            
            ## rms beta 2
            self.v_d[i] = self.beta2*self.v_d[i] + (1-self.beta2)*(params_d[i]**2)
            
            ## bias correction
            m_d_corr.append(self.m_d[i]/(1-self.beta1**t))
            v_d_corr.append(self.v_d[i]/(1-self.beta2**t))

            ## update weights and biases
            params[i] = params[i] - self.eta*(m_d_corr[i]/(np.sqrt(v_d_corr[i])+self.epsilon))
        # print(params[2][1])
        # print('\n\n')
        # print(params_d[2][1])

        return params
