import numpy as np
class AdamOptim():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw1, self.v_dw1 = 0, 0
        self.m_dw2, self.v_dw2 = 0, 0
        self.m_db1, self.v_db1 = 0, 0
        self.m_db2, self.v_db2 = 0, 0
        self.m_dk1, self.v_dk1 = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        
    def update(self, t, w1, b1, dw1, db1, w2, b2, dw2, db2, k1, dk1):
        ## momentum beta 1
        # *** weights *** #
        self.m_dw1 = self.beta1*self.m_dw1 + (1-self.beta1)*dw1
        self.m_dw2 = self.beta1*self.m_dw2 + (1-self.beta1)*dw2
        self.m_dk1 = self.beta1*self.m_dk1 + (1-self.beta1)*dk1
        # *** biases *** #
        self.m_db1 = self.beta1*self.m_db1 + (1-self.beta1)*db1
        self.m_db2 = self.beta1*self.m_db2 + (1-self.beta1)*db2
        self.m_dk1 = self.beta1*self.m_dk1 + (1-self.beta1)*dk1

        ## rms beta 2
        # *** weights *** #
        self.v_dw1 = self.beta2*self.v_dw1 + (1-self.beta2)*(dw1**2)
        self.v_dw2 = self.beta2*self.v_dw2 + (1-self.beta2)*(dw2**2)
        # *** biases *** #
        self.v_db1 = self.beta2*self.v_db1 + (1-self.beta2)*(db1**2)
        self.v_db2 = self.beta2*self.v_db2 + (1-self.beta2)*(db2**2)
        # *** kernel *** #
        self.v_dk1 = self.beta2*self.v_dk1 + (1-self.beta2)*(dk1**2)

        ## bias correction
        m_dw1_corr = self.m_dw1/(1-self.beta1**t)
        m_dw2_corr = self.m_dw2/(1-self.beta1**t)
        m_db1_corr = self.m_db1/(1-self.beta1**t)        
        m_db2_corr = self.m_db2/(1-self.beta1**t)
        m_dk1_corr = self.m_dk1/(1-self.beta1**t)

        v_dw1_corr = self.v_dw1/(1-self.beta2**t)
        v_dw2_corr = self.v_dw2/(1-self.beta2**t)
        v_db1_corr = self.v_db1/(1-self.beta2**t)        
        v_db2_corr = self.v_db2/(1-self.beta2**t)
        v_dk1_corr = self.v_dk1/(1-self.beta2**t)

        ## update weights and biases
        w1 = w1 - self.eta*(m_dw1_corr/(np.sqrt(v_dw1_corr)+self.epsilon))
        b1 = b1 - self.eta*(m_db1_corr/(np.sqrt(v_db1_corr)+self.epsilon))
        w2 = w2 - self.eta*(m_dw2_corr/(np.sqrt(v_dw2_corr)+self.epsilon))
        b2 = b2 - self.eta*(m_db2_corr/(np.sqrt(v_db2_corr)+self.epsilon))
        k1 = k1 - self.eta*(m_dk1_corr/(np.sqrt(v_dk1_corr)+self.epsilon))
        return w1, b1, w2, b2, k1

# def loss_function(m):
#     return m**2-2*m+1
# ## take derivative
# def grad_function(m):
#     return 2*m-2
def check_convergence(w0, w1):
    return (w0 == w1)

# w_0 = 0
# b_0 = 0
# adam = AdamOptim()
# t = 1 
# converged = False

# while not converged:
#     dw = grad_function(w_0)
#     db = grad_function(b_0)
#     w_0_old = w_0
#     w_0, b_0 = adam.update(t,w=w_0, b=b_0, dw=dw, db=db)
#     if check_convergence(w_0, w_0_old):
#         print('iteration '+str(t)+': weightold: '+str(w_0_old)+' weight='+str(w_0))
#         print('converged after '+str(t)+' iterations')
#         break
#     else:
#         print('iteration '+str(t)+': weightold: '+str(w_0_old)+' weight='+str(w_0))
#         t+=1