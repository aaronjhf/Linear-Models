#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

torch.manual_seed(42) 

#input dimension
D = 25
#matrix dimension
N = 500
#batch dimension
B = 64
#number of data points
P = 30000


# %%
"""
We are going to make a quadratic model which we will train with an MLP in the lazy limit.
"""
#Generate quadratic data
Xraw = torch.randn((P, D))
Wmap = torch.randn((D, 1))/np.sqrt(D)
Yraw = (Xraw@Wmap)**2
Yraw.shape

#%%
def MLP(params, x, gamma):
    for param in params[:-1]: 
        x = x @ param  /(np.sqrt(x.shape[-1]))
        x = nn.ReLU()(x)

    return x@params[-1] /(gamma*np.sqrt(x.shape[-1]))
    
# %%
"""
Let's make an input and an output can confirm that we can train on the above
"""
loss_dict = defaultdict(list)


for gamma in [1,  1e-1, 1e-3, 1e-5]:
    params = [ torch.randn((i, j), requires_grad=True) for i, j in [(D, N), (N, N), (N, 1)] ]
    params0 = [torch.clone(param.detach()) for param in params]


    fwd_fn = lambda params, Xin, gamma : MLP(params, Xin, gamma)-MLP(params0, Xin, gamma)
    loss_fn = lambda params, Xin, gamma, Yout : (fwd_fn(params, Xin, gamma)-Yout).transpose(0, 1) @(fwd_fn(params, Xin, gamma)-Yout) / Yout.shape[0]

    eta = 2*gamma**2


    optimizer = torch.optim.SGD(params, lr=eta)
    optimizer.zero_grad()
    loss = loss_fn(params, Xraw[0:B], gamma, Yraw[0:B])
    loss.backward()
    optimizer.step()

    
    loss_dict[gamma] = []
    for t in range(P//B):
        optimizer.zero_grad()
        loss = loss_fn(params, Xraw[t*B:(t+1)*B], 1, Yraw[t*B:(t+1)*B])
        loss.backward()
        optimizer.step()
        loss_dict[gamma].append(loss.item())
        

 #%%
 #Now the explicitly linearized model        
def linear_approx(params, params0, Xin, gamma):
    def f(params):
        return fwd_fn(params, Xin, gamma)
    
    v = [p - p0 for p, p0 in zip(params, params0)]
    _, jvp = torch.func.jvp(f, (params,), (v,))
    return jvp

loss_dict[0] = []
# Example usage of linear approximation
for t in range(P//B):
    jvp_result = linear_approx(params, params0, Xraw[t*B:(t+1)*B], 1)
    loss = (jvp_result - Yraw[t*B:(t+1)*B]).transpose(0, 1) @ (jvp_result - Yraw[t*B:(t+1)*B]) / Yraw[t*B:(t+1)*B].shape[0]
    loss_dict[0].append(loss.item())



# %%
for gamma in loss_dict.keys():
    plt.loglog([t for t in range(P//B)] , loss_dict[gamma])

plt.show()
# %%
"""
So far we have seen that on the quadratic task, the lazy training regime (small gamma)
is equivalent to using the linearization of the model. 
Next we will explore the Neutral Tangent Kernel (NTK), which is a good approximation at large width, where the weights naturally
become lazy.
"""
