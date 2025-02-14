#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import seaborn as sns # type: ignore

torch.manual_seed(42) 

#input dimension
D = 4
#matrix dimension
N = 2000
#batch dimension
B = 64
#number of data points
P = 10000


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
    params = [ torch.randn((i, j), requires_grad=True) for i, j in [(D, N),  (N, 1)] ]
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
params = [ torch.randn((i, j), requires_grad=True) for i, j in [(D, N),  (N, 1)] ]
params0 = [torch.clone(param.detach()) for param in params]    

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
sns.set_style("whitegrid")
sns.set_palette("plasma", len(loss_dict)-1)
plt.figure(figsize=(8, 6))




for gamma in [1e-05, 0.001, 0.1, 1]:
    plt.plot([t for t in range(P//B)] , loss_dict[gamma], alpha=1.0)
    plt.xscale('log')
    plt.yscale('log')

plt.plot([t for t in range(P//B)] , loss_dict[0], 'k--', alpha = 0.3)
plt.xscale('log')
plt.yscale('log')

plt.show()
# %%
"""
So far we have seen that on the quadratic task, the lazy training regime (small gamma)
is equivalent to using the linearization of the model. 
Next we will explore the Neutral Tangent Kernel (NTK), which is a good approximation at large width, where the weights naturally
become lazy.
So we want to train the linearized model and see how it compares to the wide MLP.
"""
#%%
params = [ torch.randn((i, j), requires_grad=True) for i, j in [(D, N), (N, 1)] ]
params0 = [torch.clone(param.detach()) for param in params]

gamma = 1
N = 5000

fwd_fn = lambda params, Xin, gamma : MLP(params, Xin, gamma)-MLP(params0, Xin, gamma)

def linear_approx(params, params0, Xin, gamma):
    def f(params):
        return fwd_fn(params, Xin, gamma)
    
    v = [p - p0 for p, p0 in zip(params, params0)]
    _, jvp = torch.func.jvp(f, (params0,), (v,))
    return jvp



linear_fwd = lambda params, Xin, gamma : linear_approx(params, params0, Xin, gamma)
linear_loss = lambda params, Xin, gamma, Yout : (linear_approx(params, params0, Xin, gamma)-Yout).transpose(0, 1) @(linear_approx(params, params0, Xin, gamma)-Yout) / Yout.shape[0]


eta = 2*gamma**2


optimizer = torch.optim.SGD(params, lr=eta)



loss_dict["linearized"] = []
for t in range(0, P//B):
    optimizer.zero_grad()
    loss = linear_loss(params, Xraw[t*B:(t+1)*B], gamma, Yraw[t*B:(t+1)*B])
    loss.backward()
    optimizer.step()
    loss_dict["linearized"].append(loss.item())
    
# %%
plt.plot([t for t in range(P//B)] , loss_dict["linearized"], alpha = 1.0)
plt.plot([t for t in range(P//B)] , loss_dict[1], alpha = 1.0)
plt.xscale('log')
plt.yscale('log')

plt.show()
# %%
"""
Computing the Neural Tangent Kernel
"""

def compute_ntk(fwd_fn, params, Xin, gamma):
    # Compute the Jacobian of the model's output with respect to its parameters
    def jacobian(fwd_fn, params, Xin, gamma):
        output = fwd_fn(params, Xin, gamma)
        jacobian = []
        for out in output:
            grad_params = torch.autograd.grad(out, params, retain_graph=True, create_graph=True)
            jacobian.append(torch.cat([g.view(-1) for g in grad_params]))
        return torch.stack(jacobian)

    # Compute the Jacobians for both input sets
    J = jacobian(fwd_fn, params, Xin, gamma)
    

    # Compute the NTK as the Gram matrix of the Jacobians
    NTK = J @ J.transpose(0, 1)
    return NTK

# Example usage
params = [torch.randn((i, j), requires_grad=True) for i, j in [(D, N), (N, 1)]]
params0 = [torch.clone(param.detach()) for param in params]
gamma = 1

fwd_fn = lambda params, Xin, gamma : MLP(params, Xin, gamma) - MLP(params0, Xin, gamma)

ntk = compute_ntk(fwd_fn, params, Xraw[0:10000], 1)
spec = np.linalg.eigvalsh(np.array(ntk.detach()))
plt.loglog([x for x in range(1, len(spec)+1)], sorted(spec)[::-1])
plt.show()

# %%
def relu_ntk(X1, X2):
    """
    Compute the NTK for a single hidden layer with ReLU activation in the infinite width limit.
    X1: torch.Tensor of shape (n1, d)
    X2: torch.Tensor of shape (n2, d)
    Returns: torch.Tensor of shape (n1, n2)
    """
    X1 = X1 / torch.norm(X1, dim=1, keepdim=True)
    X2 = X2 / torch.norm(X2, dim=1, keepdim=True)
    
    dot_product = X1 @ X2.T
    # Clip the dot product to the range [-1, 1] to avoid NaNs in arccos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    theta = torch.acos(dot_product)
    
    ntk = (torch.sin(theta) + (np.pi - theta) * torch.cos(theta)) / np.pi
    return ntk

# Example usage
ntk = relu_ntk(Xraw[0:10000], Xraw[0:10000])
print(ntk)

# Plot the eigenvalues of the NTK matrix
spec = np.linalg.eigvalsh(np.array(ntk.detach()))
plt.loglog([x for x in range(1, len(spec)+1)], sorted(spec)[::-1])
plt.show()

# %%
