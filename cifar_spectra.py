#%%
import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin

import torch
import torchvision
import torchvision.transforms as transforms
DATA_PATH = '/Users/aaronhillman/Desktop/data'

"""
This script is dedicated to reproducing the various non-trivial figures in the paper "A Solvable Model of Neural Scaling Laws"
by Alexander Maloney, Daniel A. Roberts, and James Sullyde. 
"""

"""
First we load CIFAR-10.  This a database of 50k 32x32 pixel images where each pixel has three color channels.
The images are tagged with one of ten labels.
"""

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4


trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%
full_train_data = []
full_train_labels = []

for data, labels in trainloader:
    full_train_data.append(data)
    full_train_labels.append(labels)

full_train_data = torch.cat(full_train_data, dim=0)
full_train_labels = torch.cat(full_train_labels, dim=0)
flat_cifar = full_train_data.view(50000, 32*32*3)

print(full_train_data.shape)  # Output: torch.Size([50000, 3, 32, 32])
print(full_train_labels.shape)  # Output: torch.Size([50000])

# %%
"""
This image data is now described by an ordinary design matrix and we can plot the eigenvalue spectrum.
We take a slice of images of size T and then return the x and y values of the eigenvalue data points as a tuple of two lists.
"""

def spec_plot_data(T):
    idx = torch.randint(0, 50000-T+1, (1,)).item()
    data_slice = flat_cifar[idx:idx+T]
    cov = data_slice.transpose(-2, -1) @ data_slice /T
    eiglist = lin.eigvalsh(cov)
    return (range(1, 3072+1), sorted(eiglist)[::-1])

colordict = {100: 'tab:blue', 200: 'tab:orange', 400: 'tab:green', 500: 'tab:red'}

for T in [100, 200, 400, 500][::-1]:
    xdata, ydata = spec_plot_data(T)
    plt.loglog(xdata, ydata, '*', color = colordict[T])
    plt.xlim(.5, 1000)
    plt.ylim(.0025, 500)

plt.legend(["$T$ = " +str(i) for i in [100, 200, 400, 500][::-1]])
plt.title("CIFAR-10 Spectra w/ $N_{in} = 3072$ Input Features")
plt.xlabel("$i$")
plt.ylabel("$\lambda_i$")
plt.show()


#%% 
def get_batch(chop_cifar):
    T = 3072
    idx = torch.randint(0, 50000-T+1, (1,)).item()
    return chop_cifar[idx:idx+T]


colordict = {100: 'tab:blue', 200: 'tab:orange', 400: 'tab:green', 500: 'tab:red'}

for N in [100, 200, 400, 500][::-1]:
    nindx = torch.randint(0, 3072-N+1, (1,)).item()
    chop_cifar = flat_cifar.transpose(0, 1)[nindx:nindx+N].transpose(0, 1)
    data_slice = get_batch(chop_cifar)
    cov = data_slice.transpose(-2, -1) @ data_slice /3072
    eiglist = lin.eigvalsh(cov)
    xdata, ydata = (range(1, N+1), sorted(eiglist)[::-1])
    plt.loglog(xdata, ydata, '*', color = colordict[N])
    plt.xlim(.8, 800)
    plt.ylim(.0002, 95)

plt.legend(["$N_{in}$ = " +str(i) for i in [100, 200, 400, 500][::-1]])
plt.title("CIFAR-10 Spectra w/ $T = 3072$ Data Set Size")
plt.xlabel("$i$")
plt.ylabel("$\lambda_i$")
plt.show()

plt.show()


#%%
"""
More PCA analysis on CIFAR:

While here with CIFAR, we can see what images look like when we reconstruct them only using some of the top principal components.
"""

full_cov = flat_cifar.transpose(-2, -1) @ flat_cifar /len(full_train_data)
eigsystem = lin.eigh(full_cov)

def expansion(flat_img, eig_no):
    return sum([(flat_img @ torch.tensor(eig_vec))*torch.tensor(eig_vec) for eig_vec in eigsystem.eigenvectors[-eig_no:]])

image_idx = 0
component_nums = [750, 1250, 2500, 2750, 3072]
images_out = []
for num in component_nums:
    truncated = expansion(flat_cifar[image_idx ], num)
    images_out.append(np.transpose((truncated.view(3, 32, 32)/2+0.5).numpy(), (1, 2, 0)))


f, axarr = plt.subplots(1,len(component_nums))
for i in range(len(component_nums)):
    axarr[i].imshow(images_out[i])



#%%
"""
Now we check if the performance is any better if we only compute the covariance on the specific animal subspace.
"""

flat_class = torch.stack([flat_cifar[idx] for idx in range(len(flat_cifar)) if classes[full_train_labels[idx]] == classes[full_train_labels[0]]  ])
class_cov = flat_class.transpose(-2, -1) @ flat_class /len(flat_class)
eigsystem = lin.eigh(class_cov)

def expansion(flat_img, eig_no):
    return sum([(flat_img @ torch.tensor(eig_vec))*torch.tensor(eig_vec) for eig_vec in eigsystem.eigenvectors[-eig_no:]])

image_idx = 0
component_nums = [750, 1250, 2500, 2750, 3072]
images_out = []
for num in component_nums:
    truncated = expansion(flat_class[image_idx], num)
    images_out.append(np.transpose((truncated.view(3, 32, 32)/2+0.5).numpy(), (1, 2, 0)))


f, axarr = plt.subplots(1,len(component_nums))
for i in range(len(component_nums)):
    axarr[i].imshow(images_out[i])




# %%
def umapcifar(Nout):
    u = nn.Linear(32*32*3, Nout, bias=False)
    return u(flat_cifar)

def u_spec_plot_data(Nout, T):
    idx = torch.randint(0, 50000-T+1, (1,)).item()
    data_slice = umapcifar(Nout)[idx:idx+T]
    cov = data_slice.transpose(-2, -1) @ data_slice /T
    eiglist = lin.eigvalsh(cov.detach().numpy())
    return (range(1, Nout+1)[:3072], sorted(eiglist)[::-1][:3072])

colordict = {3072: 'tab:blue', 4000: 'tab:orange', 8000: 'tab:green', 12000: 'tab:red', 15000: 'tab:purple'}

for Nout in [3072, 4000, 8000, 12000, 15000][::-1]:
    if N == 3072:
        xdata, ydata = spec_plot_data(15000)
    else:
        xdata, ydata = u_spec_plot_data(Nout, 15000)
    plt.loglog(xdata, ydata, '*', color = colordict[Nout])
    plt.xlim(.5, 6000)
    plt.ylim(1e-7, 1000)

plt.legend(["$N_{in}$ = " +str(i) for i in [4000, 8000, 12000, 15000][::-1]]+["input"])
plt.title("CIFAR-10 Spectra w/ Linear Map")
plt.xlabel("$i$")
plt.ylabel("$\lambda_i$")
plt.show()


# %%
def relumapcifar(Nout):
    u = nn.Linear(32*32*3, Nout, bias=False)
    return nn.ReLU()(u(flat_cifar))

def relu_spec_plot_data(Nout, T):
    idx = torch.randint(0, 50000-T+1, (1,)).item()
    data_slice = relumapcifar(Nout)[idx:idx+T]
    cov = data_slice.transpose(-2, -1) @ data_slice /T
    eiglist = lin.eigvalsh(cov.detach().numpy())
    return (range(1, Nout+1), sorted(eiglist)[::-1])

colordict = {3072: 'tab:blue', 4000: 'tab:orange', 8000: 'tab:green', 12000: 'tab:red', 15000: 'tab:purple'}


for Nout in [3072, 4000, 8000, 12000, 15000][::-1]:
    if N == 3072:
        xdata, ydata = spec_plot_data(15000)
    else:
        xdata, ydata = relu_spec_plot_data(Nout, 15000)
    plt.loglog(xdata, ydata, '*', color = colordict[Nout])
    plt.xlim(.5, 20000)
    plt.ylim(1e-7, 1000)



plt.legend(["$N_{in}$ = " +str(i) for i in [4000, 8000, 12000, 15000][::-1]]+["input"])
plt.title("CIFAR-10 Spectra w/ ReLU Map")
plt.xlabel("$i$")
plt.ylabel("$\lambda_i$")
plt.show()
#  
# %%

#first we will generate a random covariance matrix lambda of dimension M

M = 6000
N = 400
sigw2 = 1
sigu2 = 1
wiI = torch.normal( torch.zeros((N, M), dtype=torch.float32)  , (sigw2/M)*torch.ones((N, M)) )
uiI = torch.normal( torch.zeros((N, M), dtype=torch.float32)  , (sigu2/M)*torch.ones((N, M)) )
thetaij = torch.randn((N, N),dtype=torch.float32)
R = torch.nn.init.orthogonal_(torch.empty(M, M, dtype=torch.float32))



def gen_Lambda():
    A = torch.randn((M, M))
    return A.transpose(-2, -1) @ A


def gen_Lambda_pow(lminus, alpha):
    spec_vec = torch.tensor([lminus*M**(1+alpha)*(1/i)**(1+alpha) for i in range(1, M+1) ], dtype=torch.float32  )
    spec_vec = spec_vec/max(spec_vec)
    lambda_diag = torch.diag(spec_vec)
    return lambda_diag 

Lambda = gen_Lambda_pow(1, 1)

#%%

#how to impose that the spectrum is a power law?

#get x will now be based on this Lambda
def get_xI(n_samples):
    dist = torch.distributions.MultivariateNormal(torch.zeros(M, dtype=torch.float32) , Lambda )
    diagsample = dist.sample((n_samples,))
    return torch.einsum('ij,kj->ik', diagsample, R)

def dataset(n_points):
    xIs = get_xI(n_points)
    yis = torch.einsum('ij,kj->ik',  xIs, wiI)
    return (xIs, yis)

def phi(xI):
   return torch.einsum('j,kj->k',  xI, uiI)

def zi(xI):
    return torch.einsum('ij,j->i', thetaij, phi(xI))


#as we expect, the number of nonzero eigenvalues is limited by the number of vectors

#now we would like to construct a model for phi which we will train.
#The solution to the regularized porblem is known analytically as this is the classic
#problem in multiple regression 



# %%

gamma = 1e-7
That = 1000

def loss_data(T):
    xIs, yis = dataset(T)
    xIhats, yihats = dataset(That)
    phis = torch.stack([phi(xI) for xI in xIs])

    def qij(gamma):
        return torch.linalg.inv(gamma*torch.diag(torch.ones((N,)))+ phis.transpose(-2, -1) @ phis)

    def thetaijst(gamma):
        qq = qij(gamma)
        return torch.einsum('ai, jk, ak -> ij', yis, qq, phis)

    theta_star = thetaijst(gamma)

    def zist(xI):
        return torch.einsum('ij,j->i', theta_star, phi(xI))


    zsts =  torch.stack([zist(xIhat) for xIhat in xIhats])
    loss = sum( [sum([x.item()**2 for x in zsts[i]-yihats[i] ]  ) for i in range(That)])/That
    return loss

losses = []
T_data = []
for x in torch.logspace(1/2, 4, 80, base= 10):
    T = int(x.item())
    T_data.append(T)
    loss = loss_data(T)
    losses.append(loss)

plt.loglog(T_data, losses, '*', color='tab:orange')
plt.xlim((7e-1, 1e4))
plt.ylim((1e-7, 3e-4))
plt.show()

#%%
plt.loglog(T_data, losses, '*', color='tab:orange')
plt.xlim((7e-1, 1e4))
plt.ylim((1e-7, 3e-4))
plt.xlabel('T')
plt.ylabel('L(T, N)')
plt.show() 

# %%
#let's plot test loss as a function of ridge parameter in order to explore this

That = 1000

def loss_data(T, gamma):
    xIs, yis = dataset(T)
    xIhats, yihats = dataset(That)
    phis = torch.stack([phi(xI) for xI in xIs])

    def qij(gamma):
        return torch.linalg.inv(gamma*torch.diag(torch.ones((N,)))+ phis.transpose(-2, -1) @ phis)

    def thetaijst(gamma):
        qq = qij(gamma)
        return torch.einsum('ai, jk, ak -> ij', yis, qq, phis)

    theta_star = thetaijst(gamma)

    def zist(xI):
        return torch.einsum('ij,j->i', theta_star, phi(xI))


    zsts =  torch.stack([zist(xIhat) for xIhat in xIhats])
    loss = sum( [sum([x.item()**2 for x in zsts[i]-yihats[i] ]  ) for i in range(That)])/That
    return loss


T = 100
gamma_data = []
losses = []
for gamma in torch.logspace(-9, -6, 10):
    gamma_data.append(gamma)
    loss = loss_data(T, gamma)
    losses.append(loss)

plt.plot(gamma_data, losses)
plt.show()
# %%
plt.plot(gamma_data[:-1], losses, 'o')
plt.show()

# %%
print('save')
# %%