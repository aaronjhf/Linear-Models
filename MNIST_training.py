# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
#loading MNIST
train_data = dsets.MNIST(root = '/Users/aaronhillman/Desktop/data', train = True,transform = transforms.ToTensor(),
                   download = True)
test_data = dsets.MNIST(root = '/Users/aaronhillman/Deskstop/data', train = False,transform = transforms.ToTensor(),
                   download = True)
# %%
Xtr = torch.stack([x[0].view((28*28,)) for x in train_data]).to(device)
Ytr = torch.tensor([x[1] for x in train_data]).to(device)
Xte = torch.stack([x[0].view((28*28,)) for x in test_data]).to(device)
Yte = torch.tensor([x[1] for x in test_data]).to(device)


# %%
#setup the model
class MLP(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.width = width
        self.net = nn.Sequential( nn.Linear(28*28, width), nn.ReLU(), nn.Linear(width, 10))

    

    def forward(self, x_batch, y_batch = None):
        x_out = self.net(x_batch)
        if y_batch is None:
            loss = None
        else:
            loss = F.cross_entropy(x_out, y_batch)

        return x_out, loss

#%%
def MLP_param_count(MLPwidth):
    m = MLP(MLPwidth).to(device)
    param_count = 0
    for p in m.parameters():
        param_count += p.numel()

    return param_count

# %%

MLP_sizes = []
MLP_losses = []

for model_size in torch.logspace(1.5, 4, 50):
    
    MLP_sizes.append(MLP_param_count(int(model_size)))

    m = MLP(int(model_size)).to(device) #create the model
    print(int(model_size))

    optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-4)

    xloss = []
    yloss = []
    running_loss = 0
    B = 64

    for i in range(10000):
        rand_inds = torch.randint(0, len(Xtr), (B,))
        Xb, Yb = Xtr[rand_inds], Ytr[rand_inds]
        out_probs, loss = m.forward(Xb, Yb)

        optimizer.zero_grad(set_to_none = True)

        loss.backward()
        optimizer.step()

        running_loss += loss

        if i%200 == 0:
            xloss.append(i)
            yloss.append(running_loss/200)
            running_loss = 0
    
    MLP_losses.append(yloss[-1].item())
            
with open("MLPsizes_cuda.txt", "w") as f:
    np.savetxt(f, MLP_sizes)

with open("MLPlosses_cuda.txt", "w") as f:
    np.savetxt(f, MLP_losses)


"""Now the CNN"""

#%%
Xtr = torch.stack([x[0] for x in train_data]).to(device)
Ytr = torch.tensor([x[1] for x in train_data]).to(device)
Xte = torch.stack([x[0] for x in test_data]).to(device)
Yte = torch.tensor([x[1] for x in test_data]).to(device)
#%%
#convolution layer applied to MNIST image

class CNN(nn.Module):

    def __init__(self, width, k):
        super().__init__()
        self.width = width
        self.k = k
        self.conv = nn.Conv2d(1, width , k)
        self.lin = nn.Linear(self.width*(28-self.k+1)*(28-self.k+1), 10)
    
    def forward(self, x_batch, y_batch = None):
        B = x_batch.shape[0]
        x = self.conv(x_batch)
        x = F.relu(x)
        x = x.view((B, self.width*(28-self.k+1)*(28-self.k+1)))
        x = self.lin(x)
        
        if y_batch is None:
            loss = None
        else:
            loss = F.cross_entropy(x, y_batch)
        
        return x, loss 
    
def CNN_param_count(width, k):
    m = CNN(width, k).to(device)
    param_count = 0
    for p in m.parameters():
        param_count += p.numel()

    return param_count


#%%
CNN3_sizes = []
CNN3_losses = []
# %%

for model_size in torch.logspace(1, 3, 50):
    print(int(model_size))

    CNN3_sizes.append(CNN_param_count(int(model_size), 3))

    m = CNN(int(model_size), 3).to(device) #create the model

    optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-4)

    xloss = []
    yloss = []
    running_loss = 0
    B = 64

    for i in range(10000):
        rand_inds = torch.randint(0, len(Xtr), (B,))
        Xb, Yb = Xtr[rand_inds], Ytr[rand_inds]
        out_probs, loss = m.forward(Xb, Yb)

        optimizer.zero_grad(set_to_none = True)

        loss.backward()
        optimizer.step()

        running_loss += loss

        if i%200 == 0:
            xloss.append(i)
            yloss.append(running_loss/200)
            running_loss = 0
    
    CNN3_losses.append(yloss[-1].item())

with open("CNN3sizes.txt", "w") as f:
    np.savetxt(f, CNN3_sizes)

with open("CNN3losses.txt", "w") as f:
    np.savetxt(f, CNN3_losses)
#%%
CNN14_sizes = []
CNN14_losses = []

for model_size in torch.logspace(1, 3, 50):
    print(int(model_size))

    CNN14_sizes.append(CNN_param_count(int(model_size), 14))

    m = CNN(int(model_size), 14).to(device) #create the model

    optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-4)

    xloss = []
    yloss = []
    running_loss = 0
    B = 64

    for i in range(10000):
        rand_inds = torch.randint(0, len(Xtr), (B,))
        Xb, Yb = Xtr[rand_inds], Ytr[rand_inds]
        out_probs, loss = m.forward(Xb, Yb)

        optimizer.zero_grad(set_to_none = True)

        loss.backward()
        optimizer.step()

        running_loss += loss

        if i%200 == 0:
            xloss.append(i)
            yloss.append(running_loss/200)
            running_loss = 0
    
    CNN14_losses.append(yloss[-1].item())

with open("CNN14sizes.txt", "w") as f:
    np.savetxt(f, CNN14_sizes)

with open("CNN14losses.txt", "w") as f:
    np.savetxt(f, CNN14_losses)



# %%
