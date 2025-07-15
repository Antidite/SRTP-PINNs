import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cosine


# 问题:
# m d^2/dt^2 u + miu d/dt u + ku = 0
#u|(t = 0) = 1      d/dt u|(t = 0) = 0
def exact_calculation(delta , w_0 , t): # t is a array
    assert delta < w_0
    w = np.sqrt(w_0**2 - delta**2)
    phi = np.arctan(-delta/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi + w*t)
    exp = torch.exp(-delta*t)
    u = exp*2*A*cos
    return u

class FCN(nn.Module):
    def __init__(self,N_Input,N_Output,N_Hidden,N_Layers):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(
            nn.Linear(N_Input,N_Hidden),
            activation()
        )
        self.fch = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(N_Hidden,N_Hidden),
                activation()
            ) for _ in range(N_Layers-1)]
        )
        self.fce = nn.Linear(N_Hidden,N_Output)

    def forward(self,x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

torch.manual_seed(123)

#design loss function
#loss = lambda_1*(u|(t = 0) - 1)^2 + lambda_2*(d/dt u|(t = 0))^2 + lambda*(differential equation loss)^2

#define a neural network:
pinn = FCN(1,1,32,3)

#define boundary point
t_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)

#define training points
t_physics = torch.linspace(0,1,300).view(-1,1).requires_grad_(True)

#train
delta,w_0 = 2,20
mu,k = 2*delta,w_0**2
t_test = torch.linspace(0,1,300).view(-1,1)
u_exact = exact_calculation(delta,w_0,t_test)
optimizer = torch.optim.Adam(pinn.parameters(),lr = 1e-3)
for i in range(10001):
    optimizer.zero_grad()
    lambda_1,lambda_2 = 1e-2 , 1e-4
    u = pinn(t_boundary) #torch.Size([1]) (1,1)
    loss_1 = (torch.squeeze(u) - 1)**2 #squeeze删除所有长度为1的维度
    dudt = torch.autograd.grad(u,t_boundary,torch.ones_like(u),create_graph=True)[0] # dL/du * du/dt (torch.ones_like(u) aim to set dL/du = 1)
    loss_2 = (torch.squeeze(dudt) - 0)**2 #create_graph aim to calculate derivative again

    u = pinn(t_physics)
    dudt = torch.autograd.grad(u,t_physics,torch.ones_like(u),create_graph=True)[0] # [0] represent the 第几个自变量
    d2udt2 = torch.autograd.grad(dudt,t_physics,torch.ones_like(dudt),create_graph=True)[0]
    loss_3 = torch.mean((d2udt2 + mu*dudt + k*u)**2)

    loss = loss_1 + lambda_1*loss_2 + lambda_2*loss_3
    loss.backward()
    optimizer.step()
