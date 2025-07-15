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

#train(study mu)
delta,w_0 = 2,20
_,k = 2*delta,w_0**2
t_test = torch.linspace(0,1,300).view(-1,1)
mu = torch.nn.Parameter(torch.zeros(1,requires_grad=True))
u_exact = exact_calculation(delta,w_0,t_test)
optimizer = torch.optim.Adam([*pinn.parameters(),mu],lr = 1e-3)
t_obs = torch.rand(40).view(-1,1)
u_obs = exact_calculation(delta,w_0,t_obs) + 0.04*torch.randn_like(t_obs)
for i in range(20001):
    optimizer.zero_grad()
    lambda0 = 1e4

    u = pinn(t_physics)
    dudt = torch.autograd.grad(u,t_physics,torch.ones_like(u),create_graph=True)[0] # [0] represent the 第几个自变量
    d2udt2 = torch.autograd.grad(dudt,t_physics,torch.ones_like(dudt),create_graph=True)[0]
    loss_1 = torch.mean((d2udt2 + mu*dudt + k*u)**2)

    u = pinn(t_obs)
    loss_2 = torch.mean((u-u_obs)**2)

    loss = loss_1 + lambda0*loss_2
    if i % 2000 == 0:
        print(loss.item())
        print(mu.item())
    loss.backward()
    optimizer.step()
