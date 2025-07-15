import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_device("cuda")
#Data prepare
nu = 0.01/np.pi

def bound(x):
    phi = torch.pi*x
    return -1*torch.sin(phi)

def exact_calculation(x, t, nu, eps=1e-12):
    num1 = torch.exp(-(x - 4*t)**2 / (4*nu*(t+1)))
    num2 = torch.exp(-(x + 4*t)**2 / (4*nu*(t+1)))
    phi  = num1 + num2
    phi  = torch.clamp(phi, min=eps)
    dphi = -(x - 4*t)/(2*nu*(t+1))*num1 \
         - (x + 4*t)/(2*nu*(t+1))*num2
    return -2*nu*dphi / phi

def plot3D(x_1d: torch.Tensor,t_1d: torch.Tensor,u_grid: torch.Tensor,title="u(x, t)"):
    X, T = torch.meshgrid(
        x_1d.detach().cpu(), t_1d.detach().cpu(), indexing="ij"
    )
    X, T, U = X.numpy(), T.numpy(), u_grid.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    cf = ax.contourf(T, X, U, levels=50, cmap="rainbow")
    fig.colorbar(cf, ax=ax, label="u")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title + " (contour)")
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(7, 5))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot_surface(T, X, U, cmap="rainbow", rstride=1, cstride=1,
                     linewidth=0, antialiased=False)
    ax3.set_xlabel("t")
    ax3.set_ylabel("x")
    ax3.set_zlabel("u")
    ax3.set_title(title + " (surface)")
    plt.tight_layout()
    plt.show()

#Nerual network
class FCN(nn.Module):
    def __init__(self,N_Input,N_Output,N_Hidden,N_Layers):
        super().__init__()
        activation = nn.Sigmoid
        self.fcs = nn.Sequential(
            nn.Linear(N_Input,N_Hidden),
            activation()
        )
        self.fch = nn.Sequential(
            *[nn.Sequential(
            nn.Linear(N_Hidden,N_Hidden),
            activation()
            )for _ in range(N_Layers-1)]
        )
        self.fce = nn.Linear(N_Hidden,N_Output)
    def forward (self,x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

torch.manual_seed(123)
pinn = FCN(2,1,64,5)
pinn.train()
t_train = torch.linspace(0,1,100).requires_grad_(True)
x_train = torch.linspace(-1,1,100).requires_grad_(True)
T_train,X_train = torch.meshgrid(t_train,x_train,indexing='ij')
input_train = torch.hstack((torch.reshape(T_train,(-1,1)),
                            torch.reshape(X_train,(-1,1)))).requires_grad_(True)
t_initial = torch.tensor(0.).requires_grad_(True)
T_initial,X_initial = torch.meshgrid(t_initial,x_train,indexing='ij')
input_initial = torch.hstack((torch.reshape(T_initial,(-1,1)),
                            torch.reshape(X_initial,(-1,1)))).requires_grad_(True)
u_real = exact_calculation(x_train,t_train,nu)
optimizer_burgers = torch.optim.Adam(pinn.parameters(),lr = 1e-3)
for epoch in range(1001):
    optimizer_burgers.zero_grad()
    lambda_0 = 10
    u = pinn(input_initial)
    loss_0 = torch.mean((torch.squeeze(u) - bound(x_train))**2)

    u = pinn(input_train)
    du = torch.autograd.grad(u,input_train,torch.ones_like(u),create_graph=True)[0]
    du_dt = du[:,0]
    du_dx = du[:,1]
    d2u_dx = torch.autograd.grad(du_dx,input_train,torch.ones_like(du_dx),create_graph=True)[0]
    d2u_dx2 = d2u_dx[:,1]
    loss_physics = torch.mean((du_dt + u*du_dx - nu*d2u_dx2)**2)

    loss = loss_physics * lambda_0 + loss_0
    if epoch % 200 == 0:
        print(loss.item())
    loss.backward()
    optimizer_burgers.step()

#test
pinn.eval()
with torch.no_grad():
    u_predict = pinn(input_train).reshape(len(t_train), len(x_train))
u_exact = exact_calculation(X_train,T_train,nu).reshape(len(t_train), len(x_train))
plot3D(x_train, t_train, u_predict, title="PINN")
plot3D(x_train,t_train,u_exact, title="Exact")
loss = torch.mean((u_predict - u_exact)**2)
print(loss.item())