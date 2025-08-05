import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split

import numpy as np
import time
from pyDOE import lhs
import scipy.io

torch.set_default_dtype(torch.float)

torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def plot3D(x, t, y):
    x = torch.from_numpy(x)
    t = torch.from_numpy(t)
    y = torch.from_numpy(y)
    x_plot = x.squeeze(1)
    t_plot = t.squeeze(1)
    X, T = torch.meshgrid(x_plot, t_plot)
    F_xt = y
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(T, X, F_xt, 20, cmap='rainbow')
    fig.colorbar(cp)
    ax.set_title('F(x, t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap='rainbow')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x, t)')
    plt.show()

def plot3D_Matrix(x, t, y):
    X, T = x, t
    F_xt = y
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(T, X, F_xt, 20, cmap='rainbow')
    fig.colorbar(cp)
    ax.set_title('F(x, t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap='rainbow')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x, t)')
    plt.show()

def solutionplot(u_pred,X_u_train,u_train):
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(u_pred, interpolation='nearest', cmap='rainbow', 
                extent=[T.min(), T.max(), X.min(), X.max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 0.5, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(x,t)$', fontsize = 10)
    
    ''' 
    Slices of the solution at points t = 0.25, t = 0.50 and t = 0.75
    '''
    
    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,usol.T[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')    
    ax.set_title('$t = 0.25s$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,usol.T[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50s$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,usol.T[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75s$', fontsize = 10)
    
    plt.savefig('Burgers-Inverse.png',dpi = 500) 

class DNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers)-1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain = 1.0)
            nn.init.zeros_(self.linears[i].bias.data)
            
    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)

        x = (x - l_b)/(u_b - l_b)
        a = x.float()
        for i in range(len(layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
class FCN():
    def __init__(self, layers):
        self.iter = 0
        
        self.lambda1 = torch.tensor([lambda1], requires_grad=True).float().to(device)
        self.lambda2 = torch.tensor([lambda1], requires_grad=True).float().to(device)
        self.lambda1 = nn.Parameter(self.lambda1)
        self.lambda2 = nn.Parameter(self.lambda2)
        
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda1', self.lambda1)
        self.dnn.register_parameter('lambda2', self.lambda2)

        self.loss_function = nn.MSELoss(reduction = 'mean')

    def loss_data(self, x, u):
        loss_u = self.loss_function(self.dnn(x), u)
        return loss_u

    def loss_PDE(self, X_train_Nu):
        lambda1 = self.lambda1
        lambda2 = self.lambda2
        g = X_train_Nu.clone()
        g.requires_grad = True
        u = self.dnn(g)
        u_x_t = autograd.grad(u, g, torch.ones([X_train_Nu.shape[0], 1]).to(device), retain_graph = True, create_graph = True)[0]
        u_xx_tt = autograd.grad(u_x_t, g, torch.ones(X_train_Nu.shape).to(device), create_graph = True)[0]
        u_x = u_x_t[:, [0]]
        u_t = u_x_t[:, [1]]
        u_xx = u_xx_tt[:, [0]]
        f = u_t + (lambda1)*(self.dnn(g))*(u_x) - (lambda2)*u_xx
        loss_f = self.loss_function(f, f_hat)
        return loss_f

    def loss(self, x, y):
        loss_u = self.loss_data(x, y)
        loss_f = self.loss_PDE(x)
        loss_val = loss_u + loss_f
        return loss_val

    def closure(self):
        optimizer.zero_grad()
        loss = self.loss(X_train_Nu, U_train_Nu)
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            error_vec, _ = PINN.test()
            print(
                'Relative Error(Test): %.5f, λ_real = [1.0, %.5f], λ_PINN = [%.5f, %.5f]' %
                (
                    error_vec.cpu().detach().numpy(),
                    nu,
                    self.lambda1.item(),
                    self.lambda2.item()
                )
            )
        return loss

    def test(self):
        u_pred = self.dnn(X_true)
        error_vec = torch.linalg.norm((U_true-u_pred), 2)/torch.linalg.norm(U_true, 2)
        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred, (x.shape[0], t.shape[0]), order = 'F')
        return error_vec, u_pred

data = scipy.io.loadmat('./data/Burgers.mat')
x = data['x']
t = data['t']
usol = data['usol']

plot3D(x, t, usol)

X, T = np.meshgrid(x, t)
X_true = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
lb = X_true[0]
ub = X_true[-1]
total_points = len(x) * len(t)
N_u = 10000
idx = np.random.choice(total_points, N_u, replace = False)
U_true = usol.flatten('F')[:, None]
X_train_Nu = X_true[idx]
U_train_Nu = U_true[idx]
X_train_Nu = torch.from_numpy(X_train_Nu).float().to(device)
U_train_Nu = torch.from_numpy(U_train_Nu).float().to(device)
X_true = torch.from_numpy(X_true).float().to(device)
U_true = torch.from_numpy(U_true).float().to(device)
f_hat = torch.zeros(X_train_Nu.shape[0], 1).to(device)

lambda1 = 2.0
lambda2 = 0.02

layers = np.array([2, 20, 20, 20, 20, 20, 20, 20, 20, 1])
nu = 0.01/np.pi
PINN = FCN(layers)

params = list(PINN.dnn.parameters())

optimizer = torch.optim.LBFGS(params, 1e-1,
                              max_iter = 20000, 
                              max_eval = None, 
                              tolerance_grad = 1e-11, 
                              tolerance_change = 1e-11, 
                              history_size = 100,
                              line_search_fn = 'strong_wolfe'
                             )

optimizer.step(PINN.closure)
error_vec, u_pred = PINN.test()
solutionplot(u_pred, X_train_Nu.cpu().detach().numpy(), U_train_Nu)
x1 = X_true[:, 0]
t1 = X_true[:, 1]
arr_x1 = x1.reshape(shape=X.shape).transpose(1, 0).detach().cpu()
arr_T1 = t1.reshape(shape=X.shape).transpose(1, 0).detach().cpu()
plot3D_Matrix(arr_x1, arr_T1, torch.from_numpy(u_pred))