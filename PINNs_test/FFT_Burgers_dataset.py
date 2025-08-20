import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from sklearn.model_selection import train_test_split

import numpy as np
import time
from pyDOE import lhs
import scipy.io

from plotting_utils import plot3D, plot3D_Matrix, solutionplot

"""
Form of the Burgers' equation: u_t + u * u_x - nu * u_xx = 0, u(x, 0) = -sin(pi*x)
"""

# --- Basic Settings ---
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: ", device)

# --- Parameters ---
steps=10000
lr=1e-1
layers = np.array([2,20,20,20,20,20,20,20,20,1]) 
N_boundary = 100 
N_f = 10000 
nu = 0.01/np.pi #diffusion coefficient

# --- Neural Network ---
class FCN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.iter = 0
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.layers[i].weight.data, gain=1.0)
            nn.init.zeros_(self.layers[i].bias.data)

    def forward(self, x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)

        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)

        x = (x - l_b)/(u_b - l_b)

        a = x.float()
        for i in range(len(self.layers) - 1):
            z = self.layers[i](a)
            a = self.activation(z)

        return self.layers[-1](a)
    
    def loss_boundary(self, x, u):
        u_pred = self.forward(x)
        return self.loss_function(u_pred, u)
    
    def loss_PDE(self, X_train_Nf):
        g = X_train_Nf.clone()
        g.requires_grad = True
        u = self.forward(g)
        u_x_t = autograd.grad(u, g, torch.ones([X_train_Nf.shape[0], 1]).to(device), 
                              retain_graph=True, create_graph=True)[0]
        u_xx_tt = autograd.grad(u_x_t,g,torch.ones(X_train_Nf.shape).to(device), 
                                create_graph=True)[0]
        u_x = u_x_t[:,[0]]
        u_t = u_x_t[:,[1]]
        u_xx = u_xx_tt[:,[0]]

        f = u_t + (self.forward(g)) * (u_x) - (nu) * u_xx
        loss_f = self.loss_function(f, f_hat)

        return loss_f
    
    def loss(self, x, y, X_train_Nf):
        loss_u = self.loss_boundary(x,y)
        loss_f = self.loss_PDE(X_train_Nf)
        
        return loss_u + loss_f
    
    def closure(self):
        optimizer.zero_grad()
        loss = self.loss(X_boundary, U_boundary, X_train_Nf)
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            error_vec, _ = PINN.test()
            print("Iter: {}, Loss: {}, Error: {}".format(self.iter, loss.item(), error_vec))
        return loss

    def test(self):
        u_pred = self.forward(X_test)
        error_vec = torch.linalg.norm((u-u_pred),2)/torch.linalg.norm(u,2)
        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred,(256,100),order='F')
    
        return error_vec, u_pred
    
# --- Data Preparation ---
data = scipy.io.loadmat('./data/Burgers.mat')
x = data['x']
t = data['t']
usol = data['usol']

X, T = np.meshgrid(x,t)                         
plot3D(torch.from_numpy(x),torch.from_numpy(t),torch.from_numpy(usol)) 
     
X_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
lb = X_test[0]  
ub = X_test[-1] 
u_true = usol.flatten('F')[:,None]

# Boundary Conditions
# -1 <= x <= 1 and t = 0
left_X = np.hstack((X[0, :][:, None], T[0, :][:, None]))
left_U = usol[:, 0][:, None]

# x = -1 and 0 <= t <= 1
bottom_X = np.hstack((X[:,0][:,None], T[:,0][:,None])) #L2
bottom_U = usol[-1,:][:,None]

# x = 1 and 0 <= t <= 1
top_X = np.hstack((X[:,-1][:,None], T[:,0][:,None])) #L3
top_U = usol[0,:][:,None]

X_train = np.vstack([left_X, bottom_X, top_X]) 
U_train = np.vstack([left_U, bottom_U, top_U]) 

idx = np.random.choice(X_train.shape[0], N_boundary, replace=False) 
X_boundary = X_train[idx, :]
U_boundary = U_train[idx, :]

#Latin Hypercube sampling
X_train_Nf = lb + (ub - lb) * lhs(2, N_f)
X_train_Nf = np.vstack((X_train_Nf, X_boundary))  # Combine with boundary points

# --- Train ---
X_train_Nf = torch.from_numpy(X_train_Nf).float().to(device)
X_boundary = torch.from_numpy(X_boundary).float().to(device)
U_boundary = torch.from_numpy(U_boundary).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
u = torch.from_numpy(u_true).float().to(device)
f_hat = torch.zeros(X_train_Nf.shape[0],1).to(device)

PINN = FCN(layers)
PINN.to(device)

optimizer = torch.optim.LBFGS(PINN.parameters(), lr, 
                              max_iter = steps, 
                              max_eval = None, 
                              tolerance_grad = 1e-11, 
                              tolerance_change = 1e-11, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')
start_time = time.time()
optimizer.step(PINN.closure)
elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))

error_vec, u_pred = PINN.test()
print('Test Error: %.5f'  % (error_vec))

# --- plots ---
solutionplot(u_pred, X_train_Nf.cpu().detach().numpy(), usol, x, t)

x1=X_test[:,0]
t1=X_test[:,1]

arr_x1=x1.reshape(shape=X.shape).transpose(1,0).detach().cpu()
arr_T1=t1.reshape(shape=X.shape).transpose(1,0).detach().cpu()
arr_y1=u_pred
arr_y_test=usol

plot3D_Matrix(arr_x1,arr_T1,torch.from_numpy(arr_y1))
plot3D(torch.from_numpy(x),torch.from_numpy(t),torch.from_numpy(usol)) 