import numpy as np
import os
import scipy.io
import torch
import torch.autograd as autograd
import torch.nn as nn
import time

from plotting_utils import plot3D, plot3D_Matrix, solutionplot
from pyDOE import lhs
from torch.utils.tensorboard import SummaryWriter

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
    
# --- Neural Networks ---
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
        loss = self.loss(X_boundary, U_boundary, X_pde)
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            error_vec, _ = self.test()
            print("Iter: {}, Loss: {}, Error: {}".format(self.iter, loss.item(), error_vec))
            if hasattr(self, 'writer'):
                self.writer.add_scalar('Loss/train', loss.item(), self.iter)
        return loss

    def test(self):
        u_pred = self.forward(X_test)
        error_vec = torch.linalg.norm((u-u_pred),2)/torch.linalg.norm(u,2)
        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred,(256,100),order='F')
    
        return error_vec, u_pred

class FourierPINN(FCN):
    def __init__(self, layers, n_fourier_features=10, scale=10.0):
        fourier_input_dim = 2 * n_fourier_features * 2

        modified_layers = list(layers)
        modified_layers[0] = fourier_input_dim
        
        super().__init__(modified_layers)

        B_x = torch.randn((1, n_fourier_features)) * scale
        B_t = torch.randn((1, n_fourier_features)) * scale
        self.B_x = nn.Parameter(B_x, requires_grad=False).to(device)
        self.B_t = nn.Parameter(B_t, requires_grad=False).to(device)

    def fourier_mapping(self, x_tensor):
        x_coords = x_tensor[:, 0:1]
        t_coords = x_tensor[:, 1:2]
        
        proj_x = x_coords @ self.B_x
        proj_t = t_coords @ self.B_t
        
        features_x = torch.cat([torch.sin(proj_x), torch.cos(proj_x)], dim=1)
        features_t = torch.cat([torch.sin(proj_t), torch.cos(proj_t)], dim=1)
        
        return torch.cat([features_x, features_t], dim=1)

    def forward(self, x):
        if not torch.is_tensor(x):         
            x = torch.from_numpy(x).float().to(device)

        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)
        x_normalized = (x - l_b) / (u_b - l_b)

        features = self.fourier_mapping(x_normalized)
        a = features.float()
        for i in range(len(self.layers) - 1):
            z = self.layers[i](a)
            a = self.activation(z)

        return self.layers[-1](a)

# --- Data Preparation ---
# Automatically calculate the file path
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'Burgers.mat')
    print(f"Attempting to load file from the following absolute path: {data_path}")
    data = scipy.io.loadmat(data_path)

except NameError:
    print("Running in an interactive environment. Please ensure the working directory is correct or specify the path manually.")
    data_path = 'data/Burgers.mat'
    print(f"Attempting to use relative path: {data_path}")
    data = scipy.io.loadmat(data_path)

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

#Latin Hypercube Sampling
X_pde = lb + (ub - lb) * lhs(2, N_f)
X_pde = np.vstack((X_pde, X_boundary))  # Combine with boundary points

# --- Train ---
X_pde = torch.from_numpy(X_pde).float().to(device)
X_boundary = torch.from_numpy(X_boundary).float().to(device)
U_boundary = torch.from_numpy(U_boundary).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
u = torch.from_numpy(u_true).float().to(device)
f_hat = torch.zeros(X_pde.shape[0],1).to(device)

# FCN
print("\n--- Training Standard FCN Model ---")
pinn_fcn = FCN(layers)
pinn_fcn.to(device)
pinn_fcn.writer = SummaryWriter('runs/Standard_FCN') 

optimizer = torch.optim.LBFGS(pinn_fcn.parameters(), lr, 
                              max_iter = steps, 
                              max_eval = None, 
                              tolerance_grad = 1e-11, 
                              tolerance_change = 1e-11, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')
start_time = time.time()
optimizer.step(pinn_fcn.closure)
pinn_fcn.writer.close()
elapsed = time.time() - start_time                
print('FCN Training time: %.2f seconds' % (elapsed))
error_vec, _ = pinn_fcn.test()
print('FCN Test Error: %.5f'  % (error_vec))


# FourierPINN 
print("\n--- Training FourierPINN Model ---")
pinn_fourier = FourierPINN(layers)
pinn_fourier.to(device)
pinn_fourier.writer = SummaryWriter('runs/Fourier_PINN')

optimizer = torch.optim.LBFGS(pinn_fourier.parameters(), lr, 
                              max_iter = steps, 
                              max_eval = None, 
                              tolerance_grad = 1e-11, 
                              tolerance_change = 1e-11, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')
start_time = time.time()
optimizer.step(pinn_fourier.closure)
pinn_fourier.writer.close()
elapsed = time.time() - start_time                
print('FourierPINN Training time: %.2f seconds' % (elapsed))
error_vec, u_pred = pinn_fourier.test()
print('FourierPINN Test Error: %.5f'  % (error_vec))

# --- plots ---
solutionplot(u_pred, X_pde.cpu().detach().numpy(), usol, x, t)

x1=X_test[:,0]
t1=X_test[:,1]

arr_x1=x1.reshape(shape=X.shape).transpose(1,0).detach().cpu()
arr_T1=t1.reshape(shape=X.shape).transpose(1,0).detach().cpu()
arr_y1=u_pred
arr_y_test=usol

plot3D_Matrix(arr_x1,arr_T1,torch.from_numpy(arr_y1))
plot3D(torch.from_numpy(x),torch.from_numpy(t),torch.from_numpy(usol)) 