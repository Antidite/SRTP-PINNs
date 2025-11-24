import numpy as np
import os
import scipy.io
import torch
import torch.autograd as autograd
import torch.nn as nn
import time
import pywt

from plotting_utilis import plot3D, plot3D_Matrix, solutionplot
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
steps = 10000
lr = 1e-1
layers = np.array([2, 20, 20, 20, 20, 20, 20, 20, 20, 1])
N_boundary = 100
N_f = 10000
nu = 0.01 / np.pi  # diffusion coefficient

# --- Neural Networks ---
class FourierPINN(nn.Module):
    def __init__(self, layers, n_fourier_features=10, scale=10.0):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')

        fourier_input_dim = 2 * n_fourier_features * 2
        modified_layers = list(layers)
        modified_layers[0] = fourier_input_dim

        self.layers = nn.ModuleList(
            [nn.Linear(modified_layers[i], modified_layers[i+1])
             for i in range(len(modified_layers)-1)]
        )

        for i in range(len(self.layers)):
            nn.init.xavier_normal_(self.layers[i].weight.data, gain=1.0)
            nn.init.zeros_(self.layers[i].bias.data)

        B_x = torch.randn((1, n_fourier_features)) * scale
        B_t = torch.randn((1, n_fourier_features)) * scale
        self.B_x = nn.Parameter(B_x, requires_grad=False).to(device)
        self.B_t = nn.Parameter(B_t, requires_grad=False).to(device)

        self.iter = 0

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

    def loss_boundary(self, x, u):
        u_pred = self.forward(x)
        return self.loss_function(u_pred, u)

    def loss_PDE(self, X_train_Nf):
        g = X_train_Nf.clone()
        g.requires_grad = True
        u = self.forward(g)
        u_x_t = autograd.grad(u, g, torch.ones([X_train_Nf.shape[0], 1]).to(device),
                              retain_graph=True, create_graph=True)[0]
        u_xx_tt = autograd.grad(u_x_t, g, torch.ones(X_train_Nf.shape).to(device),
                                create_graph=True)[0]
        u_x = u_x_t[:, [0]]
        u_t = u_x_t[:, [1]]
        u_xx = u_xx_tt[:, [0]]

        f = u_t + (self.forward(g)) * (u_x) - (nu) * u_xx
        loss_f = self.loss_function(f, f_hat)

        return loss_f

    def loss(self, x, y, X_train_Nf):
        loss_u = self.loss_boundary(x, y)
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
        error_vec = torch.linalg.norm((u - u_pred), 2) / torch.linalg.norm(u, 2)
        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred, (256, 100), order='F')

        return error_vec, u_pred

# Local Wavelet PINN
class WavePINN(nn.Module):
    def __init__(self, layers, nu, device):
        super().__init__()
        self.nu = nu
        self.device = device
        self.activation = nn.Tanh()
        self.loss_fn = nn.MSELoss()

        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )
        for l in self.layers:
            nn.init.xavier_normal_(l.weight)
            nn.init.zeros_(l.bias)

    def forward(self, x):
        a = x
        for l in self.layers[:-1]:
            a = self.activation(l(a))
        return self.layers[-1](a)

    def pde_residual(self, X_interior):
        X_interior.requires_grad_(True)
        u = self.forward(X_interior)
        grads = autograd.grad(u, X_interior, torch.ones_like(u),create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        grads2 = autograd.grad(u_x, X_interior, torch.ones_like(u_x),create_graph=True)[0]
        u_xx = grads2[:, 0:1]

        f = u_t + u * u_x - self.nu * u_xx
        return f



# --- Wavelet ---
def wavelet_shock_indicator(u_grid, wavelet='db4', level=3):

    Nx, Nt = u_grid.shape
    indicator = np.zeros_like(u_grid)

    for j in range(Nt):
        u_slice = u_grid[:, j]

        # wavelet decomposition
        coe = pywt.wavedec(u_slice, wavelet, level=level)

        detail_coe = [np.zeros_like(c) for c in coe]
        for l in range(1, len(coe)):
            detail_coe[l] = coe[l]

        detail_signal = pywt.waverec(detail_coe, wavelet)

        detail_signal = detail_signal[:Nx]

        indicator[:, j] = np.abs(detail_signal)

    indicator /= (indicator.max() + 1e-12)
    return indicator


def detect_shock_patch(indicator, x, t, threshold=0.5,x_margin_ratio=0.1, t_margin_ratio=0.1):

    x_vec = x.flatten()
    t_vec = t.flatten()

    idx_x, idx_t = np.where(indicator > threshold)
    if len(idx_x) == 0:
        max_id = np.argmax(indicator)
        i_max, j_max = np.unravel_index(max_id, indicator.shape)
        idx_x = np.array([i_max])
        idx_t = np.array([j_max])

    x_min = x_vec[idx_x].min()
    x_max = x_vec[idx_x].max()
    t_min = t_vec[idx_t].min()
    t_max = t_vec[idx_t].max()

    xL, xR = x[0,0], x[-1,0]
    tL, tR = t[0,0], t[-1,0]
    x_margin = (xR - xL) * x_margin_ratio
    t_margin = (tR - tL) * t_margin_ratio

    x_min = max(xL, x_min - x_margin)
    x_max = min(xR, x_max + x_margin)
    t_min = max(tL, t_min - t_margin)
    t_max = min(tR, t_max + t_margin)

    lb_patch = np.array([x_min, t_min])
    ub_patch = np.array([x_max, t_max])
    print("Detected patch:", lb_patch, ub_patch)
    return lb_patch, ub_patch

def sample_points_in_patch(lb_patch, ub_patch, N_interior):
    X_rand = np.random.rand(N_interior, 2)
    X_patch = lb_patch + (ub_patch - lb_patch) * X_rand
    return X_patch
def sample_patch_boundary(lb_patch, ub_patch, N_each_side):
    x_min, t_min = lb_patch
    x_max, t_max = ub_patch

    # x = x_min, t in [t_min, t_max]
    t_vals = np.linspace(t_min, t_max, N_each_side)
    left = np.stack([np.full_like(t_vals, x_min), t_vals], axis=1)

    # x = x_max
    right = np.stack([np.full_like(t_vals, x_max), t_vals], axis=1)

    # t = t_min
    x_vals = np.linspace(x_min, x_max, N_each_side)
    bottom = np.stack([x_vals, np.full_like(x_vals, t_min)], axis=1)

    # t = t_max
    top = np.stack([x_vals, np.full_like(x_vals, t_max)], axis=1)

    X_b = np.vstack([left, right, bottom, top])
    return X_b

# --- Data Preparation ---
# Automatically calculate the file path
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'Burgers.mat')
    print(f"Attempting to load file from the following absolute path: {data_path}")
    data = scipy.io.loadmat(data_path)

except NameError:
    print(
        "Running in an interactive environment. Please ensure the working directory is correct or specify the path manually.")
    data_path = 'data/Burgers.mat'
    print(f"Attempting to use relative path: {data_path}")
    data = scipy.io.loadmat(data_path)

x = data['x']
t = data['t']
usol = data['usol']

X, T = np.meshgrid(x, t)
plot3D(torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(usol))

X_test = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
lb = X_test[0]
ub = X_test[-1]
u_true = usol.flatten('F')[:, None]

# Boundary Conditions
# -1 <= x <= 1 and t = 0
left_X = np.hstack((X[0, :][:, None], T[0, :][:, None]))
left_U = usol[:, 0][:, None]

# x = -1 and 0 <= t <= 1
bottom_X = np.hstack((X[:, 0][:, None], T[:, 0][:, None]))  # L2
bottom_U = usol[-1, :][:, None]

# x = 1 and 0 <= t <= 1
top_X = np.hstack((X[:, -1][:, None], T[:, 0][:, None]))  # L3
top_U = usol[0, :][:, None]

X_train = np.vstack([left_X, bottom_X, top_X])
U_train = np.vstack([left_U, bottom_U, top_U])

idx = np.random.choice(X_train.shape[0], N_boundary, replace=False)
X_boundary = X_train[idx, :]
U_boundary = U_train[idx, :]

# Latin Hypercube Sampling
X_pde = lb + (ub - lb) * lhs(2, N_f)
X_pde = np.vstack((X_pde, X_boundary))  # Combine with boundary points

# --- Train ---
X_pde = torch.from_numpy(X_pde).float().to(device)
X_boundary = torch.from_numpy(X_boundary).float().to(device)
U_boundary = torch.from_numpy(U_boundary).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
u = torch.from_numpy(u_true).float().to(device)
f_hat = torch.zeros(X_pde.shape[0], 1).to(device)

# FourierPINN
print("\n--- Training FourierPINN Model ---")
pinn_fourier = FourierPINN(layers)
pinn_fourier.to(device)
pinn_fourier.writer = SummaryWriter('runs/Fourier_PINN')

optimizer = torch.optim.LBFGS(pinn_fourier.parameters(), lr,
                              max_iter=steps,
                              max_eval=None,
                              tolerance_grad=1e-11,
                              tolerance_change=1e-11,
                              history_size=100,
                              line_search_fn='strong_wolfe')
start_time = time.time()
optimizer.step(pinn_fourier.closure)
pinn_fourier.writer.close()
elapsed = time.time() - start_time
print('FourierPINN Training time: %.2f seconds' % (elapsed))
error_vec, u_pred = pinn_fourier.test()
print('FourierPINN Test Error: %.5f' % (error_vec))

# Wavelet
print("\n--- Wavelet-based shock detection & local refinement ---")
Nx, Nt = u_pred.shape
x_vec = x
t_vec = t
indicator = wavelet_shock_indicator(u_pred, wavelet='db4', level=3)
lb_patch, ub_patch = detect_shock_patch(indicator, x, t,threshold=0.5,x_margin_ratio=0.1,t_margin_ratio=0.1)
X_interior_np = sample_points_in_patch(lb_patch, ub_patch, N_interior=4000)
X_b_np = sample_patch_boundary(lb_patch, ub_patch, N_each_side=100)
X_interior = torch.from_numpy(X_interior_np).float().to(device)
X_b = torch.from_numpy(X_b_np).float().to(device)
with torch.no_grad():
    U_b = pinn_fourier.forward(X_b)
local_layers = [2, 64, 64, 64, 1]
local_pinn = WavePINN(local_layers, nu=nu, device=device).to(device)
optimizer = torch.optim.Adam(local_pinn.parameters(), lr=1e-3)

for it in range(5000):
    optimizer.zero_grad()

    # PDE loss
    f_interior = local_pinn.pde_residual(X_interior)
    loss_pde = torch.mean(f_interior**2)

    # boundary loss
    u_b_pred = local_pinn(X_b)
    loss_b = torch.mean((u_b_pred - U_b)**2)

    loss = loss_pde + loss_b
    loss.backward()
    optimizer.step()

    if it % 500 == 0:
        print(f"[Local] iter {it}, loss={loss.item():.3e}, pde={loss_pde.item():.3e}, b={loss_b.item():.3e}")

# Combination
X_test_np = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
X_test = torch.from_numpy(X_test_np).float().to(device)
with torch.no_grad():
    u_global = pinn_fourier.forward(X_test).cpu().numpy()
u_refined = u_global.copy()
x_flat = X_test_np[:,0]
t_flat = X_test_np[:,1]
mask = (
    (x_flat >= lb_patch[0]) & (x_flat <= ub_patch[0]) &
    (t_flat >= lb_patch[1]) & (t_flat <= ub_patch[1])
)
X_patch_all = X_test[mask, :]
with torch.no_grad():
    u_local_patch = local_pinn(X_patch_all).cpu().numpy()
u_refined[mask, :] = u_local_patch
u_refined_grid = u_refined.reshape(X.shape[0], X.shape[1], order='F')

# --- plots ---
solutionplot(u_pred, X_pde.cpu().detach().numpy(), usol, x, t)

x1 = X_test[:, 0]
t1 = X_test[:, 1]

arr_x1 = x1.reshape(shape=X.shape).transpose(1, 0).detach().cpu()
arr_T1 = t1.reshape(shape=X.shape).transpose(1, 0).detach().cpu()
arr_y1 = u_pred
arr_y_test = usol

plot3D_Matrix(arr_x1, arr_T1, torch.from_numpy(arr_y1))
plot3D(torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(usol))