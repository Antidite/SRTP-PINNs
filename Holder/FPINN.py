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

#
class WaveletFeatures(nn.Module):

    def __init__(self, centers: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.register_buffer("centers", centers)  # (M,2)
        self.register_buffer("scales", scales)    # (M,2)

    def bump(self, xi):

        out = torch.zeros_like(xi)
        mask = (xi.abs() < 1.0)
        z = xi[mask]
        out[mask] = torch.exp(-1.0 / (1.0 - z**2))
        return out

    def wavelet_1d(self, xi):

        b = self.bump(xi)
        return (1.0 - 2.0 * xi**2) * b

    def forward(self, X):

        x = X[:, 0:1]  # (N,1)
        t = X[:, 1:2]  # (N,1)

        # centers: (M,2), scales: (M,2)
        cx = self.centers[:, 0].unsqueeze(0)  # (1,M)
        ct = self.centers[:, 1].unsqueeze(0)  # (1,M)
        sx = self.scales[:, 0].unsqueeze(0)   # (1,M)
        st = self.scales[:, 1].unsqueeze(0)   # (1,M)

        xi_x = (x - cx) / sx
        xi_t = (t - ct) / st

        psi_x = self.wavelet_1d(xi_x)
        psi_t = self.wavelet_1d(xi_t)

        features = psi_x * psi_t  # (N,M)
        return features


# Local Wavelet PINN
class WavePINN(nn.Module):
    def __init__(self, wavelet_features: WaveletFeatures, M: int):
        super().__init__()
        self.wavelet_features = wavelet_features
        self.linear = nn.Linear(M, 1, bias=False)

    def forward(self, X):
        phi = self.wavelet_features(X)  # (N,M)
        return self.linear(phi)         # (N,1)



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

def u_total(X):
    # X: (N,2) tensor
    with torch.no_grad():
        u_global = pinn_fourier(X)
    delta_u = wavelet_correction(X)
    return u_global + delta_u

def pde_residual_wavelet(X):
    X.requires_grad_(True)
    u = u_total(X)
    grads = autograd.grad(u, X, torch.ones_like(u),
                          create_graph=True, retain_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]

    grads2 = autograd.grad(u_x, X, torch.ones_like(u_x),
                           create_graph=True, retain_graph=True)[0]
    u_xx = grads2[:, 0:1]

    f = u_t + u * u_x - nu * u_xx
    return f

def compute_region_errors(u_pred_mat, usol_mat, x_grid, t_grid, lb_patch, ub_patch):

    x_vec = x_grid.flatten()
    t_vec = t_grid.flatten()

    X_mat, T_mat = np.meshgrid(x_vec, t_vec, indexing='ij')

    mask_patch = (
        (X_mat >= lb_patch[0]) & (X_mat <= ub_patch[0]) &
        (T_mat >= lb_patch[1]) & (T_mat <= ub_patch[1])
    )

    diff_all = usol_mat - u_pred_mat
    err_all = np.linalg.norm(diff_all) / np.linalg.norm(usol_mat)

    diff_patch = diff_all[mask_patch]
    true_patch = usol_mat[mask_patch]
    err_patch = np.linalg.norm(diff_patch) / np.linalg.norm(true_patch)

    diff_out = diff_all[~mask_patch]
    true_out = usol_mat[~mask_patch]
    err_out = np.linalg.norm(diff_out) / np.linalg.norm(true_out)

    return err_all, err_patch, err_out, mask_patch

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

print("\n--- Baseline (before wavelet correction) ---")
# u_pred: (Nx,Nt) ； usol: (Nx,Nt)
indicator = wavelet_shock_indicator(u_pred, wavelet='db4', level=3)

lb_patch, ub_patch = detect_shock_patch(
    indicator,
    x,
    t,
    threshold=0.5,
    x_margin_ratio=0.1,
    t_margin_ratio=0.1
)

err_all_base, err_patch_base, err_out_base, mask_patch = compute_region_errors(
    u_pred_mat=u_pred,
    usol_mat=usol,
    x_grid=x,
    t_grid=t,
    lb_patch=lb_patch,
    ub_patch=ub_patch
)
print("Baseline shock patch lb =", lb_patch, ", ub =", ub_patch)
print(f"[Baseline] Global  rel L2 error : {err_all_base:.5e}")
print(f"[Baseline] Patch   rel L2 error : {err_patch_base:.5e}")
print(f"[Baseline] Outside rel L2 error : {err_out_base:.5e}")


# Wavelet
print("\n--- Wavelet-based shock detection & localized correction ---")
Nx, Nt = u_pred.shape
x_vec = x
t_vec = t
indicator = wavelet_shock_indicator(u_pred, wavelet='db4', level=3)
lb_patch, ub_patch = detect_shock_patch(indicator, x, t,threshold=0.5,x_margin_ratio=0.1,t_margin_ratio=0.1)
Nx_c, Nt_c = 8, 4  # the center of wavelet
x_centers = np.linspace(lb_patch[0], ub_patch[0], Nx_c)
t_centers = np.linspace(lb_patch[1], ub_patch[1], Nt_c)

centers_list = []
for xc in x_centers:
    for tc in t_centers:
        centers_list.append([xc, tc])
centers_np = np.array(centers_list, dtype=np.float32)
M = centers_np.shape[0]

x_scale = (ub_patch[0] - lb_patch[0]) / Nx_c
t_scale = (ub_patch[1] - lb_patch[1]) / Nt_c
scales_np = np.tile(np.array([[x_scale, t_scale]], dtype=np.float32), (M, 1))

centers_tensor = torch.from_numpy(centers_np).to(device)
scales_tensor = torch.from_numpy(scales_np).to(device)

wavelet_features = WaveletFeatures(centers_tensor, scales_tensor).to(device)
wavelet_correction = WavePINN(wavelet_features, M).to(device)

for p in pinn_fourier.parameters():
    p.requires_grad = False
optimizer_w = torch.optim.Adam(wavelet_correction.parameters(), lr=1e-3)
lambda_b = 1.0  # bound loss
steps_wavelet = 5000
for it in range(steps_wavelet):
    optimizer_w.zero_grad()

    f_pde = pde_residual_wavelet(X_pde)
    loss_pde = torch.mean(f_pde**2)

    u_b = u_total(X_boundary)
    loss_b = torch.mean((u_b - U_boundary)**2)

    loss = loss_pde + lambda_b * loss_b
    loss.backward()
    optimizer_w.step()

    if it % 500 == 0:
        print(f"[Wavelet] iter {it}, loss={loss.item():.3e}, "
              f"pde={loss_pde.item():.3e}, b={loss_b.item():.3e}")

print("--- Wavelet correction finished ---")

with torch.no_grad():
    u_pred_total = u_total(X_test).cpu().numpy()
u_pred_total = np.reshape(u_pred_total, (256, 100), order='F')

err_all_w, err_patch_w, err_out_w, _ = compute_region_errors(
    u_pred_mat=u_pred_total,
    usol_mat=usol,
    x_grid=x,
    t_grid=t,
    lb_patch=lb_patch,
    ub_patch=ub_patch
)

print("=== After wavelet correction ===")
print(f"[Wavelet] Global  rel L2 error : {err_all_w:.5e}")
print(f"[Wavelet] Patch   rel L2 error : {err_patch_w:.5e}")
print(f"[Wavelet] Outside rel L2 error : {err_out_w:.5e}")

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