import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

import numpy as np
import time
import scipy.io
import os

from plotting_utils import plot3D, plot3D_Matrix, solutionplot

"""
Burgers' 方程的傅里葉变换版本: 
u_t + u * u_x - nu * u_xx = 0, u(x, 0) = -sin(pi*x)
核心思想: 神经网络预测频域系数, 通过 IFFT 得到时空域的解。
"""

# --- 基础设置 ---
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("正在使用设备: ", device)

# --- 参数 ---
steps = 20000  # Adam优化器通常需要更多的迭代次数
lr = 1e-3
# **改动1: 网络的最后一层改为2, 用于输出复数的实部和虚部**
layers = np.array([2, 30, 30, 30, 30, 30, 30, 2]) 
N_boundary = 100 
nu = 0.01 / np.pi  # 扩散系数

# --- 神经网络 ---
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

    def forward(self, x_and_t_grid):
        """
        **改动2: 核心前向传播函数。**
        输入 (x,t) 网格, 输出经过 IFFT 变换后的解 u(x,t)。
        """
        if not torch.is_tensor(x_and_t_grid):         
            x_and_t_grid = torch.from_numpy(x_and_t_grid).float().to(device)

        # 归一化输入
        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)
        x_and_t_grid = (x_and_t_grid - l_b) / (u_b - l_b)

        # 通过神经网络得到频域系数的实部和虚部
        a = x_and_t_grid.float()
        for i in range(len(self.layers) - 1):
            z = self.layers[i](a)
            a = self.activation(z)
        freq_coeffs_pairs = self.layers[-1](a)
        
        # 组合成复数
        freq_coeffs_complex = torch.complex(freq_coeffs_pairs[:, 0:1], freq_coeffs_pairs[:, 1:2])
        
        # 获取网格尺寸 (这是为了 ifft2)
        nx = len(torch.unique(x_and_t_grid[:, 0]))
        nt = len(torch.unique(x_and_t_grid[:, 1]))

        # 将系数向量重塑为 2D 网格
        freq_grid = freq_coeffs_complex.reshape(nt, nx)
        
        # 进行二维傅里葉逆变换得到解
        solution_grid_complex = torch.fft.ifft2(freq_grid)
        
        # 我们只关心实部解
        solution_real = solution_grid_complex.real
        
        # 将解展平以便计算损失
        return solution_real.flatten()[:, None]

    def loss_boundary(self, x_b, u_b):
        """
        **改动3: 边界损失的计算方式。**
        我们需要在完整的物理网格上进行预测, 然后挑选出边界点进行比较。
        """
        # 在完整的物理网格上进行前向传播
        u_pred_full = self.forward(X_train_Nf)
        
        # 从完整解中提取边界点对应的预测值
        # 注意: 这需要 X_boundary 是 X_train_Nf 的子集
        # 为了简单起见，我们直接用真值索引来演示
        # 在实际应用中，需要一个更鲁棒的查找方法
        u_pred_on_boundary = u_pred_full[boundary_indices]
        
        return self.loss_function(u_pred_on_boundary, u_b)

    def loss_PDE(self, x_f):
        # clone() 和 requires_grad=True 是为了能够计算高阶导数
        g = x_f.clone()
        g.requires_grad = True
        
        # 经过 forward 得到的 u 已经是 ifft 的结果
        u = self.forward(g)
        
        # 使用 autograd 计算 u 对 g (即 x 和 t) 的导数
        u_x_t = autograd.grad(u, g, torch.ones_like(u).to(device), create_graph=True)[0]
        u_xx_tt = autograd.grad(u_x_t, g, torch.ones_like(u_x_t).to(device), create_graph=True)[0]

        u_x = u_x_t[:, [0]]
        u_t = u_x_t[:, [1]]
        u_xx = u_xx_tt[:, [0]]

        # 代入 Burgers' 方程
        f = u_t + u * u_x - nu * u_xx
        
        # 目标是让 f 趋近于 0
        loss_f = self.loss_function(f, torch.zeros_like(f))
        return loss_f
    
# --- 数据准备 ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'Burgers.mat')
    data = scipy.io.loadmat(data_path)
except Exception:
    data_path = 'data/Burgers.mat' # 适用于 Colab 或类似环境
    data = scipy.io.loadmat(data_path)

x_data = data['x']  # shape (256, 1)
t_data = data['t']  # shape (100, 1)
usol = data['usol'] # shape (256, 100)

X_grid, T_grid = np.meshgrid(x_data.flatten(), t_data.flatten())
X_test_grid = np.hstack((X_grid.flatten()[:, None], T_grid.flatten()[:, None]))
lb = X_test_grid.min(axis=0)
ub = X_test_grid.max(axis=0)
u_true = usol.flatten('F')[:, None]

# 边界条件
# t = 0 (初始条件)
initial_X = np.hstack((x_data, np.zeros_like(x_data)))
initial_U = usol[:, 0][:, None]
# x = -1 (左边界)
left_boundary_X = np.hstack((np.full_like(t_data, -1.0), t_data))
left_boundary_U = usol[0, :][:, None]
# x = 1 (右边界)
right_boundary_X = np.hstack((np.full_like(t_data, 1.0), t_data))
right_boundary_U = usol[-1, :][:, None]

# 合并所有边界点
X_boundary_all = np.vstack([initial_X, left_boundary_X, right_boundary_X])
U_boundary_all = np.vstack([initial_U, left_boundary_U, right_boundary_U])

# 从所有边界点中随机选择 N_boundary 个点
idx = np.random.choice(X_boundary_all.shape[0], N_boundary, replace=False)
X_boundary = X_boundary_all[idx, :]
U_boundary = U_boundary_all[idx, :]

# **改动4: 创建一个规则的物理点网格, 而不是使用 LHS**
X_train_Nf = X_test_grid

# 找到边界点在完整物理网格中的索引
# 这是一个简化的查找，实际中可能需要更精确的匹配
boundary_indices = [np.where((X_train_Nf == xb).all(axis=1))[0][0] for xb in X_boundary]

# --- 训练 ---
X_train_Nf = torch.from_numpy(X_train_Nf).float().to(device)
X_boundary = torch.from_numpy(X_boundary).float().to(device)
U_boundary = torch.from_numpy(U_boundary).float().to(device)
X_test = torch.from_numpy(X_test_grid).float().to(device)
u = torch.from_numpy(u_true).float().to(device)

PINN = FCN(layers)
PINN.to(device)

# **改动5: 使用 Adam 优化器**
optimizer = torch.optim.Adam(PINN.parameters(), lr=lr)

start_time = time.time()
for i in range(steps + 1):
    optimizer.zero_grad()
    
    # 计算损失
    loss_b = PINN.loss_boundary(X_boundary, U_boundary)
    loss_f = PINN.loss_PDE(X_train_Nf)
    loss = loss_b + loss_f
    
    loss.backward()
    optimizer.step()
    
    if i % 500 == 0:
        # 使用完整的测试集来评估误差
        u_pred_test = PINN.forward(X_test)
        error_vec = torch.linalg.norm((u - u_pred_test), 2) / torch.linalg.norm(u, 2)
        print(f"Iter: {i}, Loss: {loss.item():.4e}, Test Error: {error_vec.item():.4e}")

elapsed = time.time() - start_time
print(f'训练用时: {elapsed:.2f} 秒')

# --- 测试和绘图 ---
u_pred_numpy = PINN.forward(X_test).cpu().detach().numpy()
error_vec = torch.linalg.norm((u - torch.from_numpy(u_pred_numpy).float().to(device)), 2) / torch.linalg.norm(u, 2)
print(f'最终测试误差: {error_vec.item():.5f}')

# 为了绘图, 需要将预测结果重塑为 (nx, nt)
u_pred_grid = u_pred_numpy.reshape(len(t_data), len(x_data)).T # 转置以匹配 usol 的 (nx, nt) 格式

# 绘制预测解
plot3D_Matrix(
    torch.from_numpy(X_grid.T),
    torch.from_numpy(T_grid.T),
    torch.from_numpy(u_pred_grid)
)
# 绘制真实解
plot3D(
    torch.from_numpy(x_data),
    torch.from_numpy(t_data),
    torch.from_numpy(usol)
)