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

# --- 设置 ---
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前使用的设备:", device)

# --- 绘图函数 ---
def plot3D(x, t, y):
    x_np = x if isinstance(x, np.ndarray) else x.cpu().numpy()
    t_np = t if isinstance(t, np.ndarray) else t.cpu().numpy()
    y_np = y if isinstance(y, np.ndarray) else y.cpu().numpy()

    x_squeezed = x_np.squeeze()
    t_squeezed = t_np.squeeze()
    
    if x_squeezed.ndim == 0: x_squeezed = np.array([x_squeezed])
    if t_squeezed.ndim == 0: t_squeezed = np.array([t_squeezed])

    X, T = np.meshgrid(x_squeezed, t_squeezed)
    F_xt = y_np

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(T, X, F_xt.T, 20, cmap='rainbow')
    fig.colorbar(cp)
    ax.set_title('F(x, t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()

    ax = plt.axes(projection='3d')
    ax.plot_surface(T, X, F_xt.T, cmap='rainbow')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u(x, t)')
    plt.show()


def solutionplot(u_pred, X_f_train, u_exact, x, t):
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(u_pred.T, interpolation='nearest', cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_f_train[:,1], X_f_train[:,0], 'kx', label = 'Collocation Points (%d)' % (X_f_train.shape[0]), markersize = 0.5, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(x,t)$', fontsize = 10)
    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, u_exact[:,25], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x, u_pred[:,25], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')    
    ax.set_title('$t = 0.25s$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, u_exact[:,50], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x, u_pred[:,50], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50s$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, u_exact[:,75], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x, u_pred[:,75], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75s$', fontsize = 10)
    
    plt.savefig('Burgers-Forward.png', dpi=500)
    plt.show()

# --- 神经网络定义 ---
class DNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers)-1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain = 1.0)
            nn.init.zeros_(self.linears[i].bias.data)
            
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        
        x_normalized = 2.0 * (x - lb_T) / (ub_T - lb_T) - 1.0
        
        a = x_normalized.float()
        for i in range(len(layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

# --- PINN 核心类 ---
class FCN():
    def __init__(self, layers, lambda1, lambda2):
        self.iter = 0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        self.dnn = DNN(layers).to(device)
        
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(), 
            max_iter=20000, 
            max_eval=None, 
            tolerance_grad=1e-11, 
            tolerance_change=1e-11, 
            history_size=100,
            line_search_fn='strong_wolfe'
        )
        
        self.loss_function = nn.MSELoss(reduction='mean')

    def loss_ic(self, x, u):
        return self.loss_function(self.dnn(x), u)

    def loss_bc(self, x, u):
        return self.loss_function(self.dnn(x), u)

    def loss_pde(self, X_f):
        g = X_f.clone()
        g.requires_grad = True
        
        u = self.dnn(g)
        
        u_x_t = autograd.grad(u, g, torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
        u_xx_tt = autograd.grad(u_x_t, g, torch.ones(g.shape).to(device), create_graph=True)[0]
        
        u_t = u_x_t[:, [1]]
        u_x = u_x_t[:, [0]]
        u_xx = u_xx_tt[:, [0]]
        
        f = u_t + self.lambda1 * u * u_x - self.lambda2 * u_xx
        
        f_hat = torch.zeros(f.shape[0], 1).to(device)
        return self.loss_function(f, f_hat)

    def loss(self, x_ic, u_ic, x_bc, u_bc, x_f):
        loss_i = self.loss_ic(x_ic, u_ic)
        loss_b = self.loss_bc(x_bc, u_bc)
        loss_f = self.loss_pde(x_f)
        
        loss_val = loss_i + loss_b + loss_f
        return loss_val

    def closure(self):
        self.optimizer.zero_grad()
        loss = self.loss(X_ic_T, u_ic_T, X_bc_T, U_bc_T, X_f_T)
        loss.backward()
        
        self.iter += 1
        if self.iter % 100 == 0:
            error_vec, _ = self.test()
            print(
                'Iter: %d, Loss: %.3e, Relative Error: %.3e' %
                (self.iter, loss.item(), error_vec.cpu().detach().numpy())
            )
        return loss
        
    def train(self):
        self.dnn.train()
        self.optimizer.step(self.closure)

    def test(self):
        self.dnn.eval()
        u_pred = self.dnn(X_true_T)
        error_vec = torch.linalg.norm((U_true_T - u_pred), 2) / torch.linalg.norm(U_true_T, 2)
        u_pred_grid = u_pred.cpu().detach().numpy().reshape(len(x), len(t))
        return error_vec, u_pred_grid

# --- 主程序 ---
if __name__ == "__main__":
    # 定义问题参数
    nu = 0.01 / np.pi
    lambda1 = 1.0
    layers = np.array([2, 20, 20, 20, 20, 20, 20, 20, 20, 1])
    # 定义求解域
    x_domain = np.array([-1.0, 1.0])
    t_domain = np.array([0.0, 1.0])
    lb = np.array([x_domain[0], t_domain[0]])
    ub = np.array([x_domain[1], t_domain[1]])
    lb_T = torch.from_numpy(lb).float().to(device)
    ub_T = torch.from_numpy(ub).float().to(device)
    # 准备训练数据
    N_ic, N_bc, N_f = 100, 100, 20000
    x_ic = np.linspace(x_domain[0], x_domain[1], N_ic)[:, None]
    t_ic = np.zeros((N_ic, 1))
    X_ic = np.hstack((x_ic, t_ic))
    u_ic = -np.sin(np.pi * x_ic)
    t_bc = np.linspace(t_domain[0], t_domain[1], N_bc)[:, None]
    x_bc_left = np.full((N_bc, 1), x_domain[0])
    x_bc_right = np.full((N_bc, 1), x_domain[1])
    X_bc = np.vstack((np.hstack((x_bc_left, t_bc)), np.hstack((x_bc_right, t_bc))))
    u_bc = np.zeros((2 * N_bc, 1))
    X_f = lb + (ub - lb) * lhs(2, N_f)
    X_ic_T = torch.from_numpy(X_ic).float().to(device)
    u_ic_T = torch.from_numpy(u_ic).float().to(device)
    X_bc_T = torch.from_numpy(X_bc).float().to(device)
    U_bc_T = torch.from_numpy(u_bc).float().to(device)
    X_f_T = torch.from_numpy(X_f).float().to(device)
    # 准备测试/验证数据
    data = scipy.io.loadmat('./data/Burgers.mat')
    x = data['x'].flatten()[:, None]
    t = data['t'].flatten()[:, None]
    usol = np.real(data['usol'])
    X, T = np.meshgrid(x, t)
    X_true = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    U_true = usol.T.flatten()[:, None]
    X_true_T = torch.from_numpy(X_true).float().to(device)
    U_true_T = torch.from_numpy(U_true).float().to(device)
    # 初始化并训练模型
    pinn = FCN(layers, lambda1, nu)
    pinn.train()
    # 测试和可视化
    error_vec, u_pred = pinn.test()
    print("最终相对误差 (L2 norm): %.5f" % error_vec)
    solutionplot(u_pred, X_f, usol, x, t)
    plot3D(x, t, usol)
    plot3D(x, t, u_pred)