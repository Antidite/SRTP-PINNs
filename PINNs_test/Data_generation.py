import torch
import numpy
import torch.autograd
from torch import Tensor, meshgrid
import time
import torch.nn as nn
import torch.optim as optim
from pyDOE import lhs
import scipy.io

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split

def plot3D(x,t,y):
    x_plot = x
    t_plot = t
    X,T = meshgrid(x_plot,t_plot)
    F_xt = y
    fig,ax = plt.subplots(1,1)
    cp = ax.contourf(T,X, F_xt,20,cmap = "rainbow")
    fig.colorbar(cp)
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()
    ax = plt.axes(projection = '3d')
    ax.plot_surface(T.numpy(),X.numpy(),F_xt.numpy(),cmap = "rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.show()

def plot3D_Matrix(x,t,y):
    X,T = x,t
    F_xt = y
    fig,ax = plt.subplots(1,1)
    cp = ax.contourf(T,X, F_xt,20,cmap = "rainbow")
    fig.colorbar(cp)
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.show()

def f_real(x,t):
    return torch.exp(-t)*(torch.sin(numpy.pi*x))

x = torch.linspace(start = -1,end = 1,steps = 200)
t = torch.linspace(start = 0,end = 1,steps = 100)
X,T = torch.meshgrid(x,t)
y_real = f_real(X,T)
plot3D(x,t,y_real)
x_test = torch.hstack(X.transpose(1,0))