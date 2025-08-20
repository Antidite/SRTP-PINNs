import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

def plot3D_Matrix(x,t,y):
    X,T= x,t
    F_xt = y
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(T,X, F_xt,20,cmap="rainbow")
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(),cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
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
    plt.show()