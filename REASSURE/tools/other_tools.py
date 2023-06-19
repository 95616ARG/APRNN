import numpy as np, torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def mesh_grid(x_range, y_range, num):
    x, y = torch.linspace(x_range[0], x_range[1], num), torch.linspace(y_range[0], y_range[1], num)
    grid_x, grid_y = torch.meshgrid(x, y)
    return torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], dim=-1)


def Plot_3D(X, Y, Z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.show()


