"""
Plots samples or new shapes in the semantic space.

Author(s): Wei Chen (wchen459@gmail.com)
"""

import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('..')
from utils import gen_grid


def plot_topology(values, ax):
    ax.contourf(values>0, 15)
    # ax.contour(values, levels=[0.0], linewidths=3, alpha=0.5)
    ax.axis('equal')
    ax.axis('off')
    

def plot_grid(points_per_axis, gen_func, d=2, bounds=(0.0, 1.0), fname=None):
    
    ''' Uniformly plots synthesized shapes in the first two dimensions of the latent space '''
    
    Z = np.zeros((points_per_axis**2, d))
    Z[:, :2] = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid
    X = gen_func(Z)
    has_x0 = (type(X).__name__ == 'tuple')
    if has_x0:
        X, z0, x0 = X
    X = X.reshape(points_per_axis, points_per_axis, X.shape[1], X.shape[2])
    fig, axes = plt.subplots(ncols=points_per_axis, nrows=points_per_axis+has_x0, 
                             figsize=(points_per_axis*2, (points_per_axis+has_x0)*2), 
                             constrained_layout=True)
    if has_x0:
        plot_topology(x0, axes[0, 0])
        axes[0, 0].set_title('({:.2f},{:.2f})'.format(z0[0], z0[1]))
        for j in range(1, points_per_axis):
            axes[0, j].axis('off')
    for i in range(points_per_axis):
        for j in range(points_per_axis):
            plot_topology(X[i, j], axes[i+has_x0, j])
            axes[i+has_x0, j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    # plt.tight_layout()
    plt.savefig(fname+'.png', dpi=300)
    plt.close()
    

def plot_random(points_per_axis, gen_func, d=2, bounds=(0.0, 1.0), fname=None):
    
    ''' Plots synthesized shapes uniformly at random in the latent space '''
    
    Z = np.random.uniform(bounds[0], bounds[1], (points_per_axis**2, d))
    X = gen_func(Z)
    has_x0 = (type(X).__name__ == 'tuple')
    if has_x0:
        X, z0, x0 = X
    X = X.reshape(points_per_axis, points_per_axis, X.shape[1], X.shape[2])
    fig, axes = plt.subplots(ncols=points_per_axis, nrows=points_per_axis+has_x0, 
                             figsize=(points_per_axis*2, (points_per_axis+has_x0)*2), 
                             constrained_layout=True)
    if has_x0:
        plot_topology(x0, axes[0, 0])
        axes[0, 0].set_title('({:.2f},{:.2f})'.format(z0[0], z0[1]))
        for j in range(1, points_per_axis):
            axes[0, j].axis('off')
    for i in range(points_per_axis):
        for j in range(points_per_axis):
            plot_topology(X[i, j], axes[i+has_x0, j])
            axes[i+has_x0, j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    # plt.tight_layout()
    plt.savefig(fname+'.png', dpi=300)
    plt.close()
        

