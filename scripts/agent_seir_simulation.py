import sys
sys.path.append('../sir')
from agent_seir import PopulationSEIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import itertools as it
import matplotlib.ticker as mticker

# global hyper parameters
N = 10000
T = 100
e0 = i0 = 0.001
b, k = 1, 0.01


# check how s,e,i,r change with f
fs = np.logspace(-2,0,10)
result = [] # (fs, T, seir)
for f in fs:
    pop = PopulationSEIR(N, e0, i0)
    seirs = pop.simulation(T=T, b=b, f=f, k=k)
    result.append(seirs)

result = np.asarray(result)
result.shape

def plot_lines(ax, df, cmap, start=0.2, end=1):
    """
    plot simulated result with various value of f
    input
        df: shape (len(fs),T), every row is, e.g. i(t), for certain f
        cmap: an plt.cm colormap object, see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        start, end: scale location in color map
    """
    cols = [cmap(x) for x in np.linspace(start, end, len(df))]
    for i, col in enumerate(cols):
        ax.plot(df[i,:], c=col)
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Fraction')
    return ax

    # # add colorbar
    # divider = make_axes_locatable(plt.gca())
    # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    # cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
    # plt.gcf().add_axes(ax_cb)

fig, ax = plt.subplots(2,2, sharex='all', sharey='all')
ax[0,0] = plot_lines(ax[0,0], result[:,:,0], mpl.cm.Blues)
ax[0,1] = plot_lines(ax[0,1], result[:,:,1], mpl.cm.Purples)
ax[1,0] = plot_lines(ax[1,0], result[:,:,2], mpl.cm.OrRd)
ax[1,1] = plot_lines(ax[1,1], result[:,:,3], mpl.cm.Greens)
fig.text(0.5, 0.04, 'Time', va='center', ha='center')
fig.text(0.04, 0.5, 'Fraction', va='center', ha='center', rotation='vertical')
plt.savefig('../output/agent_seir_by_f.png', dpi=300)


# 3-D Phase diagram in PDF

def simulate(bs, ks, fs, T=T, N=N, e0=e0, i0=i0):
    """
    return
        bfks: (bs*ks*fs, 3)
        itss: (bs*ks*fs, T), record of i(t) for every (b,f,k) setting
    """
    bfks0 = []
    itss0 = []

    for b,f,k in it.product(bs, fs, ks):
        pop = PopulationSEIR(N, e0, i0)
        seirs = pop.simulation(T=T, b=b, f=f, k=k)
        its = seirs[:, 2] # i(t)
        bfks0.append([b,f,k])
        itss0.append(its)

    bfks = np.asarray(bfks0).reshape(len(itss0), 3)
    itss = np.asarray(itss0) # nsim * (T+1)
    return bfks, itss

def plot_3d(bfks, its, cmap='OrRd', title=None, save_as=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter3D(bfks[:,0], np.log(bfks[:,1]), np.log(bfks[:,2]),
                        c=its, cmap=cmap, vmin=0, vmax=1, alpha=0.5, linewidths=0)

    ax.set_xlabel('b')
    ax.set_ylabel('f')
    ax.set_zlabel('k')
    def log_tick_formatter(val, pos=None):
        return "{:.2f}".format(np.exp(val))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    fig.colorbar(scat, pad=0.2, shrink=0.8)
    if title:
        plt.title(title)
    if save_as:
        plt.savefig(f'../output/agent_seir_{save_as}.png', dpi=300)

bs = range(1,7)
ks = np.logspace(-2,-0.5,7)
fs = np.logspace(-2,0,7)
T = 100
bfks, itss = simulate(bs, ks, fs, T=100)

plot_3d(bfks, itss[:,50], title=f't = {50}', save_as='bfk_it')


for t in range(T):
    plot_3d(bfks, itss[:,t], title=f't = {t}')
    plt.show()
