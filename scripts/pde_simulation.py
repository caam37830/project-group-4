import sys
sys.path.append('../sir')
from pde import *

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# parameters
M = 200; i0 = 0.001; T = 100; b = 5; k = 0.05

# interesting p, starting from center
pos = (M // 2, M // 2)
ps = np.logspace(-4, -1, 10) / M

res = []
for p in ps:
    res.append(pde_model(i0, pos, M, T, b, k, p))
res = np.asarray(res).mean(axis=(2,3))

def plot_lines(ax, df, title, cmap=mpl.cm.OrRd, start=0.2, end=1):
    """
    plot simulated result with various value of f
    input
        df: shape (len(fs),T), every row is, e.g. i(t), for certain f
        cmap: a matplot.cm colormap object, see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        start, end: scale location in color map
    """
    cols = [cmap(x) for x in np.linspace(start, end, len(df))]
    for i, col in enumerate(cols):
        ax.plot(df[i,:], c=col)
    ax.set_title(title)
    return ax

f, ax = plt.subplots(1, 3, figsize=(12,4), sharex='all', sharey='all')
ax[0] = plot_lines(ax[0], res[:,0,:], "s(t)", mpl.cm.Blues)
ax[1] = plot_lines(ax[1], res[:,1,:], "i(t)", mpl.cm.OrRd)
ax[2] = plot_lines(ax[2], res[:,2,:], "r(t)", mpl.cm.Greens)
f.text(0.5, 0.04, 'Time', va='center', ha='center')
f.text(0.04, 0.5, 'Fraction', va='center', ha='center', rotation='vertical')
# plt.savefig("../docs/figs/pde_2d_by_p.png", dpi=300)


# timestamp plots

def plot_time(s, i, r, title):
    
    def plot_sim(x, ax, color, title=False):
        ts = [10,20,30,40,50]
        for i, t in enumerate(ts):
            ax[i].imshow(x[:,:,t], cmap=color, origin="lower")
            if title:
                ax[i].set_title(f"t = {t}")
    
    f, ax = plt.subplots(3, 5, figsize=(15,9), sharex=True, sharey=True)
    plot_sim(s, ax[0,:], color=mpl.cm.Blues, title=True)
    plot_sim(i, ax[1,:], color=mpl.cm.OrRd)
    plot_sim(r, ax[2,:], color=mpl.cm.Greens)
    ax[0,0].set_ylabel("s(t)", fontsize=14)
    ax[1,0].set_ylabel("i(t)", fontsize=14)
    ax[2,0].set_ylabel("r(t)", fontsize=14)
    f.text(0.5, 0.94, "Change of s(t), i(t), r(t). Starting from " + title,
           va='center', ha='center', fontsize=16)

p = 0.001 / M

# center
pos = (M // 2, M // 2)
s, i, r = pde_model(i0, pos, M, T, b, k, p)
plot_time(s, i, r, "center")
# plt.savefig("../docs/figs/pde_2d_time_center.png", dpi=500)

# corner
pos = (0, 0)
s, i, r = pde_model(i0, pos, M, T, b, k, p)
plot_time(s, i, r, "corner")
# plt.savefig("../docs/figs/pde_2d_time_corner.png", dpi=500)

# random
pos = (np.random.randint(20, M-20), np.random.randint(20, M-20))
s, i, r = pde_model(i0, pos, M, T, b, k, p)
plot_time(s, i, r, "random place")
# plt.savefig("../docs/figs/pde_2d_time_random.png", dpi=500)




