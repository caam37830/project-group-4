import sys
sys.path.append('../sir')
from pde import *

import numpy as np
import matplotlib.pyplot as plt

# parameters
M = 200; i0 = 0.0001; T = 100; b = 7; k = 0.1

# interesting p
pos = (M // 2, M // 2)
ps = np.logspace(-4, -0.5, 10) / M

res = []
for p in ps:
    res.append(pde_model(i0, pos, M, T, b, k, p))
res = np.asarray(res).mean(axis=(2,3))


def plot_lines(ax, df, cmap=mpl.cm.OrRd, start=0.2, end=1):
    cols = [cmap(x) for x in np.linspace(start, end, len(df))]
    for i, col in enumerate(cols):
        ax.plot(df[i,:], c=col)
    return ax


f, ax = plt.subplots(1, 3, sharex='all', sharey='all')
ax[0] = plot_lines(ax[0], res[:,0,:], mpl.cm.Blues)
ax[1] = plot_lines(ax[1], res[:,1,:], mpl.cm.OrRd)
ax[2] = plot_lines(ax[2], res[:,2,:], mpl.cm.Greens)
fig.text(0.5, 0.04, 'Time', va='center', ha='center')
fig.text(0.04, 0.5, 'Fraction', va='center', ha='center', rotation='vertical')
plt.savefig('../output/agent_seir_by_f.png', dpi=300)
