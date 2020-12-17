import sys
sys.path.append('../sir')
from pde import *

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# parameters
M = 200; i0 = 0.001; T = 100; b = 5; k = 0.05

# interesting p, starting from center
pos = (M // 2, M // 2)
ps = np.logspace(-4, -1, 10) / M

res = []
for p in ps:
    res.append(pde_model(i0, pos, M, T, b, k, p))
res = np.asarray(res).mean(axis=(2,3))
res.shape
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

# gif

def plot_spread_gif(s, i, r, loc, Tmax=T, save_as=None):
    fig, ax = plt.subplots(1, 4, figsize=(15,3), dpi=300)
    # plt.subplots_adjust(hspace = -0.1)
    # fig.suptitle(f'Spread from {loc}')
    plt.text(0.5, 1.3, ' '*50+f'Spread from {loc}', horizontalalignment='center',
    fontsize=14, verticalalignment='bottom', transform=ax[1].transAxes)
    plt.subplots_adjust(bottom=0.3)

    ax[0].set_title(r"$s(x,y,t)$")
    ax[1].set_title(r"$i(x,y,t)$")
    ax[2].set_title(r"$r(x,y,t)$")
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    ax[2].xaxis.set_visible(False)
    ax[2].yaxis.set_visible(False)
    ax[3].set_xlim(0,Tmax)
    ax[3].set_xlabel('time')
    ax[3].set_ylim(0,1)
    ax[3].yaxis.tick_right()
    ax[3].set_title('means over time')

    ims = []
    for t in range(Tmax+1):

        im0 = ax[0].imshow(s[:,:,t], cmap=mpl.cm.Blues, vmin=0, vmax=1, origin="lower")
        im1 = ax[1].imshow(i[:,:,t], cmap=mpl.cm.OrRd, vmin=0, vmax=1, origin="lower")
        im2 = ax[2].imshow(r[:,:,t], cmap=mpl.cm.Greens, vmin=0, vmax=1, origin="lower")

        # ax[3].clear()
        im3s, = ax[3].plot(s[:,:,:t+1].mean(axis=(0,1)), label='s(t)', c='C0')
        im3i, = ax[3].plot(i[:,:,:t+1].mean(axis=(0,1)), label='i(t)', c='C1')
        im3r, = ax[3].plot(r[:,:,:t+1].mean(axis=(0,1)), label='r(t)', c='C2')


        ims.append([im0, im1, im2, im3s, im3i, im3r]) # add subplots to a list

    custom_lines = [Line2D([0], [0], color='C0'),
                    Line2D([0], [0], color='C1'),
                    Line2D([0], [0], color='C2')]
    lgd = ax[3].legend(custom_lines, [r'$s(t)$', r'$i(t)$', r'$r(t)$'],
                        loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3)
    # cbaxes0 = fig.add_axes([0.152, 0.1, 0.115, 0.05])
    # cbaxes1 = fig.add_axes([0.354, 0.1, 0.115, 0.05])
    # cbaxes2 = fig.add_axes([0.556, 0.1, 0.115, 0.05])
    cbaxes0 = fig.add_axes([0.152, 0.1, 0.115, 0.05])
    cbaxes1 = fig.add_axes([0.354, 0.1, 0.115, 0.05])
    cbaxes2 = fig.add_axes([0.556, 0.1, 0.115, 0.05])
    fig.colorbar(im0, ax=ax[0], orientation="horizontal", shrink=0.82, cax=cbaxes0)
    fig.colorbar(im1, ax=ax[1], orientation="horizontal", shrink=0.82, cax=cbaxes1)
    fig.colorbar(im2, ax=ax[2], orientation="horizontal", shrink=0.82, cax=cbaxes2)

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False)
    if save_as:
        ani.save(f"../docs/figs/pde_2d_spread_{save_as}.gif", writer='pillow', dpi=300)

p = 0.001 / M

# center
pos = (M // 2, M // 2)
s, i, r = pde_model(i0, pos, M, T, b, k, p)
plot_time(s, i, r, "center")
# plt.savefig("../docs/figs/pde_2d_time_center.png", dpi=500)
plot_spread_gif(s,i,r,'center', save_as='center')


# corner
pos = (0, 0)
s, i, r = pde_model(i0, pos, M, T, b, k, p)
plot_time(s, i, r, "corner")
# plt.savefig("../docs/figs/pde_2d_time_corner.png", dpi=500)
plot_spread_gif(s,i,r,'corner', save_as='corner')


# random
pos = (np.random.randint(20, M-20), np.random.randint(20, M-20))
s, i, r = pde_model(i0, pos, M, T, b, k, p)
plot_time(s, i, r, "random place")
# plt.savefig("../docs/figs/pde_2d_time_random.png", dpi=500)
plot_spread_gif(s,i,r,'random', save_as='random')




 # 1 + 1 + n
res = []
p = 0.005 / M
pos = (M // 2, M // 2)
res.append(pde_model(i0, pos, M, T, b, k, p))

pos = (0, 0)
res.append(pde_model(i0, pos, M, T, b, k, p))

for _ in range(10):
    pos = (np.random.randint(0, M), np.random.randint(0, M))
    res.append(pde_model(i0, pos, M, T, b, k, p))

res = np.asarray(res).mean(axis=(2,3))
cols = ['pink','lime','cyan']
plt.plot(res[0,1,:], color=cols[0], label="center")
plt.plot(res[1,1,:], color=cols[1], label="corner", alpha=0.5)
plt.plot(res[2,1,:], color=cols[2], label="random", alpha=0.5)
for i in range(3, 12):
    plt.plot(res[i,1,:], color=cols[2], alpha=0.5)
plt.legend()
plt.savefig("../docs/figs/pde_loc2.png", dpi=300)
