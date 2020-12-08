import numpy as np
import math
from numpy.random import randint, rand, randn, choice
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
sys.path.append('../')
from scipy.spatial import KDTree
from sir.agent_pop2d import Population2d
from matplotlib.lines import Line2D


def plot_lines(ax, df, cmap,label = None, start=0.2, end=1):
    """
    plot simulated result with various value of f
    input
        df: shape (len(fs),T), every row is, e.g. i(t), for certain f
        cmap: an plt.cm colormap object, see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        start, end: scale location in color map
    """
    cols = [cmap(x) for x in np.linspace(start, end, len(df))]
    for i, col in enumerate(cols):
        if label != None:
            ax.plot(df[i, :], c=col, label=label[i])
        else:
            ax.plot(df[i, :], c=col)
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Fraction')
    return ax


# parameter settings
N=1000
I = 1
T = 50
b = 7
k = 0.1
q = math.sqrt(b/(math.pi*N))

# investigate different ps starting at center
def simulate_step(ps):
    results = np.zeros((len(ps),T+1,3))
    for i, p in enumerate(ps):
        pop = pop = Population2d(N,I)
        counts_sir = pop.simulation(T, p, q=q, k=0.1) # shape = (T+1,3), columns are S, I, R
        results[i,...] = counts_sir / N # convert to fractions, store in results
    return results
#generate scale plot
def plot_scale_effect(results):
    fig, ax = plt.subplots(1,3,figsize = (12,4),sharex='all', sharey='all')
    ax[0] = plot_lines(ax[0], results[:,:,0], ps, mpl.cm.Blues) #S
    ax[1] = plot_lines(ax[1], results[:,:,1], ps, mpl.cm.OrRd) #I
    ax[2] = plot_lines(ax[2], results[:,:,2], ps, mpl.cm.Greens) #R
    plt.savefig('../output/agent_2d_by_p.png', dpi=300)

ps = np.round(np.linspace(0,1,20),1)
r_ps = simulate_step(ps)
plot_scale_effect(r_ps)

# investigate location
# loc = ['center','corner']
# loc.extend(['random']*5)
# p = 0.01
# results_loc = np.zeros((len(loc),T+1,3))
# for i, l in enumerate(loc):
#     pop = Population2d(N,I,l)
#     counts_sir = pop.simulation(T,p=p, q=q,k=0.1) # shape = (T+1,3), columns are S, I, R
#     results_loc[i,...] = counts_sir / N # convert to fractions, store in results
# #location plot
# fig, ax = plt.subplots(1,1, sharex='all', sharey='all')
# ax = plot_lines(ax, results_loc[:,:,1],loc,mpl.cm.tab20)
# ax.legend(loc = 4)
# fig.show()

def simulate_pos(p,n1 = 5,n2 = 5,n3 = 20):
    results = np.zeros(((n1 + n2 + n3), T + 1, 3))
    def run_sim(start, end, loc):
        for i in range(start, end):
            pop = Population2d(N, I, loc)
            counts_sir = pop.simulation(T, p=p, q=q, k=0.1)  # shape = (T+1,3), columns are S, I, R
            results[i, ...] = counts_sir / N
    run_sim(0, n1, 'center')
    run_sim(n1, n1 + n2, 'corner')
    run_sim(n1 + n2, n1 + n2 + n3, 'random')
    return results

def plot_loc_effect_2d(results,n1 = 5,n2 = 5,n3 = 20,save_as=None):
    """
    investigate center/corner/random starting point with
    moving step p
    ns: num of simulation
    """
    custom_lines = [Line2D([0], [0], color='r', lw=4),
                    Line2D([0], [0], color='g', lw=4),
                    Line2D([0], [0], color='b', lw=4)]
    plt.plot(results[:n1, :, 1].T, c='r')
    plt.plot(results[n1:n1 + n2, :, 1].T, c='g')
    plt.plot(results[n1 + n2:n1 + n2 + n3, :, 1].T, c='b')
    plt.legend(custom_lines, ['Center', 'Corner', 'Random'])
    plt.xlabel("Time")
    plt.ylabel("Infectious Rate")
    if save_as != None:
        plt.savefig(f'../output/agent_loc_p_{save_as}.png', dpi=300)
    plt.show()

r_p0 = simulate_pos(0)
plot_loc_effect_2d(r_p0,save_as=0)
r_p05 = simulate_pos(0.5)
plot_loc_effect_2d(r_p05,save_as=0.5)
r_p1 = simulate_pos(1)
plot_loc_effect_2d(r_p1,save_as=1)
