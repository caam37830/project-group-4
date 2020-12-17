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
import math
from tqdm import tqdm
import matplotlib.animation as animation

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
        if label is None:
            ax.plot(df[i, :], c=col)
        else:
            ax.plot(df[i, :], c=col, label=label[i])

    # ax.set_xlabel('Time')
    # ax.set_ylabel('Fraction')
    return ax


# parameter settings
N = 1000
I = 1
T = 50
b = 7
k = 0.05
q = math.sqrt(b/(math.pi*N))

# investigate different ps starting at center
def simulate_step(ps):
    results = np.zeros((len(ps),T+1,3))
    for i, p in tqdm(enumerate(ps)):
        pop = pop = Population2d(N,I)
        counts_sir = pop.simulation(T, p, q=q, k=0.1) # shape = (T+1,3), columns are S, I, R
        results[i,...] = counts_sir / N # convert to fractions, store in results
    return results

# generate sir line plot for different ps
def plot_step_effect(results,save_as=None):
    fig, ax = plt.subplots(1,3,figsize = (12,4),sharex='all', sharey='all')
    ax[0] = plot_lines(ax[0], results[:,:,0], mpl.cm.Blues,ps) #S
    ax[1] = plot_lines(ax[1], results[:,:,1], mpl.cm.OrRd,ps) #I
    ax[2] = plot_lines(ax[2], results[:,:,2], mpl.cm.Greens,ps) #R
    if save_as != None:
        plt.savefig(f'../docs/figs/{save_as}.png', dpi=300)
    plt.show()


ps = np.linspace(0,1,10)
r_ps = simulate_step(ps)
plot_step_effect(r_ps,'agent_2d_by_p')


cols = ['pink','lime','cyan']
def simulate_pos(p, n1 = 5, n2 = 5, n3 = 20):
    results = np.zeros(((n1 + n2 + n3), T + 1, 3))
    def run_sim(start, end, loc):
        print(f'simulating {loc}')
        for i in tqdm(range(start, end)):
            pop = Population2d(N, I, loc)
            counts_sir = pop.simulation(T, p=p, q=q, k=0.1)  # shape = (T+1,3), columns are S, I, R
            results[i, ...] = counts_sir / N
    run_sim(0, n1, 'center')
    run_sim(n1, n1 + n2, 'corner')
    run_sim(n1 + n2, n1 + n2 + n3, 'random')
    return results

def plot_loc_effect_2d(results, p, n1 = 5,n2 = 5,n3 = 20, save_as=None):
    """
    investigate center/corner/random starting point with
    moving step p
    ns: num of simulation
    """
    custom_lines = [Line2D([0], [0], color=cols[0], lw=4),
                    Line2D([0], [0], color=cols[1], lw=4),
                    Line2D([0], [0], color=cols[2], lw=4)]
    plt.plot(results[:n1, :, 1].T, c=cols[0], alpha=1)
    plt.plot(results[n1:n1 + n2, :, 1].T, c=cols[1], alpha=0.5)
    plt.plot(results[n1 + n2:n1 + n2 + n3, :, 1].T, c=cols[2], alpha=0.5)
    plt.legend(custom_lines, ['center', 'corner', 'random'])
    plt.xlabel("Time")
    plt.ylabel("Infectious Rate")
    plt.title(f"Infectious Rate by Different Starting Position with p = {p}")
    if save_as != None:
        plt.savefig(f'../docs/figs/agent_loc_p_{save_as}.png', dpi=300)
    plt.show()

p = 0
r_p0 = simulate_pos(p)
plot_loc_effect_2d(r_p0, p, save_as=p)
p = 0.5
r_p05 = simulate_pos(p)
plot_loc_effect_2d(r_p05,p,save_as=p)
p = 1
r_p1 = simulate_pos(p)
plot_loc_effect_2d(r_p1,p,save_as=p)


# show spread in 2-d scatter plots . take snapshots at certain time points
T = [0,2,4,6,10]
loc = ['center','corner','random']


def plot_scatter(ax,pop,s=10):
    ax.scatter(pop.s_info[:, 1], pop.s_info[:, 2], s=s)
    ax.scatter(pop.i_info[:, 1], pop.i_info[:, 2], s=s)
    if pop.r_info.size > 0:
        ax.scatter(pop.r_info[:, 1], pop.r_info[:, 2], s=s)

def plot_spread(p,save_as=None):
    f, ax = plt.subplots(len(loc), len(T), figsize=(20, 12), sharex=True, sharey=True)
    for i, l in enumerate(loc):
        pop = Population2d(N,I,l)
        for j, t in enumerate(T):
            pop.simulation(t, p = p, q = q, k = k)
            plot_scatter(ax[i,j],pop)
    for i, l in enumerate(loc):
        ax[i, 0].set_ylabel("{}".format(l))
    for j, t in enumerate(T):
        ax[0, j].set_title("t = {}".format(T[j]))
    f.text(0.5, 0.92, f'Spread of Virus by time and position with p = {p}', ha='center', fontsize=18)
    if save_as != None:
        plt.savefig(f'../docs/figs/agent_spread_p_{save_as}.png', dpi=300)
    plt.show()
    plt.show()

plot_spread(0,0)
plot_spread(0.5,0.5)
plot_spread(1,1)


# show spread in 2-d scatter plots by gif
its = np.zeros((1,3))
def update(t, Tmax, p, ax, s, pop_center, pop_corner, pop_random):
    i_tmp = []

    SIRs = pop_center.simulation(T=1, p=p, q=q, k=k)
    i_tmp.append(SIRs[1,1]/N)
    ax[0].clear()
    ax[0].scatter(pop_center.s_info[:, 1], pop_center.s_info[:, 2], s=s)
    ax[0].scatter(pop_center.i_info[:, 1], pop_center.i_info[:, 2], s=s)
    if pop_center.r_info.size > 0:
        ax[0].scatter(pop_center.r_info[:, 1], pop_center.r_info[:, 2], s=s)
    ax[0].set_title('center')
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)

    SIRs = pop_corner.simulation(T=1, p=p, q=q, k=k)
    i_tmp.append(SIRs[1,1]/N)
    ax[1].clear()
    ax[1].scatter(pop_corner.s_info[:, 1], pop_corner.s_info[:, 2], s=s)
    ax[1].scatter(pop_corner.i_info[:, 1], pop_corner.i_info[:, 2], s=s)
    if pop_corner.r_info.size > 0:
        ax[1].scatter(pop_corner.r_info[:, 1], pop_corner.r_info[:, 2], s=s)
    ax[1].set_title('corner')
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)

    SIRs = pop_random.simulation(T=1, p=p, q=q, k=k)
    i_tmp.append(SIRs[1,1]/N)
    ax[2].clear()
    ax[2].scatter(pop_random.s_info[:, 1], pop_random.s_info[:, 2], s=s)
    ax[2].scatter(pop_random.i_info[:, 1], pop_random.i_info[:, 2], s=s)
    if pop_random.r_info.size > 0:
        ax[2].scatter(pop_random.r_info[:, 1], pop_random.r_info[:, 2], s=s)
    ax[2].set_title('random')
    ax[2].xaxis.set_visible(False)
    ax[2].yaxis.set_visible(False)

    global its
    its = np.vstack((its, np.asarray(i_tmp).reshape(1,3)))
    ax[3].clear()
    ax[3].plot(range(len(its)),its[:,0], label='center', c=cols[0])
    ax[3].plot(range(len(its)),its[:,1], label='corner', c=cols[1])
    ax[3].plot(range(len(its)),its[:,2], label='random', c=cols[2])
    ax[3].legend()
    ax[3].set_xlim(0,Tmax)
    ax[3].set_xlabel('time')
    ax[3].set_ylim(0,1)
    ax[3].set_title('infectious rate')
    ax[3].yaxis.tick_right()

def plot_spread_gif(p, Tmax=T, save_as=None):
    fig, ax = plt.subplots(1, 4, dpi=200, figsize=(16,4))
    s = 10

    global its

    pop_center = Population2d(N,I,'center')
    its[0,0] = pop_center.count_SIR()[1]/N
    ax[0].scatter(pop_center.s_info[:, 1], pop_center.s_info[:, 2], s=s)
    ax[0].scatter(pop_center.i_info[:, 1], pop_center.i_info[:, 2], s=s)
    if pop_center.r_info.size > 0:
        ax[0].scatter(pop_center.r_info[:, 1], pop_center.r_info[:, 2], s=s)
    ax[0].set_title('center')
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)

    pop_corner = Population2d(N,I,'corner')
    its[0,1] = pop_corner.count_SIR()[1]/N
    ax[1].scatter(pop_corner.s_info[:, 1], pop_corner.s_info[:, 2], s=s)
    ax[1].scatter(pop_corner.i_info[:, 1], pop_corner.i_info[:, 2], s=s)
    if pop_corner.r_info.size > 0:
        ax[1].scatter(pop_corner.r_info[:, 1], pop_corner.r_info[:, 2], s=s)
    ax[1].set_title('corner')
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)

    pop_random = Population2d(N,I,'random')
    its[0,2] = pop_random.count_SIR()[1]/N
    ax[2].scatter(pop_random.s_info[:, 1], pop_random.s_info[:, 2], s=s)
    ax[2].scatter(pop_random.i_info[:, 1], pop_random.i_info[:, 2], s=s)
    if pop_random.r_info.size > 0:
        ax[2].scatter(pop_random.r_info[:, 1], pop_random.r_info[:, 2], s=s)
    ax[2].set_title('random')
    ax[2].xaxis.set_visible(False)
    ax[2].yaxis.set_visible(False)

    ax[3].plot(range(len(its)),its[:,0], label='center', c=cols[0])
    ax[3].plot(range(len(its)),its[:,1], label='corner', c=cols[1])
    ax[3].plot(range(len(its)),its[:,2], label='random', c=cols[2])
    ax[3].legend()
    ax[3].set_xlim(0,Tmax)
    ax[3].set_xlabel('time')
    ax[3].set_ylim(0,1)
    ax[3].yaxis.tick_right()
    ax[3].set_title('infectious rate')

    ani = animation.FuncAnimation(fig, update, frames=Tmax,
                                fargs=(Tmax, p, ax, s, pop_center, pop_corner, pop_random),
                                    interval=100, blit=False)
    if save_as:
        ani.save(f"../docs/figs/{save_as}.gif", writer='pillow')


np.random.seed(0)
its = np.zeros((1,3))
plot_spread_gif(0,Tmax=100,save_as='agent_2d_spread_p0_isolated')

np.random.seed(1)
its = np.zeros((1,3))
plot_spread_gif(0,Tmax=100,save_as='agent_2d_spread_p0')

np.random.seed(0)
its = np.zeros((1,3))
plot_spread_gif(0.5,Tmax=100,save_as='agent_2d_spread_p05')

np.random.seed(0)
its = np.zeros((1,3))
plot_spread_gif(1,Tmax=100,save_as='agent_2d_spread_p1')
