"""
Because we're investigating how a new disease spreads, you can start with
a small number of infectious individuals (e.g. 0.1% of the population)
for both the continuous and discrete simulations, with the remainder of
the population in the susceptible category.

# dennis: let's use N = 1000 and initial infectious fraction = 0.1%

Your simulations should produces some plots of how s, i, and r change over
the length of the simulation for a couple different choices of k and b.

# dennis: include title, axis labels, legend. can try some extreme values of k and b

You should also investigate the qualitative behavior of the simulations based on
the parameters b and k in a phase diagram. For instance, how does the
total percentage of the population infected at some point in the simulation
depend on these parameters? Are there parameter regimes where i quickly goes
to 0? Are there parameter regimes where everyone is eventually infected?

# dennis: see https://caam37830.github.io/book/09_computing/agent_based_models.html

"""

import sys
sys.path.append('../')
from sir.ode import ode_model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.animation as animation


def plot_sim(result, b, k, ax, accessory=True):
    """
    plot the fraction of people in each group over time
    result = np.array([s, i, r])
    """
    ax.plot(result[0], label="Susceptible")
    ax.plot(result[1], label="Infectious")
    ax.plot(result[2], label="Removed")
    if accessory:
        ax.set_title("b={:.3f}, k={:.3f}".format(b, k))
        ax.set_xlabel("time")
        ax.set_ylabel("fraction of people")
        ax.legend()

def run_sim(i0, T, bs, ks):
    """
    return the fraction of people in each group from time 0 to time T
    """
    results = np.zeros((len(bs), len(ks), 3, int(T)))
    for i, b in enumerate(bs):
        for j, k in enumerate(ks):
            results[i,j,...] = np.array(ode_model(i0, T, b, k))
    return results

# an example
N = 10000; T = 100; i0 = 0.001
b = 1; k = 0.1
f, ax = plt.subplots()
plot_sim(ode_model(i0, T, b, k), b, k, ax)
# plt.show()



# facet plot

def facet_plot(results, bs, ks):
    f, ax = plt.subplots(len(bs), len(ks), figsize=(20, 16), sharex=True, sharey=True)
    for i, b in enumerate(bs):
        for j, k in enumerate(ks):
            plot_sim(results[i,j], b, k, ax[i,j], accessory=False)

    for i, b in enumerate(bs):
        ax[i,0].set_ylabel("b = {}".format(b))
    for j, k in enumerate(ks):
        ax[0,j].set_title("k = {:.2f}".format(k))
    ax[0,-1].legend()
    f.text(0.5, 0.95, 'Facet Diagram for Different Parameter Values', ha='center', fontsize=18)
    f.text(0.5, 0.08, 'Time', ha='center', fontsize=14)
    f.text(0.08, 0.5, 'Fraction of People', va='center', rotation='vertical', fontsize=14)
    # plt.savefig("output/ode_facet_plot.png")

bs = np.arange(1, 11, 2, dtype=np.int64)
ks = np.logspace(-2, 0, 5)
results_small = run_sim(i0, T, bs, ks)
facet_plot(results_small, bs, ks)

# phase diagram at time t1, t2 etc
def log_tick_formatter(val, pos=None):
    return "{:.2f}".format(np.exp(val))

def phase_plot_ts(results, bs, ks, ts):
    f, ax = plt.subplots(1, 3, figsize=(16,4))
    for i, t in enumerate(ts): # three time points
        m = ax[i].imshow(results[::-1,:,1,t],
                         extent=[np.min(np.log(ks)), np.max(np.log(ks)), np.min(bs), np.max(bs)],
                         vmin=0, vmax=1, cmap='OrRd')
        ax[i].axis('auto')
        ax[i].xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax[i].set_title(f"t = {t}")
    f.colorbar(m, ax=[ax[0],ax[1],ax[2]])
    f.text(0.5, 0.95, 'Phase Diagram of Infection Rate at Different Times', ha='center', fontsize=14)
    f.text(0.5, 0.01, 'k: recover fraction', ha='center', fontsize=12)
    f.text(0.08, 0.5, 'b: number of interactions', va='center', rotation='vertical', fontsize=12)
    # plt.savefig("output/ode_phase_plot.png", dpi=200)
    f.savefig('../docs/figs/ode_phase_plot.png', dpi=300)

# phase diagram gif
def update(t, bs, ks, ax, cmap='OrRd'):
    ax.clear()
    ax.imshow(results_large[::-1,:,1,t], aspect='auto',
                     extent=[np.min(np.log(ks)), np.max(np.log(ks)), np.min(bs), np.max(bs)],
                     vmin=0, vmax=1, cmap=cmap)
    ax.set_title(f"t = {t}")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.set_xlabel('k: recover fraction')
    ax.set_ylabel('b: number of interactions')
    return ax

def phase_plot_gif(results, Tmax=T, cmap='OrRd'):
    fig, ax = plt.subplots(dpi=200, figsize=(6,4))
    im = ax.imshow(results[::-1,:,1,0], aspect='auto', extent=[np.min(np.log(ks)), np.max(np.log(ks)), np.min(bs), np.max(bs)], vmin=0, vmax=1, cmap=cmap)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.set_xlabel('k: recover fraction')
    ax.set_ylabel('b: number of interactions')
    colbar = fig.colorbar(im, pad=0.2, shrink=0.8)
    colbar.set_label(r'      $i(t)$', rotation=0)
    ani = animation.FuncAnimation(fig, update, frames=Tmax, fargs=(bs, ks, ax, cmap),
                                    interval=100, blit=False)
    ani.save("../docs/figs/ode_phase_plot.gif", writer='pillow')

bs = np.arange(1, 21, dtype=np.int64)
ks = np.logspace(-2, 0, 20)
results_large = run_sim(i0, T, bs, ks)
ts = [5, 10, 50]
phase_plot_ts(results_large, bs, ks, ts)
phase_plot_gif(results_large, Tmax=100)
