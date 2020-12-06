

import sys, os
sys.path.append(os.getcwd())
from sir.ode_seir import *

import numpy as np
import matplotlib.pyplot as plt

def plot_sim(result, a, b, k, ax, accessory=True):
    """
    plot the fraction of people in each group over time
    result = np.array([s, e, i, r])
    """
    ax.plot(result[0], label="Susceptible")
    ax.plot(result[1], label="Exposed")
    ax.plot(result[2], label="Infectious")
    ax.plot(result[3], label="Removed")
    if accessory:
        ax.set_title("a={:.3f}, b={:.3f}, k={:.3f}".format(a, b, k))
        ax.set_xlabel("time")
        ax.set_ylabel("fraction of people")
        ax.legend()

def run_sim(i0, T, as, bs, ks):
    """
    return the fraction of people in each group from time 0 to time T
    """
    results = np.zeros((len(as), len(bs), len(ks), 3, int(T)))
    for l, a in enumerate(as):
        for i, b in enumerate(bs):
            for j, k in enumerate(ks):
                results[l, i,j,...] = np.array(ode_model(i0, T, a, b, k))
    return results

# an example
N = 10000; T = 100; i0 = 0.001
b = 1; k = 0.1
f, ax = plt.subplots()
plot_sim(ode_model(i0, T, a, b, k), a, b, k, ax)
# plt.show()

# facet plot
bs = np.arange(1, 11, 2, dtype=np.int64)
ks = np.logspace(-2, 0, 5)
results_small = run_sim(i0, T, bs, ks)

f, ax = plt.subplots(len(bs), len(ks), figsize=(20, 16), sharex=True, sharey=True)
for i, b in enumerate(bs):
    for j, k in enumerate(ks):
        plot_sim(results_small[i,j], b, k, ax[i,j], accessory=False)

for i, b in enumerate(bs):
    ax[i,0].set_ylabel("b = {}".format(b))
for j, k in enumerate(ks):
    ax[0,j].set_title("k = {:.2f}".format(k))
ax[0,-1].legend()
f.text(0.5, 0.95, 'Facet Diagram for Different Parameter Values', ha='center', fontsize=18)
f.text(0.5, 0.08, 'Time', ha='center', fontsize=14)
f.text(0.08, 0.5, 'Fraction of People', va='center', rotation='vertical', fontsize=14)
# plt.savefig("output/ode_facet_plot.png")

# phase diagram at time t
bs = np.arange(1, 11, dtype=np.int64)
ks = np.logspace(-2, 0, 10)
results_large = run_sim(i0, T, bs, ks)

t = 10
f, ax = plt.subplots(1, 3, figsize=(15,5))
for i, t in enumerate([5, 10, 50]):
    m = ax[i].imshow(results_large[::-1,:,1,t],
                     extent=[np.min(ks), np.max(ks), np.min(bs), np.max(bs)],
                     vmin=0, vmax=1)
    ax[i].axis('auto')
    ax[i].set_title(f"t = {t}")
f.colorbar(m, ax=[ax[0],ax[1],ax[2]])
f.text(0.5, 0.95, 'Phase Diagram of Infection Rate at Different Times', ha='center', fontsize=14)
f.text(0.5, 0.01, 'k: recover fraction', ha='center', fontsize=12)
f.text(0.08, 0.5, 'b: number of interactions', va='center', rotation='vertical', fontsize=12)
# plt.savefig("output/ode_phase_plot.png", dpi=200)
