
import sys, os
sys.path.append(os.getcwd())
from sir.agent_seir import *

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint, rand, choice

def count_seir(pop):
    E = sum(p.is_exposed() for p in pop)
    I = sum(p.is_infected() for p in pop)
    R = sum(p.is_recovered() for p in pop)
    S = len(pop) - E - I - R
    return np.array([S, E, I, R])

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

def run_sim(a, b, k, N, T):
    """
    return the number of people in each group from time 0 to time T
    """
    pop = [Person() for i in range(N)] # our population
    infected = choice(range(N), size=int(N*0.001), replace=False) # 0.1% infected at T=0
    for i in infected:
        pop[i].infect()
    counts_seir = count_seir(pop)

    for t in range(T):
        infected = [i for i in range(N) if pop[i].is_infected()]
        for i in infected:
            # each infectious person contact b people, including S,I,R
            contacts = choice(list(set(range(N))-set([i])), size=b, replace=False)
            for j in contacts:
                pop[j].contact() # if S, then get infected
            # k fraction of the infectious population recover
            if rand() < k:
                pop[i].recover()

        exposed = [i for i in range(N) if pop[i].is_exposed()]
        for i in exposed:
            if rand() < a:
                pop[i].infect()

        # append SEIR counts at time t
        counts_seir = np.vstack((counts_seir, count_seir(pop)))
    return counts_seir


# plot how s, i, and r change over the length of the simulation
#for each single case
# sim1 = run_sim(8,0.01,1000,100) # shape = (T+1, 3), columns are S, I, R
# plt.plot(sim1[:,0],label = "Susceptible")
# plt.plot(sim1[:,1],label = "Infected")
# plt.plot(sim1[:,2],label = "Recovered")
# plt.legend()
# plt.show()

# simulation for bs and ks
T = 100
N = 1000
#for facet plot
f_bs = np.arange(1, 11, 2, dtype=np.int64)
f_ks = np.logspace(-2, 0, 5)
f_results = np.zeros((len(f_bs), len(f_ks), T+1, 3)) # empty container

for i, b in enumerate(f_bs):
    for j, k in enumerate(f_ks):
        counts_seir = run_sim(b,k,N,T) # shape = (T+1, 3), columns are S, I, R
        f_results[i,j,...] = counts_seir / N # convert to fractions, store in results

# facet plot
f, ax = plt.subplots(len(f_bs), len(f_ks), figsize=(20, 16), sharex=True, sharey=True)
for i, b in enumerate(f_bs):
     for j, k in enumerate(f_ks):
        plot_sim(f_results[i,j].T, b, k, ax[i,j], accessory=False)

for i, b in enumerate(f_bs):
    ax[i, 0].set_ylabel("b = {}".format(b))
for j, k in enumerate(f_ks):
    ax[0, j].set_title("k = {:.2f}".format(k))
ax[0, -1].legend()
f.text(0.5, 0.95, 'Facet Diagram for Different Parameter Values', ha='center', fontsize=18)
f.text(0.5, 0.08, 'Time', ha='center', fontsize=14)
f.text(0.08, 0.5, 'Fraction of People', va='center', rotation='vertical', fontsize=14)
# plt.savefig("output/agent_facet_plot.png")

# phase diagram
p_bs = np.arange(1, 11, dtype=np.int64)
p_ks = np.logspace(-2, 0, 10)
p_results = np.zeros((len(p_bs), len(p_ks), T+1, 3)) # empty container

for i, b in enumerate(p_bs):
    for j, k in enumerate(p_ks):
        counts_seir = run_sim(b,k,N,T) # shape = (T+1, 3), columns are S, I, R
        p_results[i,j,...] = counts_seir / N
# phase diagram at time t
#t = 10
f, ax = plt.subplots(1, 3, figsize=(15,5))
for i, t in enumerate([5, 10, 50]):
    m = ax[i].imshow(p_results[::-1,:,t,1],
                     extent=[np.min(p_ks), np.max(p_ks), np.min(p_bs), np.max(p_bs)],
                     vmin=0, vmax=1)
    ax[i].axis('auto')
    ax[i].set_title("t = {}".format(t))
f.colorbar(m, ax=[ax[0],ax[1],ax[2]])
f.text(0.5, 0.95, 'Phase Diagram of Infection Rate at Different Times', ha='center', fontsize=14)
f.text(0.5, 0.01, 'k: recover fraction', ha='center', fontsize=12)
f.text(0.08, 0.5, 'b: number of interactions', va='center', rotation='vertical', fontsize=12)
# plt.savefig("output/agent_phase_plot.png", dpi=200)
