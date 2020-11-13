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
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint, rand, choice
from sir import *
from scripts import ode_simulation as ode

def count_sir(pop):
    I = sum(p.is_infected() for p in pop)
    R = sum(p.is_recovered() for p in pop)
    S = len(pop) - I - R
    return np.array([S,I,R])

def run_sim(b, k, N, T):
    """
    return the number of people in each group from time 0 to time T
    """
    pop = [Person() for i in range(N)] # our population
    infected = choice(range(N), size=int(N*0.001), replace=False) # 0.1% infected at T=0
    for i in infected:
        pop[i].infect()
    counts_sir = count_sir(pop)

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

        # append SIR counts at time t
        counts_sir = np.vstack((counts_sir, count_sir(pop)))
    return counts_sir


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
# #for facet plot
# f_bs = np.arange(1, 11, 2, dtype=np.int64)
# f_ks = np.logspace(-2, 0, 5)
# f_results = np.zeros((len(f_bs), len(f_ks), T+1, 3)) # empty container
#
# for i, b in enumerate(f_bs):
#     for j, k in enumerate(f_ks):
#         counts_sir = run_sim(b,k,N,T) # shape = (T+1, 3), columns are S, I, R
#         f_results[i,j,...] = counts_sir / N # convert to fractions, store in results
#
# # facet plot
# f, ax = plt.subplots(len(f_bs), len(f_ks), figsize=(20, 16), sharex=True, sharey=True)
# for i, b in enumerate(f_bs):
#      for j, k in enumerate(f_ks):
#         ode.plot_sim(f_results[i,j].T, b, k, ax[i,j], accessory=False)
#
# for i, b in enumerate(f_bs):
#     ax[i, 0].set_ylabel("b = {}".format(b))
# for j, k in enumerate(f_ks):
#     ax[0, j].set_title("k = {:.2f}".format(k))
# ax[0, -1].legend()
# f.text(0.5, 0.08, 'Time', ha='center', fontsize=14)
# f.text(0.08, 0.5, 'Fraction of People', va='center', rotation='vertical', fontsize=14)
# plt.savefig("../output/agent_facet_plot.png")

#for phase diagram
p_bs = np.arange(1, 11, dtype=np.int64)
p_ks = np.logspace(-2, 0, 10)
p_results = np.zeros((len(p_bs), len(p_ks), T+1, 3)) # empty container

for i, b in enumerate(p_bs):
    for j, k in enumerate(p_ks):
        counts_sir = run_sim(b,k,N,T) # shape = (T+1, 3), columns are S, I, R
        p_results[i,j,...] = counts_sir / N
# phase diagram at time t
t = 10 # t <= T
plt.figure(figsize=(10,5))
plt.imshow(p_results[:,:,t,1], extent=[np.min(p_bs), np.max(p_bs),np.min(p_ks), np.max(p_ks)]) # 1 = infectious
plt.colorbar()
plt.yscale('log')
plt.axis('auto')
plt.xlabel('b: number of interactions')
plt.ylabel('k: recover fraction')
plt.title('Phase Diagram of Susceptible Rate at Day {}'.format(t))
plt.savefig("../output/agent_phase_plot.png")
