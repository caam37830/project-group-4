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
    # update the population
        infected = [i for i in range(N) if pop[i].is_infected()]
        all_contacts = np.array([], dtype=int)

        # each infectious person contact b people, including S,I,R
        for i in infected:
            contacts = choice(list(set(range(N))-set([i])), size=b, replace=False)
            all_contacts = np.concatenate((all_contacts, contacts))

        # contacts may get infected
        for i in all_contacts:
            pop[i].contact()

        # recover
        # k fraction of the infectious population recover
        recover = choice(infected, size=int(len(infected)*k), replace=False)
        for i in recover:
            pop[i].recover()

        # append SIR counts at time t
        counts_sir = np.vstack((counts_sir, count_sir(pop)))
    return counts_sir

# plot how s, i, and r change over the length of the simulation
sim1 = run_sim(8,0.01,1000,100) # shape = (T+1, 3), columns are S, I, R
plt.plot(sim1[:,0],label = "Susceptible")
plt.plot(sim1[:,1],label = "Infected")
plt.plot(sim1[:,2],label = "Recovered")
plt.legend()
plt.show()

# simulation for bs and ks
T = 10
N = 1000
bs = np.arange(1, 11, dtype=np.int64)
ks = np.logspace(-2,0,10)
results = np.zeros((len(bs), len(ks), T+1, 3)) # empty container

for i, b in enumerate(bs):
    for j, k in enumerate(ks):
        counts_sir = run_sim(b,k,N,T) # shape = (T+1, 3), columns are S, I, R
        results[i,j,...] = counts_sir / N # convert to fractions, store in results

# facet plot




# phase diagram at time t
t = 10 # t <= T
plt.figure(figsize=(10,5))
plt.imshow(cts[:,:,t,1], extent=[np.min(bs), np.max(bs),np.min(ks), np.max(ks)]) # 1 = infectious
plt.colorbar()
plt.yscale('log')
plt.axis('auto')
plt.xlabel('b: number of interactions')
plt.ylabel('k: recover fraction')
plt.show()
