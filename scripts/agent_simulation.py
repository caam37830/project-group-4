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
from numpy.random import randint, rand
from sir import *
sys.path.append("..")

def count_infected(pop):
    return sum(p.is_infected() for p in pop)
def count_recovered(pop):
    return sum(p.is_recovered() for p in pop)
def count_susceptible(pop,N):
    return N - count_infected(pop) - count_recovered(pop)

def run_sim(b, k, N, T):
    """
    return the number of people enlightened at time T
    """
    pop = [Person() for i in range(N)] # our population
    infect = randint(N, size = int(0.001*N))
    for i in infect:
        pop[i].infect()
    counts_I = [count_infected(pop)]
    counts_R = [count_recovered(pop)]
    counts_S = [count_susceptible(pop,N)]
    for t in range(T):
    # update the population
        for i in range(N):
            if pop[i].is_infected():
        # person i infect susceptible with infection rate b/N
                for j in range(N):
                    if not pop[j].is_infected() and not pop[j].is_recovered():
                        if rand() < b/N:
                            pop[j].infect()
        # infected population get recovered with rate k
            if rand() < k:
                pop[i].recover()
          # add to our counts
        counts_I.append(count_infected(pop))
        counts_R.append(count_recovered(pop))
        counts_S.append(count_susceptible(pop,N))
    return  counts_I,counts_R,counts_S

# plot how s, i, and r change over the length of the simulation
sim1 = run_sim(5,0.01,5000,20)
print(sim1[0])
plt.plot(sim1[2],label = "Susceptible")
plt.plot(sim1[0],label = "Infected")
plt.plot(sim1[1],label = "Recovered")
plt.legend()
plt.show()

T = 20
N = 1000
bs = np.arange(1,3, dtype=np.int64)
ks = np.logspace(-2,-1,10)
cts_I = np.zeros((len(bs), len(ks),T+1))
cts_R = np.zeros((len(bs), len(ks),T+1))
cts_S = np.zeros((len(bs), len(ks),T+1))

for i, b in enumerate(bs):
    for j, k in enumerate(ks):
        cts_I[i,j,],cts_R,cts_S = run_sim(b,k,N,T)
#phase diagram at time t
t = 10
plt.figure(figsize=(10,5))
plt.imshow(cts_I[:len(bs),:len(ks),t], extent=[np.min(bs), np.max(bs),np.min(ks), np.max(ks)])
plt.colorbar()
# plt.axis('square')
plt.yscale('log')
plt.xlabel('b')
plt.ylabel('k')
plt.show()


