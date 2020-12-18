import sys
sys.path.append('../sir')
from agent import Person
from agent_pop import Population
from agent_pop2d import Population2d
from pde import *

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint, rand, choice
from time import time
from tqdm import tqdm

# Person() vs Population()
def agent_person_sim(b, k, N, T):
    """
    return the number of people in each group from time 0 to time T
    """
    pop = [Person() for i in range(N)] # our population
    infected = choice(range(N), size=int(N*0.001), replace=False) # 0.1% infected at T=0
    for i in infected:
        pop[i].infect()

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

b = 5
k = 0.1
Ns = np.logspace(2,4,10).astype(int)
T = 100

t_person = []
t_pop_slow = []
t_pop = []

for N in tqdm(Ns):
    t0 = time()
    agent_person_sim(b,k,N,T)
    t_person.append(time() - t0)

    pop = Population(N, 0.001)
    t0 = time()
    for t in range(T):
        pop.infect_slow(b)
        pop.recover(k)
    t_pop_slow.append(time() - t0)

    pop = Population(N, 0.001)
    t0 = time()
    for t in range(T):
        pop.infect(b)
        pop.recover(k)
    t_pop.append(time() - t0)


logNs = np.log10(Ns)
plt.plot(logNs, np.log10(t_person), c='C0', label='Person()')
plt.plot(logNs, np.log10(t_pop_slow), c='C1', label='Population()')
plt.plot(logNs, np.log10(t_pop), c='C2', label='Population() improved')
plt.xlabel(r"$\log_{10}(N)$")
plt.ylabel(r"$\log_{10}(t)$")
plt.legend()
plt.title('Computation times for different methods in agent-based models')
plt.savefig('../docs/figs/comp_perf_agent.png',dpi=200)
t_person[-1]/t_pop_slow[-1]
t_pop_slow[-1]/t_pop[-1]

# Pop2d
t_pop2d_slow = []
t_pop2d = []

T=100; p=0.2; q=0.1; k=0.1
for N in tqdm(Ns):
    q = np.sqrt(b/(np.pi*N))
    pop = Population2d(N, int(N*0.001))
    t0 = time()
    for t in range(T):
        pop.move(p)
        pop.infect(q)
        pop.recover(k)
    t_pop2d_slow.append(time() - t0)

    pop = Population2d(N, int(N*0.001))
    t0 = time()
    for t in range(T):
        pop.move(p)
        pop.infect_opt(q)
        pop.recover(k)
    t_pop2d.append(time() - t0)

plt.plot(logNs, np.log10(t_pop2d_slow), c='C3', label=r'Query by $I$')
plt.plot(logNs, np.log10(t_pop2d), c='C4', label=r'Query by $I$ or $S$')
plt.xlabel(r"$\log_{10}(N)$")
plt.ylabel(r"$\log_{10}(t)$")
plt.legend()
plt.title('Computation times for different methods in 2-d agent-based models')
plt.savefig('../docs/figs/comp_perf_agent2d.png',dpi=200)
t_pop2d_slow[-1]/t_pop2d[-1]
t_pop2d[-1]

# PDE
M = 200; i0 = 0.001; T = 100; b = 5; k = 0.05

pos = (M // 2, M // 2)
ps = np.logspace(-4, -1, 10) / M

ts = []
for p in ps:
    t0 = time()
    pde_model(i0, pos, M, T, b, k, p)
    ts.append(time()-t0)

plt.plot(ps, np.log10(ts), c='C5')
plt.xlabel(r"$p$")
plt.ylabel(r"$\log_{10}(t)$")
plt.title(r'Computation times for different $p$ in the PDE model')
plt.savefig('../docs/figs/comp_perf_pde.png',dpi=200)
