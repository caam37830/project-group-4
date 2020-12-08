import sys
sys.path.append('../sir')
from agent_varparm import PopulationVarparm
import pandas as pd
import matplotlib.pyplot as plt


def plot_sir(sirs, sirs2=None, save_as=None):
    T = len(sirs)
    fig, ax = plt.subplots()
    ax.plot(range(T), sirs[:,0], label = r'$s$')
    ax.plot(range(T), sirs[:,1], label = r'$i$')
    ax.plot(range(T), sirs[:,2], label = r'$r$')
    ax.set_xlabel('Time')
    ax.set_ylabel('Proportion')

    if sirs2 is not None:
        ax.plot(range(T), sirs2[:,0], label = r'$s^\prime$', c='C0', linestyle='--')
        ax.plot(range(T), sirs2[:,1], label = r'$i^\prime$', c='C1', linestyle='--')
        ax.plot(range(T), sirs2[:,2], label = r'$r^\prime$', c='C2', linestyle='--')

    fig.legend(bbox_to_anchor=(1,0.5), loc="center left", borderaxespad=0)

    if save_as:
        fig.savefig(f'../output/{save_as}.png', dpi=300, bbox_inches='tight')

    # fig.show()


# hyper parm
N = 10000
i0 = 0.001
T = 100

# base case
b, k, p = 10, 0.01, 0.1
pop = PopulationVarparm(N,i0)
sirs = pop.simulation(T=T, b=b, k=k, p=p)
plot_sir(sirs, save_as='agent_varparm_base_case')


# social distancing
pop = PopulationVarparm(N,i0)
bs = [10]*5 + [3]*25 + [1]*70 # social distancing
sirs_dist = pop.simulation(T=T, b=bs, k=k, p=p)
plot_sir(sirs, sirs_dist, 'agent_varparm_social_distancing')

# drug developement and distribution
pop = PopulationVarparm(N,i0)
ks = [0.01]*50 + [0.1]*20 + [0.8]*30
sirs_drug = pop.simulation(T=T, b=b, k=ks, p=p)
plot_sir(sirs, sirs_drug, 'agent_varparm_drug')

# virus mutation
pop = PopulationVarparm(N,i0)
ps = [0.1]*20 + [0.8]*80 # mutation, become more infectious
sirs_mut = pop.simulation(T=T, b=b, k=k, p=ps)
plot_sir(sirs, sirs_mut, 'agent_varparm_mutation')

# all together
pop = PopulationVarparm(N,i0)
sirs_all = pop.simulation(T=T, b=bs, k=ks, p=ps)
plot_sir(sirs, sirs_all, 'agent_varparm_all')
