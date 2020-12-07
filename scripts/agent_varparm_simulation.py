import sys
sys.path.append('../sir')
from agent_varparm import PopulationVarparm



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

pop = PopulationVarparm(1000, 0.001)

sirs = pop.simulation(T=100,
                    b=[10]*10 + [5]*10 + [1]*80, # social distancing
                    k=[0.01]*50 + [0.1]*30 + [0.5]*20, # drug developed and distributed
                    p=[0.1]*20 + [0.8]*80) # mutation, become more infectious


# simulation for bs and ks
T = 100
N = 1000
#for facet plot
f_bs = np.arange(1, 11, 2, dtype=np.int64)
f_ks = np.logspace(-2, 0, 5)
f_results = np.zeros((len(f_bs), len(f_ks), T+1, 3)) # empty container

for i, b in enumerate(f_bs):
    for j, k in enumerate(f_ks):
        counts_sir = run_sim(b,k,N,T) # shape = (T+1, 3), columns are S, I, R
        f_results[i,j,...] = counts_sir / N # convert to fractions, store in results

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
        counts_sir = run_sim(b,k,N,T) # shape = (T+1, 3), columns are S, I, R
        p_results[i,j,...] = counts_sir / N
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
