import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

EFM_PATH = "efms.json"
with open(EFM_PATH) as f:
  efms_file = json.load(f)

N = efms_file['stoich']
efms = efms_file['efms']
bounds = efms_file['reactionLowerBounds']
reaction_names = efms_file['reactionNames']

M_count = len(N)
R_count = len(N[0])
EFMs_count = len(efms[0])
tol = 1e-7

print(f'Metabolities count: {M_count}')
print(f'Reactions count: {R_count}')
print(f'EFMs count: {EFMs_count}')

N = np.array(N)
efms = np.array(efms)
bounds = np.array(bounds)

def check_constraints(N, efms, bounds):
    mat_res = N @ efms
    
    # Ne=0?
    if np.allclose(mat_res, 0, atol=tol):
        print("all EFMs have N*e=0")
    else:
        print("some EFMs have N*e!=0")
    
    # for irreversible reactions ri >= 0?
    mask = (np.sum(efms < 0, axis=1) > 0)
    rel_bounds = bounds[mask]
    if (rel_bounds < 0).all():
        print("all irrev EFMs have ri >= 0")
    else:
        print("some irrev EFMs have ri < 0")

def min_cuts(efms, reaction_names):
    growth_reaction = reaction_names.index("mue")
    growth_fluxes = np.abs(efms[growth_reaction, :]) > tol
    num_growth_reactions = np.sum(growth_fluxes)
    print(f"sum of growth-related EFMs: {num_growth_reactions}")
    
    path_lengths = (np.abs(efms) > tol).sum(axis=0)
    
    plt.hist(path_lengths, color = 'blue', edgecolor = 'black', alpha = 0.5, label = 'All EFM Pathway Lengths')
    plt.hist(path_lengths[growth_fluxes], color = 'red', edgecolor = 'black', alpha = 0.5, label = 'Growth EFM Pathway Lengths')
    plt.legend(loc='upper left')
    plt.xlabel('EFM Pathway Length')
    plt.ylabel('EFM Count')
    plt.savefig("hist.png")
    
    mc_1_reactions = []
    non_essential = []
    reaction_fluxes_list = []
    for i in range(R_count):
        reaction_fluxes = np.abs(efms[i, :]) > tol
        reaction_fluxes_growth = np.bitwise_and(growth_fluxes, reaction_fluxes)
        reaction_fluxes_list.append(reaction_fluxes_growth)
        participation = np.sum(reaction_fluxes_growth) / num_growth_reactions
        if participation == 1:
            mc_1_reactions.append({'reaction': reaction_names[i], 'participation': participation})
        else:
            non_essential.append(i)
    
    print(f"reactions with 100% participation: {len(mc_1_reactions)}")
    print(*mc_1_reactions, sep="\n")
    
    
    mc_2_reactions = []
    for r1, r2 in list(combinations(non_essential, 2)):
        reaction_fluxes_growth = np.bitwise_or(reaction_fluxes_list[r1], reaction_fluxes_list[r2])
        participation = np.sum(reaction_fluxes_growth) / num_growth_reactions
        if participation == 1:
            mc_2_reactions.append({'reaction_1': reaction_names[r1], 'reaction_2': reaction_names[r2], 'participation': participation})
    
    print(f"minimal reaction pairs with 100% participation: {len(mc_2_reactions)}")
    print(*mc_2_reactions, sep="\n")

check_constraints(N, efms, bounds)
min_cuts(efms, reaction_names)

