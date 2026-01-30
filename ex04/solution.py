import json
import numpy as np

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

print(f'Metabolities count: {M_count}')
print(f'Reactions count: {R_count}')
print(f'EFMs count: {EFMs_count}')

N = np.array(N)
efms = np.array(efms)

mat_res = N @ efms

tol = 1e-7
# Ne=0?
if not np.allclose(mat_res, 0, atol=tol):
    # check which violate
    mask = ~np.isclose(mat_res, 0, atol=tol)
    col_indices = np.where(mask.any(axis=0))[0]
    
    print(f"EFMs that violate N*e=0: {col_indices}")
else:
    print("all EFMs have N*e=0")
