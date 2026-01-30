import numpy as np
from itertools import product

A = np.array([
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0]
])

C = np.array([
    [0.1, 0.7, 0.8, 0.1, 1, 0.6, 0.7],
    [0.2, 0.9, 0, 0.4, 1, 0, 0.1],
    [0.1, 0.8, 0.1, 0.7, 1, 0.9, 0.7],
    [0, 1, 0.8, 0.9, 0.9, 0, 0.9],
    [0, 0.9, 0, 1, 0.7, 0.7, 0.8]
])

cutoff = 0.5

C_bit = (C >= cutoff).astype(int)

def get_parents(A, node):
    return np.where(A[:, node])[0]

probs = []
for node in range(len(A)):
    probs_node = []
    pa = get_parents(A, node)
    # {0,1}^|pa|
    combs = [list(p) for p in product([0, 1], repeat=len(pa))]
    for c in combs:
        # cols where the parents have the values of c
        matching_cols = np.where((C_bit[pa] == np.array(c)[:, None]).all(axis=0))[0]
        # rows where node is 1
        counts_per_row = C_bit[:, matching_cols].sum(axis=1)[node]
        prob = counts_per_row / len(matching_cols)
        probs_node.append([1-prob, prob])

    probs.append(probs_node)

# marginalize C
def margin_C(probs):
    p = 0
    for a, b, d, e in product(range(2), repeat=4):
        p += probs[0][0][a] * probs[4][0][e] * probs[1][2*a + e][b] * probs[2][b][1] * probs[3][a][d]
    return p

# marginalize A,C
def margin_A_C(probs):
    p = 0
    for b, d, e in product(range(2), repeat=3):
        p += probs[0][0][1] * probs[4][0][e] * probs[1][2 + e][b] * probs[2][b][1] * probs[3][1][d]
    return p

# marginalize E,C
def margin_E_C(probs):
    p = 0
    for a, b, d in product(range(2), repeat=3):
        p += probs[0][0][a] * probs[4][0][1] * probs[1][2*a + 1][b] * probs[2][b][1] * probs[3][a][d]
    return p

p_C = margin_C(probs)
p_A_C = margin_A_C(probs)
p_E_C = margin_E_C(probs)


print(f"p(A=1|C=1) = {p_A_C / p_C}, p(E=1|C=1) = {p_E_C / p_C}")

