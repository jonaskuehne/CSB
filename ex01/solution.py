import numpy as np
import matplotlib.pyplot as plt

def load_graph():
    file = open("allyeastJan14_2002-nr.txt", "r")

    edges = []
    nodes = set()

    for line in file:
        proteins = line.rstrip().split("\t")
        nodes.update(set(proteins))
        edges.append(proteins)
    
    nodes = list(sorted(nodes))
    
    nodes_to_index = {val: i for i, val in enumerate(nodes)}
    
    e = np.zeros((len(nodes), len(nodes)), dtype=np.uint8)
    vdeg = np.zeros((len(nodes)), dtype=np.uint32)
    
    for (p1, p2) in edges:
        ind1 = nodes_to_index[p1]
        ind2 = nodes_to_index[p2]
    
        vdeg[ind1] += 1
        vdeg[ind2] += 1
    
        e[min(ind1, ind2), max(ind1, ind2)] = 1

    return nodes, edges, nodes_to_index

def create_e_vdeg(nodes, edges, nodes_to_index):
    e = np.zeros((len(nodes), len(nodes)), dtype=np.uint8)
    vdeg = np.zeros((len(nodes)), dtype=np.uint32)
    
    for (p1, p2) in edges:
        ind1 = nodes_to_index[p1]
        ind2 = nodes_to_index[p2]
    
        vdeg[ind1] += 1
        vdeg[ind2] += 1
    
        e[min(ind1, ind2), max(ind1, ind2)] = 1

    return e, vdeg

from collections import Counter
def create_hist(vdeg):
    # 1. Count frequencies
    counts = Counter(vdeg)
    x, y = zip(*sorted(counts.items()))

    # 2. Plot as scatter or line
    plt.scatter(x, y)

    # 3. Scale axes
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Degree (k)')
    plt.ylabel('Frequency')
    plt.show()
    

nodes, edges, nodes_to_index = load_graph()
e, vdeg = create_e_vdeg(nodes, edges, nodes_to_index)
create_hist(vdeg)

