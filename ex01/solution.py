import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

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

def create_hist(vdeg):
    counts = Counter(vdeg)
    x, y = zip(*sorted(counts.items()))

    plt.scatter(x, y)

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Degree (k)')
    plt.ylabel('Frequency')
    plt.savefig("hist.png")

def k_cores(vdeg, e):
    cores = np.zeros((len(vdeg)), dtype=np.uint32)
    sorted_indices = sorted(range(len(vdeg)), key=lambda i: vdeg[i])

    adj_list = defaultdict(set)
    for v in range(len(vdeg)):
        for u in range(len(vdeg)):
            if e[u,v] or e[v,u]:
                adj_list[v].add(u)

    while sorted_indices:
        v = sorted_indices.pop(0)
        cores[v] = vdeg[v]
        reorder = False
        for u in adj_list[v]:
            if vdeg[u] > vdeg[v]:
                vdeg[u] -= 1
                reorder = True
        if reorder:
            sorted_indices.sort(key=lambda v: vdeg[v])

    return cores
    
    

nodes, edges, nodes_to_index = load_graph()
e, vdeg = create_e_vdeg(nodes, edges, nodes_to_index)
create_hist(vdeg)
cores = k_cores(vdeg, e)
print(f"number of cores: {len(set(cores))}, max: {max(cores)}, min: {min(cores)}")

