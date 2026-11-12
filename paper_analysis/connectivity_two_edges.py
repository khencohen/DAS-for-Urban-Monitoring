"""Connectivity at the two limits of fiber accessibility.

random   : uncoordinated access; bond percolation, largest connected component of the
           instrumented subgraph (Newman-Ziff, exact in the number of instrumented edges).
designed : full access; the largest component obtainable with k instrumented segments on a
           connected street graph is a tree on k+1 junctions, so P_inf = min(k+1, n)/n and
           every junction is connected at k = n-1 (a spanning tree), c = (n-1)/m ~ 2/<k>.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_graph

OUT = os.path.dirname(os.path.abspath(__file__))
CITIES = ['Tel_Aviv', 'Barcelona', 'Manhattan', 'Berlin', 'Paris', 'Amsterdam', 'Rome',
          'Buenos_Aires', 'San_Francisco', 'Tokyo']
N_PERM = 20
F_LIST = [0.1, 0.5, 0.9, 0.99]


def newman_ziff(E, n, m, rng):
    parent = list(range(n)); size = [1] * n
    order = rng.permutation(m)
    big = 1; s = np.empty(m + 1); s[0] = 1
    for k, e in enumerate(order, 1):
        a, b = int(E[e, 0]), int(E[e, 1])
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]; b = parent[b]
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a; size[a] += size[b]
            if size[a] > big:
                big = size[a]
        s[k] = big
    return s / n


def main():
    res = {}
    for name in CITIES:
        G = load_graph(name)
        nodes = list(G.nodes()); idx = {u: i for i, u in enumerate(nodes)}
        E = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=np.int64)
        n, m = len(nodes), len(E)
        rng = np.random.default_rng(7)
        curves = np.array([newman_ziff(E, n, m, rng) for _ in range(N_PERM)])
        mean = curves.mean(0)
        c_axis = np.arange(m + 1) / m
        c_rand = {}
        for f in F_LIST:
            per = [float(c_axis[np.argmax(cv >= f)]) for cv in curves]
            c_rand[f] = (float(np.mean(per)), float(np.std(per)))
        c_des = {f: (f * n - 1) / m for f in F_LIST}
        sub = np.unique(np.linspace(0, m, 400).astype(int))
        res[name] = {'n': n, 'm': m, 'k': 2 * m / n, 'c_random': c_rand, 'c_designed': c_des,
                     'c_spanning_tree': (n - 1) / m,
                     'curve_c': c_axis[sub].tolist(), 'curve_Pinf': mean[sub].tolist()}
        print(f"{name:<14} <k>={2*m/n:.2f}  P=0.5: random {c_rand[0.5][0]:.3f} designed {c_des[0.5]:.3f} | "
              f"P=0.99: random {c_rand[0.99][0]:.3f} designed {c_des[0.99]:.3f} | spanning tree {(n-1)/m:.3f}", flush=True)
    json.dump({k: {kk: ({str(a): b for a, b in vv.items()} if isinstance(vv, dict) else vv) for kk, vv in v.items()}
               for k, v in res.items()}, open(os.path.join(OUT, 'connectivity_two_edges.json'), 'w'), indent=1)
    print('saved connectivity_two_edges.json')


if __name__ == '__main__':
    main()
