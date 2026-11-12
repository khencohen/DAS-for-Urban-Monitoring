"""Does the giant component mean city-scale coverage?  Cluster extent on real street networks.

For random placement at coverage c, a signal can be followed along instrumented streets only within
the connected instrumented cluster. The cluster extent E(c) is the geographic size (bounding-box
diagonal) of the cluster that contains a randomly chosen instrumented street, averaged over streets,
and normalized by the size of the city. We compare the coverage at which E reaches half (and 90%) of
the city with the percolation threshold c* (P_inf = 0.5) of the same network.
"""
import os, sys, json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_graph

OUT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'); os.makedirs(PAPER, exist_ok=True)
CITIES = ['Tel_Aviv', 'Barcelona', 'Manhattan', 'Berlin', 'Paris', 'Amsterdam', 'Rome', 'Buenos_Aires', 'San_Francisco', 'Tokyo']
NAMES = {'Tel_Aviv': 'Tel Aviv', 'Buenos_Aires': 'Buenos Aires', 'San_Francisco': 'San Francisco', 'Tokyo': 'Shibuya (Tokyo)'}
CS = np.linspace(0.0, 1.0, 101)[1:]
N_RUN = 5


def crossing(cs, y, level):
    y = np.maximum.accumulate(y)
    if y[-1] < level:
        return float('nan')
    j = int(np.argmax(y >= level))
    return float(cs[j - 1] + (level - y[j - 1]) * (cs[j] - cs[j - 1]) / (y[j] - y[j - 1])) if j > 0 else float(cs[0])


res = {}
for name in CITIES:
    G = load_graph(name)
    nodes = list(G.nodes()); idx = {u: i for i, u in enumerate(nodes)}
    E = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=np.int64)
    xy = np.array([[G.nodes[u]['x'], G.nodes[u]['y']] for u in nodes]); n, m = len(nodes), len(E)
    city_diag = float(np.hypot(*(xy.max(0) - xy.min(0))))
    rng = np.random.default_rng(3)
    ext = np.zeros((N_RUN, len(CS))); gcc = np.zeros((N_RUN, len(CS)))
    for r in range(N_RUN):
        p = rng.random(m)
        for j, c in enumerate(CS):
            keep = p < c; Ek = E[keep]; mk = len(Ek)
            if mk == 0:
                continue
            A = sp.coo_matrix((np.ones(mk), (Ek[:, 0], Ek[:, 1])), shape=(n, n))
            ncomp, lab = connected_components(A, directed=False)
            xmin = np.full(ncomp, np.inf); xmax = np.full(ncomp, -np.inf); ymin = np.full(ncomp, np.inf); ymax = np.full(ncomp, -np.inf)
            np.minimum.at(xmin, lab, xy[:, 0]); np.maximum.at(xmax, lab, xy[:, 0]); np.minimum.at(ymin, lab, xy[:, 1]); np.maximum.at(ymax, lab, xy[:, 1])
            diag = np.hypot(xmax - xmin, ymax - ymin)
            ecount = np.bincount(lab[Ek[:, 0]], minlength=ncomp)
            ext[r, j] = (ecount * diag).sum() / mk / city_diag
            gcc[r, j] = np.bincount(lab, minlength=ncomp).max() / n
    e, g = ext.mean(0), gcc.mean(0)
    res[name] = {'k': 2 * m / n, 'city_km': city_diag / 1000, 'c': CS.tolist(), 'extent': e.tolist(), 'gcc': g.tolist(),
                 'c_star': crossing(CS, g, 0.5), 'c_ext50': crossing(CS, e, 0.5), 'c_ext90': crossing(CS, e, 0.9),
                 'extent_at_cstar': float(np.interp(crossing(CS, g, 0.5), CS, e)), 'extent_at_cstar_minus10': float(np.interp(crossing(CS, g, 0.5) - 0.1, CS, e))}
    rr = res[name]
    print(f"{name:<14} city {city_diag/1000:5.1f} km  c*={rr['c_star']:.3f}  c(E=50%)={rr['c_ext50']:.3f}  c(E=90%)={rr['c_ext90']:.3f}  "
          f"E(c*)={rr['extent_at_cstar']*100:.0f}%  E(c*-0.1)={rr['extent_at_cstar_minus10']*100:.0f}%", flush=True)
json.dump(res, open(os.path.join(OUT, 'city_scale_extent.json'), 'w'), indent=1)

FS_LABEL, FS_TICK, FS_LEG, FS_PANEL = 8, 7, 5.5, 10
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
order = sorted(CITIES, key=lambda c: res[c]['c_star']); cmap = plt.cm.viridis
for i, name in enumerate(order):
    r = res[name]; col = cmap(i / (len(order) - 1))
    a1.plot(np.array(r['c']) * 100, np.array(r['extent']) * 100, color=col, lw=1.1, label=f"{NAMES.get(name, name)}")
    a1.plot([r['c_star'] * 100], [r['extent_at_cstar'] * 100], 'o', ms=3, color=col, mec='k', mew=0.4, zorder=5)
    a2.scatter(r['c_star'] * 100, r['c_ext50'] * 100, s=28, color=col, edgecolors='k', linewidths=0.4, zorder=5)
a1.set_xlabel('Coverage ratio $c$ (%)', fontsize=FS_LABEL); a1.set_ylabel('Cluster extent (% of city size)', fontsize=FS_LABEL)
a1.set_xlim(0, 100); a1.set_ylim(0, 102)
a1.legend(fontsize=FS_LEG, loc='upper left', frameon=False, ncol=2, columnspacing=0.8, handlelength=1.3)
a2.plot([50, 80], [50, 80], 'k--', lw=0.8, alpha=0.6, label='equality')
a2.set_xlabel('Percolation threshold $c^*$ (%)', fontsize=FS_LABEL); a2.set_ylabel('Coverage at half-city extent (%)', fontsize=FS_LABEL)
a2.set_xlim(52, 78); a2.set_ylim(52, 78); a2.legend(fontsize=FS_LEG, loc='upper left', frameon=False)
for ax, lab in zip((a1, a2), 'AB'):
    ax.grid(True, alpha=0.22, lw=0.5); ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.6)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    ax.text(-0.17, 1.06, lab, transform=ax.transAxes, fontsize=FS_PANEL, fontweight='bold', va='top')
plt.tight_layout(pad=0.4, w_pad=2.0)
fig.savefig(os.path.join(OUT, 'fig_city_scale_extent.pdf'), dpi=300)
fig.savefig(os.path.join(PAPER, 'fig_city_scale_extent.pdf'), dpi=300)
fig.savefig(os.path.join(OUT, 'fig_city_scale_extent.png'), dpi=160)
print('saved city_scale_extent.json and figure')
