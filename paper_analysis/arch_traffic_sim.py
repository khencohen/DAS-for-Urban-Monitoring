"""Three deployment architectures for one task (traffic monitoring) in one city (San Francisco).

Long-range DAS   : a unit follows one continuous fiber route (proxy: Euler trails of the major-road
                   network, motorway/trunk/primary and links) for up to t = 25 km; pieces are added in
                   order of the traffic they carry.
Micro-DAS        : the greedy design of the paper (traffic per interrogator), any street, 500 m per unit.
Hybrid           : half of the monitored length on long-range pieces (all of the backbone once it is
                   exhausted), the remainder on micro-DAS
                   units placed on the streets not already monitored.
Metrics versus monitored fiber length L: traffic captured (%), city coverage (% of 1 km cells that
contain a street and have a monitored street), and the number of interrogators.
"""
import os, sys, json, pickle
import numpy as np, networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import optimal_routing as orx
from backbone import backbone, BACKBONE_FRAC

OUT = os.path.dirname(os.path.abspath(__file__))

def _rasterize_maps(fig):
    """Rasterize street base maps and other dense collections so the PDF stays small."""
    for ax in fig.axes:
        for c in ax.collections:
            c.set_rasterized(True)
        for im in ax.images:
            im.set_rasterized(True)


PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'); os.makedirs(PAPER, exist_ok=True)
CITY, T_LR, CELL = 'San Francisco', 25000.0, 1000.0
MAJOR = {'motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link', 'primary_link'}

G = pickle.load(open(os.path.join(REPO, 'city_cache', CITY.replace(' ', '_') + '.pkl'), 'rb'))
edges = orx.extract_edges(G); base_lines = orx.get_base_lines(G)
hw, seen = [], set()
for u, v, k, d in G.edges(keys=True, data=True):
    ek = (min(u, v), max(u, v))
    if ek in seen:
        continue
    seen.add(ek); h = d.get('highway', 'residential'); hw.append(h[0] if isinstance(h, list) else h)
pieces, major = backbone(edges, hw)
W = np.array([e['weight'] for e in edges], float); Wt = W.sum()
L = np.array([e['length_m'] for e in edges]); mid = np.array([[e['mx'], e['my']] for e in edges]); m = len(edges)
bid = np.floor((mid - mid.min(0)) / CELL).astype(int); key = bid[:, 0] * 100000 + bid[:, 1]
cells, inv = np.unique(key, return_inverse=True); ncell = len(cells)


def metrics(sel):
    sel = np.asarray(sel, int)
    return {'km': float(L[sel].sum() / 1000), 'traffic': float(W[sel].sum() / Wt),
            'cells': float(len(np.unique(inv[sel])) / ncell)}

print(f"{CITY}: backbone {BACKBONE_FRAC*100:g}% of network, {major.sum()} segments, {L[major].sum()/1000:.0f} km, {len(pieces)} long-range pieces (<= {T_LR/1000:g} km), "
      f"median piece {np.median([L[p].sum() for p in pieces])/1000:.2f} km, max {max(L[p].sum() for p in pieces)/1000:.1f} km")

KM = [5, 10, 20, 35, 50, 75, 100, 150, 200, 250, 300, 400, 500, 700, 1000, 1400, 1853]
res = {'long_range': [], 'micro': [], 'hybrid': [], 'city_km': float(L.sum() / 1000), 'major_km': float(L[major].sum() / 1000),
       'cells': ncell, 'major_cells': float(len(np.unique(inv[major])) / ncell), 'n_pieces': len(pieces)}
for km in KM:
    budget = km * 1000.0
    sel = []; used = 0.0; n_units = 0
    for p in pieces:
        pl = L[p].sum()
        if used + pl > budget:
            continue
        sel.extend(p); used += pl; n_units += 1
    r = metrics(sel); r['units'] = n_units; res['long_range'].append(r)
    s_m = orx.optimize_traffic_km(edges, km)
    r = metrics(s_m); r['units'] = int(np.ceil(L[s_m] / orx.DAS_SENSING_RANGE).sum()); res['micro'].append(r)
    sel = []; used = 0.0; n_units = 0
    for p in pieces:
        pl = L[p].sum()
        if used + pl > budget / 2:
            continue
        sel.extend(p); used += pl; n_units += 1
    rest = [i for i in range(m) if i not in set(sel)]
    sub = [edges[i] for i in rest]
    s_sub = orx.optimize_traffic_km(sub, (budget - used) / 1000.0)      # the remainder goes to micro-DAS
    s_h = sel + [rest[j] for j in s_sub]
    r = metrics(s_h); r['units'] = n_units + int(np.ceil(L[[rest[j] for j in s_sub]] / orx.DAS_SENSING_RANGE).sum()); res['hybrid'].append(r)
    print(f"L={km:>5} km | long-range: {res['long_range'][-1]['units']:>4} units, traffic {res['long_range'][-1]['traffic']*100:5.1f}%, cells {res['long_range'][-1]['cells']*100:5.1f}% | "
          f"micro: {res['micro'][-1]['units']:>5} units, traffic {res['micro'][-1]['traffic']*100:5.1f}%, cells {res['micro'][-1]['cells']*100:5.1f}% | "
          f"hybrid: {res['hybrid'][-1]['units']:>5} units, traffic {res['hybrid'][-1]['traffic']*100:5.1f}%, cells {res['hybrid'][-1]['cells']*100:5.1f}%", flush=True)
res['KM'] = KM
json.dump(res, open(os.path.join(OUT, 'arch_traffic_sim.json'), 'w'), indent=1)

# ---------------------------------------------------------------- figure: maps at 250 km + curves
KM_MAP = 250
budget = KM_MAP * 1000.0
sel_lr = []; used = 0.0
for p in pieces:
    if used + L[p].sum() <= budget:
        sel_lr.extend(p); used += L[p].sum()
sel_mi = orx.optimize_traffic_km(edges, KM_MAP)
sel_h1 = []; used = 0.0
for p in pieces:
    if used + L[p].sum() <= budget / 2:
        sel_h1.extend(p); used += L[p].sum()
rest = [i for i in range(m) if i not in set(sel_h1)]
sel_h2 = [rest[j] for j in orx.optimize_traffic_km([edges[i] for i in rest], KM_MAP - used / 1000.0)]

FS_LABEL, FS_TICK, FS_LEG, FS_PANEL = 8, 7, 6, 10
LR_C, MI_C = '#1565C0', '#D32F2F'
fig = plt.figure(figsize=(7.0, 4.9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.35, wspace=0.42)
for k, (title, groups) in enumerate((('Long-range DAS', [(sel_lr, LR_C)]), ('Micro-DAS', [(sel_mi, MI_C)]), ('Hybrid', [(sel_h1, LR_C), (sel_h2, MI_C)]))):
    ax = fig.add_subplot(gs[0, k])
    orx.plot_city_base(ax, base_lines, color='#d0d0d0', lw=0.25, alpha=0.6)
    for sel, col in groups:
        ax.add_collection(LineCollection([edges[i]['line'] for i in sel], colors=col, linewidths=1.0, zorder=5))
    ax.autoscale(); ax.set_aspect('equal'); ax.axis('off')
    r = metrics([i for g in groups for i in g[0]]); units = (len([p for p in pieces if set(p) <= set(sel_lr)]) if k == 0 else
             (int(np.ceil(L[sel_mi] / 500).sum()) if k == 1 else len([p for p in pieces if set(p) <= set(sel_h1)]) + int(np.ceil(L[sel_h2] / 500).sum())))
    ax.set_title(f"{title}: {r['km']:.0f} km, {units} units\ntraffic {r['traffic']*100:.0f}%, city cells {r['cells']*100:.0f}%", fontsize=FS_LEG + 0.5)
    ax.text(-0.02, 1.12, 'ABC'[k], transform=ax.transAxes, fontsize=FS_PANEL, fontweight='bold', va='bottom', ha='right')
cols = {'long_range': (LR_C, 'Long-range DAS'), 'micro': (MI_C, 'Micro-DAS'), 'hybrid': ('#6a1b9a', 'Hybrid')}
for k, (metric, lab) in enumerate((('traffic', 'Traffic captured (%)'), ('cells', 'City cells monitored (%)'), ('units', 'Interrogators'))):
    ax = fig.add_subplot(gs[1, k])
    for arch, (col, name) in cols.items():
        y = [r[metric] * (1 if metric == 'units' else 100) for r in res[arch]]
        ax.plot(KM, y, color=col, lw=1.5, label=name)
    ax.set_xscale('log'); ax.set_xlabel('Monitored fiber length (km)', fontsize=FS_LABEL); ax.set_ylabel(lab, fontsize=FS_LABEL)
    if metric == 'units':
        ax.set_yscale('log')
    ax.grid(True, alpha=0.22, lw=0.5); ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.6)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    ax.text(-0.28, 1.02, 'DEF'[k], transform=ax.transAxes, fontsize=FS_PANEL, fontweight='bold', va='bottom')
    if k == 0:
        ax.legend(fontsize=FS_LEG, loc='upper left', frameon=False)
_rasterize_maps(fig)
fig.savefig(os.path.join(OUT, 'fig_architectures_traffic.pdf'), dpi=300, bbox_inches='tight')
_rasterize_maps(fig)
fig.savefig(os.path.join(PAPER, 'fig_architectures_traffic.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig_architectures_traffic.png'), dpi=150, bbox_inches='tight')
print('saved arch_traffic_sim.json and figure')
