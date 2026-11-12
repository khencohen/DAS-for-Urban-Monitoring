"""Backbone routes: the proxy for continuous long-range fiber routes (round 10).

Round 9 and earlier used every OpenStreetMap motorway, trunk and primary road (9-38% of the network).
Following A. Lellouch's comment, the backbone is now the top BACKBONE_FRAC = 4% of the drivable network
length: Euler trails of the motorway/trunk/primary graph (cut into pieces of at most T_LR = 25 km),
added in decreasing order of the traffic they carry until 4% of the total street length is reached.
"""
import numpy as np, networkx as nx

MAJOR = {'motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link', 'primary_link'}
BACKBONE_FRAC, T_LR = 0.04, 25000.0


def major_pieces(edges, hw, t_lr=T_LR):
    """Euler-trail pieces (lists of edge indices) of the major-road graph, longest traffic first."""
    L = np.array([e['length_m'] for e in edges]); W = np.array([e['weight'] for e in edges], float)
    H = nx.MultiGraph()
    for i in np.where(np.array([h in MAJOR for h in hw]))[0]:
        H.add_edge(edges[i]['u'], edges[i]['v'], idx=int(i), real=True)
    pieces = []
    for comp in nx.connected_components(H):
        C = H.subgraph(comp).copy()
        odd = [n for n in C.nodes if C.degree(n) % 2]
        for a, b in zip(odd[::2], odd[1::2]):
            C.add_edge(a, b, idx=-1, real=False)
        cur, cur_len = [], 0.0
        for u, v, k in nx.eulerian_circuit(C, keys=True):
            d = C.edges[u, v, k]
            if d['real']:
                if cur_len + L[d['idx']] > t_lr and cur:
                    pieces.append(cur); cur, cur_len = [], 0.0
                cur.append(d['idx']); cur_len += L[d['idx']]
            else:
                if cur:
                    pieces.append(cur)
                cur, cur_len = [], 0.0
        if cur:
            pieces.append(cur)
    pieces.sort(key=lambda p: -W[p].sum())
    return pieces


def backbone(edges, hw, frac=BACKBONE_FRAC, t_lr=T_LR):
    """Return (pieces, mask): the backbone pieces and the boolean edge mask of the backbone."""
    L = np.array([e['length_m'] for e in edges]); cap = frac * L.sum()
    pieces, used, out = major_pieces(edges, hw, t_lr), 0.0, []
    for p in pieces:
        pl = L[p].sum()
        if used + pl > cap:
            continue
        out.append(p); used += pl
    mask = np.zeros(len(edges), bool)
    for p in out:
        mask[p] = True
    return out, mask
