"""Direction 1: sensing range with fiber continuity, and a quantitative three-architecture comparison.

Part A (range with continuity). For the six traffic layouts of Fig. 3, count interrogators when
one unit of range t can follow the fiber through consecutive monitored segments:
  none   : one unit per segment, ceil(l / t)                            (model of the paper)
  street : continuity along consecutive segments of the same named street
  any    : continuity through any junction of the monitored subgraph    (optimistic bound)
A unit interrogates a trail (no branching). Trails come from an Euler decomposition of each
component (odd-degree junctions paired by virtual edges), and a trail of length L needs ceil(L / t) units.

Part B (architectures). Proxy for continuous metro/feeder/backbone routes: OpenStreetMap
motorway, trunk and primary roads with their links ("major roads"). Architectures:
  long-range : major roads interrogated by long-range units (t = 10, 25, 50 km; continuity 'any')
  distributed: the greedy short-range designs of the paper
  hybrid     : major roads by long-range units plus the same short-range budget on the other streets
"""
import os, sys, json, pickle
import numpy as np, networkx as nx
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import optimal_routing as orx
from backbone import backbone, BACKBONE_FRAC

OUT = os.path.dirname(os.path.abspath(__file__))
MAJOR = {'motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link', 'primary_link'}


def load(city):
    G = pickle.load(open(os.path.join(REPO, 'city_cache', city.replace(' ', '_') + '.pkl'), 'rb'))
    edges = orx.extract_edges(G)
    meta, seen = [], set()
    for u, v, k, d in G.edges(keys=True, data=True):          # same order and de-duplication as extract_edges
        ek = (min(u, v), max(u, v))
        if ek in seen:
            continue
        seen.add(ek)
        hw = d.get('highway', 'residential'); hw = hw[0] if isinstance(hw, list) else hw
        nm = d.get('name'); nm = nm[0] if isinstance(nm, list) else nm
        meta.append((hw, nm))
    assert len(meta) == len(edges)
    return G, edges, meta


def trails(edge_list):
    """Euler decomposition of the subgraph formed by edge_list [(u, v, length)] into trail lengths."""
    H = nx.MultiGraph()
    for u, v, l in edge_list:
        H.add_edge(u, v, real=True, length=l)
    out = []
    for comp in nx.connected_components(H):
        C = H.subgraph(comp).copy()
        odd = [n for n in C.nodes if C.degree(n) % 2]
        for a, b in zip(odd[::2], odd[1::2]):
            C.add_edge(a, b, real=False, length=0.0)
        cur = 0.0; started = False
        for u, v, k in nx.eulerian_circuit(C, keys=True):
            d = C.edges[u, v, k]
            if d['real']:
                cur += d['length']; started = True
            else:
                if started:
                    out.append(cur)
                cur = 0.0; started = False
        if started:
            out.append(cur)
    return out


def units(edges, meta, idx, t, mode):
    L = np.array([edges[i]['length_m'] for i in idx])
    if mode == 'none':
        return int(np.maximum(1, np.ceil(L / t)).sum())
    if mode == 'any':
        return int(sum(max(1, int(np.ceil(x / t))) for x in trails([(edges[i]['u'], edges[i]['v'], edges[i]['length_m']) for i in idx])))
    groups = {}
    for i in idx:
        key = meta[i][1] if meta[i][1] else ('__unnamed__', i)
        groups.setdefault(key, []).append(i)
    n = 0
    for g in groups.values():
        n += sum(max(1, int(np.ceil(x / t))) for x in trails([(edges[i]['u'], edges[i]['v'], edges[i]['length_m']) for i in g]))
    return n


def part_a():
    BUD = {'San Francisco': [50, 250], 'Berlin': [163, 816], 'Singapore': [122, 611]}
    res = {}
    for city, kms in BUD.items():
        G, edges, meta = load(city)
        for km in kms:
            sel = orx.optimize_traffic_km(edges, km)
            row = {}
            for t in (500, 2000, 10000, 25000):
                row[t] = {m: units(edges, meta, sel, t, m) for m in ('none', 'street', 'any')}
            res[f'{city} {km} km'] = row
            print(f"A {city:<14} {km:>4} km: " + " | ".join(f"t={t/1000:g}km none {r['none']} street {r['street']} any {r['any']}" for t, r in row.items()), flush=True)
    return res


def part_b():
    res = {}
    # ---- traffic
    BUD = {'San Francisco': [354, 1974], 'Berlin': [1089, 5662], 'Singapore': [958, 4661]}
    for city, Bs in BUD.items():
        G, edges, meta = load(city)
        W = np.array([e['weight'] for e in edges], float); Wt = W.sum()
        Lm = np.array([e['length_m'] for e in edges])
        C = np.ceil(Lm / orx.DAS_SENSING_RANGE).astype(int)
        pieces, major = backbone(edges, [x[0] for x in meta])
        midx = np.where(major)[0]
        lr_units = {t: units(edges, meta, midx, t, 'any') for t in (10000, 25000, 50000)}
        eff = W / np.maximum(C, 1); order = np.argsort(-eff, kind='stable')

        def greedy(B, allowed):
            used = 0; val = 0.0
            for i in order:
                if allowed[i] and used + C[i] <= B:
                    used += C[i]; val += W[i]
            return val
        major_share = W[major].sum() / Wt
        # short-range units needed to match the long-range-only traffic share
        used = 0; val = 0.0; need = None
        for i in order:
            used += C[i]; val += W[i]
            if val >= W[major].sum():
                need = used; break
        rows = {}
        for B in Bs:
            rows[B] = {'distributed': greedy(B, np.ones(len(W), bool)) / Wt,
                       'hybrid': (W[major].sum() + greedy(B, ~major)) / Wt}
        res[f'traffic {city}'] = {'major_km': float(Lm[major].sum() / 1000), 'total_km': float(Lm.sum() / 1000),
                                  'major_segments': int(major.sum()), 'long_range_units': lr_units,
                                  'long_range_share': float(major_share), 'short_units_to_match': need, 'budgets': rows}
        print(f"B traffic {city}: major {Lm[major].sum()/1000:.0f} of {Lm.sum()/1000:.0f} km, LR units {lr_units}, "
              f"LR-only {major_share*100:.1f}% (matched by {need} short units); " +
              "; ".join(f"B={B}: distributed {r['distributed']*100:.1f}% hybrid {r['hybrid']*100:.1f}%" for B, r in rows.items()), flush=True)
    # ---- activity (Paris)
    G, edges, meta = load('Paris')
    pop = np.maximum(orx.load_population_per_edge(edges, G, 'Paris'), 0).astype(float); Pt = pop.sum()
    mid = np.array([[e['mx'], e['my']] for e in edges]); m = len(mid)
    Lm = np.array([e['length_m'] for e in edges])
    tree = cKDTree(mid); pairs = tree.query_ball_tree(tree, r=orx.ACTIVITY_COVERAGE_RADIUS)
    r_, c_ = [], []
    for i, nb in enumerate(pairs):
        r_.extend([i] * len(nb)); c_.extend(nb)
    cov = csr_matrix((np.ones(len(r_), np.float32), (r_, c_)), shape=(m, m))
    pieces, major = backbone(edges, [x[0] for x in meta]); midx = np.where(major)[0]
    base_cov = np.asarray(cov[midx].sum(0)).ravel() > 0

    def greedy_act(T, covered0, allowed):
        covered = covered0.copy(); n = 0
        for _ in range(T):
            g = np.asarray(cov.dot(pop * (~covered))).ravel(); g[~allowed] = -1
            b = int(np.argmax(g))
            if g[b] <= 0:
                break
            allowed = allowed.copy(); allowed[b] = False
            covered |= np.asarray(cov[b].toarray()).ravel().astype(bool); n += 1
        return pop[covered].sum() / Pt
    lr_share = pop[base_cov].sum() / Pt
    need = None
    for T in range(1, 400):
        if greedy_act(T, np.zeros(m, bool), np.ones(m, bool)) >= lr_share:
            need = T; break
    rows = {T: {'distributed': greedy_act(T, np.zeros(m, bool), np.ones(m, bool)),
                'hybrid': greedy_act(T, base_cov, ~major)} for T in (4, 10, 25, 74)}
    res['activity Paris'] = {'major_km': float(Lm[major].sum() / 1000), 'total_km': float(Lm.sum() / 1000),
                             'major_segments': int(major.sum()),
                             'long_range_units': {t: units(edges, meta, midx, t, 'any') for t in (10000, 25000, 50000)},
                             'long_range_share': float(lr_share), 'short_units_to_match': need, 'budgets': rows}
    print(f"B activity Paris: major {Lm[major].sum()/1000:.0f} of {Lm.sum()/1000:.0f} km, LR units {res['activity Paris']['long_range_units']}, "
          f"LR-only {lr_share*100:.1f}% (matched by {need} short units); " +
          "; ".join(f"T={T}: distributed {r['distributed']*100:.1f}% hybrid {r['hybrid']*100:.1f}%" for T, r in rows.items()), flush=True)
    # ---- EEW (Mexico City): long-range cables on major roads provide sensing zones >= 5 km apart
    G, edges, meta = load('Mexico City')
    pop = np.maximum(orx.load_population_per_edge(edges, G, 'Mexico City'), 0).astype(float)
    eq = orx.sample_earthquake_locations(edges, pop, orx.EEW_MC_SAMPLES, np.random.RandomState(42))
    mid = np.array([[e['mx'], e['my']] for e in edges]); Lm = np.array([e['length_m'] for e in edges])
    pieces, major = backbone(edges, [x[0] for x in meta]); midx = np.where(major)[0]
    rng = np.random.default_rng(5); pts = []
    for i in rng.permutation(midx):                             # thin to zones at least 5 km apart
        if not pts or np.min(np.hypot(*(mid[pts] - mid[i]).T)) >= orx.EEW_MIN_SPACING:
            pts.append(i)
    from eew_common import H_PRIMARY, VP_PRIMARY
    H, VP = H_PRIMARY, VP_PRIMARY

    def det(xy):
        D = np.sort(np.sqrt(cdist(xy, eq) ** 2 + H ** 2), axis=0)
        return float(D[0].max() / VP), float(D[min(3, len(xy) - 1)].mean() / VP)
    orx.EQ_DEPTH = H
    g6 = mid[orx.optimize_eew(edges, 6, eq, orx.EEW_MIN_SPACING)]
    g10 = mid[orx.optimize_eew(edges, 10, eq, orx.EEW_MIN_SPACING)]
    lr = mid[pts]
    res['eew Mexico City'] = {'major_km': float(Lm[major].sum() / 1000), 'total_km': float(Lm.sum() / 1000),
                              'long_range_units': {t: units(edges, meta, midx, t, 'any') for t in (10000, 25000, 50000)},
                              'sensing_zones': len(pts), 'long_range': det(lr), 'greedy6': det(g6), 'greedy10': det(g10),
                              'hybrid6': det(np.vstack([lr, g6]))}
    r = res['eew Mexico City']
    print(f"B eew Mexico City: major {r['major_km']:.0f} of {r['total_km']:.0f} km, LR units {r['long_range_units']}, zones {len(pts)}; "
          f"worst/mean-4th (s): LR {r['long_range'][0]:.2f}/{r['long_range'][1]:.2f}, greedy6 {r['greedy6'][0]:.2f}/{r['greedy6'][1]:.2f}, "
          f"greedy10 {r['greedy10'][0]:.2f}/{r['greedy10'][1]:.2f}, hybrid {r['hybrid6'][0]:.2f}/{r['hybrid6'][1]:.2f}", flush=True)
    return res


if __name__ == '__main__':
    prev = os.path.join(OUT, 'infra_architectures.json')
    if '--part-b-only' in sys.argv and os.path.exists(prev):
        A = {k: {int(t): v for t, v in r.items()} for k, r in json.load(open(prev))['range_continuity'].items()}
    else:
        A = part_a()
    B = part_b()
    json.dump({'range_continuity': {k: {str(t): v for t, v in r.items()} for k, r in A.items()}, 'backbone_frac': BACKBONE_FRAC,
               'architectures': json.loads(json.dumps(B, default=str))},
              open(os.path.join(OUT, 'infra_architectures.json'), 'w'), indent=1)
    print('saved infra_architectures.json')
