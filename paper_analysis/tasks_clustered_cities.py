"""Street-level versus clustered access in ten cities (SI Fig. S9 replacement).

At a fixed coverage (traffic c = 2%, activity c = 0.1%), the greedy design is run within an
accessible set containing a fraction a of all segments, for a from c to 1:
  street : accessible segments chosen independently at random
  block  : accessible segments are all those inside randomly chosen 2 x 2 km blocks
20 draws per value of a. Output: tasks_clustered_cities.json
"""
import os, sys, json, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks_many_cities import load, CITIES, traffic_greedy, OUT
import optimal_routing as orx

C_TRAFFIC, C_ACTIVITY = 0.02, 0.001
A_GRID = [1, 1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]        # multiples of c, capped at a = 1
BLOCK_L = 2000.0
N_DRAW = 20


def blocks(mid):
    bid = np.floor((mid - mid.min(0)) / BLOCK_L).astype(np.int64)
    key = bid[:, 0] * 100000 + bid[:, 1]
    return np.unique(key, return_inverse=True, return_counts=True)


def access_mask(m, a, model, rng, bk):
    if a >= 1:
        return np.ones(m, bool)
    k = max(1, int(round(a * m)))
    if model == 'street':
        mask = np.zeros(m, bool); mask[rng.choice(m, k, replace=False)] = True
        return mask
    ukeys, inv, counts = bk
    order = rng.permutation(len(ukeys))
    nb = int(np.searchsorted(np.cumsum(counts[order]), k)) + 1
    chosen = np.zeros(len(ukeys), bool); chosen[order[:nb]] = True
    return chosen[inv]


def a_values(c):
    out = []
    for k in A_GRID:
        a = k * c
        if a >= 1:
            out.append(1.0); break
        out.append(a)
    return out


def run_traffic(city):
    rng = np.random.default_rng(23)
    edges, _ = load(city)
    W = np.array([e['weight'] for e in edges], float); Wt = W.sum()
    C = np.ceil(np.array([e['length_m'] for e in edges]) / orx.DAS_SENSING_RANGE).astype(int)
    mid = np.array([[e['mx'], e['my']] for e in edges]); m = len(W); bk = blocks(mid)
    B = int(round(C_TRAFFIC * C.sum()))
    rows = []
    for a in a_values(C_TRAFFIC):
        r = {'a': a}
        for model in ('street', 'block'):
            v = np.array([traffic_greedy(W, C, B, access_mask(m, a, model, rng, bk)) for _ in range(N_DRAW)]) / Wt
            r[model] = [float(v.mean()), float(v.std())]
        rows.append(r)
    print(f"traffic {city:<14} a=c {rows[0]['street'][0]*100:.1f}/{rows[0]['block'][0]*100:.1f}  full {rows[-1]['street'][0]*100:.1f}", flush=True)
    return {'city': city, 'task': 'traffic', 'c': C_TRAFFIC, 'm': m, 'rows': rows}


def run_activity(city):
    rng = np.random.default_rng(29)
    edges, pop = load(city)
    mid = np.array([[e['mx'], e['my']] for e in edges]); m = len(mid); bk = blocks(mid)
    p = np.maximum(pop, 0).astype(float); Pt = p.sum()
    tree = cKDTree(mid); pairs = tree.query_ball_tree(tree, r=orx.ACTIVITY_COVERAGE_RADIUS)
    rows_, cols_ = [], []
    for i, nb in enumerate(pairs):
        rows_.extend([i] * len(nb)); cols_.extend(nb)
    cov = csr_matrix((np.ones(len(rows_), np.float32), (rows_, cols_)), shape=(m, m))
    T = max(1, int(round(C_ACTIVITY * m)))

    def greedy(mask):
        covered = np.zeros(m, bool); mask = mask.copy()
        for _ in range(T):
            g = np.asarray(cov.dot(p * (~covered))).ravel(); g[~mask] = -1
            b = int(np.argmax(g))
            if g[b] <= 0:
                break
            mask[b] = False
            covered |= np.asarray(cov[b].toarray()).ravel().astype(bool)
        return p[covered].sum() / Pt

    rows = []
    for a in a_values(C_ACTIVITY):
        r = {'a': a}
        for model in ('street', 'block'):
            v = np.array([greedy(access_mask(m, a, model, rng, bk)) for _ in range(N_DRAW if a < 1 else 1)])
            r[model] = [float(v.mean()), float(v.std())]
        rows.append(r)
    print(f"activity {city:<14} T={T} a=c {rows[0]['street'][0]*100:.1f}/{rows[0]['block'][0]*100:.1f}  full {rows[-1]['street'][0]*100:.1f}", flush=True)
    return {'city': city, 'task': 'activity', 'c': C_ACTIVITY, 'T': T, 'm': m, 'rows': rows}


def dispatch(job):
    kind, city = job
    return {'traffic': run_traffic, 'activity': run_activity}[kind](city)


if __name__ == '__main__':
    jobs = [('traffic', c) for c in CITIES] + [('activity', c) for c in CITIES]
    with Pool(5) as pool:
        res = pool.map(dispatch, jobs, chunksize=1)
    fn = os.path.join(OUT, 'tasks_clustered_cities.json')
    json.dump(res, open(fn, 'w'), indent=1)
    print('saved', fn)
