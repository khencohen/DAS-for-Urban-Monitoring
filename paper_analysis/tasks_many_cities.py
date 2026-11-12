"""Traffic and activity monitoring versus coverage, averaged over the ten cities.

For each city of Fig. S? (the ten cities of the percolation analysis) and each
coverage ratio c (fraction of street segments instrumented):

  random    : uniformly random instrumented segments (100 draws)
  partial   : a fraction F_DESIGN (1%) of the streets is designed with full access and
              the remaining coverage is placed at random (20 draws)
  designed  : greedy design with full access

Traffic: budget B = round(c * sum of interrogator costs), knapsack greedy as in the
paper; value = share of the city's traffic volume captured.
Activity: T = round(c * m) units of 600 m radius, greedy maximum coverage; value =
share of the population within range.

Output: tasks_many_cities.json
"""
import os, sys, json, pickle, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('WORLDPOP_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'population_data'))
import optimal_routing as orx

OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'city_cache')
CITIES = ['Tel Aviv', 'Barcelona', 'Manhattan', 'Berlin', 'Paris', 'Amsterdam', 'Rome',
          'Buenos Aires', 'San Francisco', 'Tokyo']
F_DESIGN = 0.01                              # partial: this fraction of the streets is designed, the rest random
C_GRID = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
C_TRAFFIC = C_GRID
C_ACTIVITY = C_GRID
N_RANDOM, N_PARTIAL = 100, 20


def load(city):
    G = pickle.load(open(os.path.join(CACHE, city.replace(' ', '_') + '.pkl'), 'rb'))
    edges = orx.extract_edges(G)
    pop = orx.load_population_per_edge(edges, G, city)
    return edges, pop


def street_mask(m, a, rng):
    if a >= 1:
        return np.ones(m, bool)
    mask = np.zeros(m, bool)
    mask[rng.choice(m, max(1, int(round(a * m))), replace=False)] = True
    return mask


# ---------------------------------------------------------------- traffic
def traffic_greedy(W, C, B, mask, return_sel=False):
    eff = W / np.maximum(C, 1)
    idx = np.where(mask)[0]
    idx = idx[np.argsort(-eff[idx], kind='stable')]
    used = 0; val = 0.0; sel = []
    for i in idx:
        if used + C[i] <= B:
            used += C[i]; val += W[i]; sel.append(i)
            if used >= B:
                break
    return (val, used, sel) if return_sel else val


def traffic_partial(W, C, B, B1, rng):
    """Design a budget B1, then add random segments until the budget B is used."""
    val, used, sel = traffic_greedy(W, C, B1, np.ones(len(W), bool), return_sel=True)
    taken = np.zeros(len(W), bool); taken[sel] = True
    for i in rng.permutation(len(W)):
        if taken[i]:
            continue
        if used + C[i] <= B:
            used += C[i]; val += W[i]; taken[i] = True
        if used >= B:
            break
    return val


def traffic_random(W, C, B, rng):
    used = 0; val = 0.0; n = 0
    for i in rng.permutation(len(W)):
        if used + C[i] <= B:
            used += C[i]; val += W[i]; n += 1
        if used >= B:
            break
    return val, n


def run_traffic(city):
    rng = np.random.default_rng(11)
    edges, _ = load(city)
    W = np.array([e['weight'] for e in edges], float); Wt = W.sum()
    C = np.ceil(np.array([e['length_m'] for e in edges]) / orx.DAS_SENSING_RANGE).astype(int)
    m = len(W); Ct = C.sum()
    rows = []
    for c in C_TRAFFIC:
        B = int(round(c * Ct))
        rnd = [traffic_random(W, C, B, rng) for _ in range(N_RANDOM)]
        rv = np.array([r[0] for r in rnd]) / Wt
        c_eff = np.mean([r[1] for r in rnd]) / m
        full = traffic_greedy(W, C, B, np.ones(m, bool)) / Wt
        B1 = min(B, int(round(F_DESIGN * Ct)))
        part = np.array([traffic_partial(W, C, B, B1, rng) for _ in range(N_PARTIAL)]) / Wt
        rows.append({'c': c, 'c_eff': float(c_eff), 'B': B,
                     'random': [float(rv.mean()), float(rv.std())],
                     'partial': [float(part.mean()), float(part.std())],
                     'designed': float(full)})
        print(f"traffic {city:<14} c={c*100:5.1f}%  random {rv.mean()*100:5.1f}  partial {part.mean()*100:5.1f}  designed {full*100:5.1f}", flush=True)
    return {'city': city, 'task': 'traffic', 'm': m, 'rows': rows}


# ---------------------------------------------------------------- activity
def run_activity(city):
    rng = np.random.default_rng(17)
    edges, pop = load(city)
    if pop is None:
        print('no population raster for', city); return None
    mid = np.array([[e['mx'], e['my']] for e in edges]); m = len(mid)
    p = np.maximum(pop, 0).astype(float); Pt = p.sum()
    tree = cKDTree(mid); pairs = tree.query_ball_tree(tree, r=orx.ACTIVITY_COVERAGE_RADIUS)
    rows_, cols_ = [], []
    for i, nb in enumerate(pairs):
        rows_.extend([i] * len(nb)); cols_.extend(nb)
    cov = csr_matrix((np.ones(len(rows_), np.float32), (rows_, cols_)), shape=(m, m))

    def greedy(mask, T, return_sel=False):
        covered = np.zeros(m, bool); sel = []
        mask = mask.copy()
        for _ in range(T):
            g = np.asarray(cov.dot(p * (~covered))).ravel(); g[~mask] = -1
            b = int(np.argmax(g))
            if g[b] <= 0:
                break
            mask[b] = False; sel.append(b)
            covered |= np.asarray(cov[b].toarray()).ravel().astype(bool)
        return (p[covered].sum() / Pt, sel) if return_sel else p[covered].sum() / Pt

    def partial(T, T1, rng):
        """T1 designed units, the remaining T - T1 at random."""
        _, sel = greedy(np.ones(m, bool), T1, return_sel=True)
        rest = np.setdiff1d(np.arange(m), sel)
        extra = rng.choice(rest, T - len(sel), replace=False) if T > len(sel) else []
        return covered_frac(np.r_[sel, extra].astype(int))

    def covered_frac(sel):
        cv = np.asarray(cov[sel].sum(0)).ravel() > 0
        return p[cv].sum() / Pt

    rows = []
    for c in C_ACTIVITY:
        T = max(1, int(round(c * m)))
        rv = np.array([covered_frac(rng.choice(m, T, replace=False)) for _ in range(N_RANDOM)])
        full = greedy(np.ones(m, bool), T)
        T1 = min(T, max(1, int(round(F_DESIGN * m))))
        part = np.array([partial(T, T1, rng) for _ in range(N_PARTIAL if T > T1 else 1)])
        rows.append({'c': c, 'T': T,
                     'random': [float(rv.mean()), float(rv.std())],
                     'partial': [float(part.mean()), float(part.std())],
                     'designed': float(full)})
        print(f"activity {city:<14} c={c*100:6.2f}% T={T:4d}  random {rv.mean()*100:5.1f}  partial {part.mean()*100:5.1f}  designed {full*100:5.1f}", flush=True)
    return {'city': city, 'task': 'activity', 'm': m, 'rows': rows}


def dispatch(job):
    kind, city = job
    t = time.time()
    r = {'traffic': run_traffic, 'activity': run_activity}[kind](city)
    if r is not None:
        r['seconds'] = time.time() - t
    return r


if __name__ == '__main__':
    jobs = [('traffic', c) for c in CITIES] + [('activity', c) for c in CITIES]
    with Pool(5) as pool:
        res = [r for r in pool.map(dispatch, jobs, chunksize=1) if r is not None]
    fn = os.path.join(OUT, 'tasks_many_cities.json')
    json.dump(res, open(fn, 'w'), indent=1)
    print('saved', fn)
