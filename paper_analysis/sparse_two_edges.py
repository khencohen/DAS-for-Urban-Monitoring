"""Sparse-coverage regime at the two limits of fiber accessibility.

For each city street graph G (junctions = nodes, street segments = edges) and hop
distance d, compute the coverage c = |instrumented segments| / |E| needed so that a
fraction P of junctions has an instrumented segment within d hops, for:

  random   : uncoordinated access, segments instrumented uniformly at random
             (Monte Carlo over random orderings; direct test of Eq. 2 of the paper)
  designed : full access, greedy set cover (lazy greedy maximum coverage)
  grid     : access points on a regular square lattice (nearest segment to each
             lattice point), a planner-style policy
  block    : spatially clustered uncoordinated access, all segments inside randomly
             chosen 1 km x 1 km blocks

"Segment within d hops of junction v" follows the neighbourhood-volume definition of
Section S6: the nearer endpoint of the segment is at most d-1 hops from v, so d = 1
means a segment incident to v.
"""
import os, sys, json, time, heapq, zlib
import numpy as np, networkx as nx, scipy.sparse as sp
from scipy.sparse import csgraph
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_graph

OUT = os.path.dirname(os.path.abspath(__file__))
CITIES = ['Tel_Aviv', 'Barcelona', 'Manhattan', 'Berlin', 'Paris', 'Amsterdam', 'Rome',
          'Buenos_Aires', 'San_Francisco', 'Tokyo']
D_LIST = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
P_LIST = [0.5, 0.9, 0.99]
N_PERM = 10          # random orderings
N_BLOCK_ORDERS = 5   # random block orderings
BLOCK_L = 1000.0     # m
GRID_SPACINGS = np.geomspace(60, 12000, 48)


def build(G):
    nodes = list(G.nodes()); idx = {u: i for i, u in enumerate(nodes)}
    E = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=np.int64)
    n, m = len(nodes), len(E)
    A = sp.csr_matrix((np.ones(2 * m, np.float32), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n))
    A.data[:] = 1
    xy = np.array([[G.nodes[u]['x'], G.nodes[u]['y']] for u in nodes])
    return nodes, E, A, xy


def quantile_c(first_cov_frac, P):
    """first_cov_frac: per-node coverage fraction at which the node becomes covered."""
    return float(np.quantile(first_cov_frac, P, method='higher'))


def random_edge(E, A, n, m, rng):
    res = {d: {P: [] for P in P_LIST} for d in D_LIST}
    indptr, indices = A.indptr, A.indices
    for _ in range(N_PERM):
        rank = rng.permutation(m).astype(np.int64)
        r0 = np.full(n, m, dtype=np.int64)
        np.minimum.at(r0, E[:, 0], rank); np.minimum.at(r0, E[:, 1], rank)
        f = r0.copy(); rad = 0
        for d in D_LIST:
            while rad < d - 1:
                f = np.minimum(f, np.minimum.reduceat(f[indices], indptr[:-1])); rad += 1
            frac = (f + 1) / m
            for P in P_LIST:
                res[d][P].append(quantile_c(frac, P))
    return {d: {P: (float(np.mean(v)), float(np.std(v))) for P, v in dd.items()} for d, dd in res.items()}


def designed_edge(E, A, n, m):
    res = {}
    I = sp.identity(n, dtype=np.float32, format='csr')
    A1 = (A + I).tocsr(); A1.data[:] = 1
    R = I.copy(); rad = 0
    for d in D_LIST:
        while rad < d - 1:
            R = (R @ A1).tocsr(); R.data[:] = 1; rad += 1
        ip, ix = R.indptr, R.indices
        rows = [ix[ip[i]:ip[i + 1]] for i in range(n)]
        sizes = np.diff(ip)
        heap = [(-(int(sizes[a]) + int(sizes[b])), e) for e, (a, b) in enumerate(E)]
        heapq.heapify(heap)
        covered = np.zeros(n, bool); ncov = 0; picks = 0
        targets = sorted(P_LIST); ti = 0; out = {}
        while ti < len(targets) and heap:
            _, e = heapq.heappop(heap)
            a, b = E[e]
            u = np.union1d(rows[a], rows[b])
            g = int(np.count_nonzero(~covered[u]))
            if heap and g < -heap[0][0]:
                heapq.heappush(heap, (-g, e)); continue
            if g == 0:
                break
            covered[u] = True; ncov += g; picks += 1
            while ti < len(targets) and ncov >= targets[ti] * n:
                out[targets[ti]] = picks / m; ti += 1
        res[d] = out
        del rows
    return res


def hop_dist_from_edges(A, n, E, sel):
    """Hop distance from every node to the nearest endpoint of a selected edge."""
    src = np.unique(E[sel].ravel())
    col = sp.csr_matrix((np.ones(len(src), np.float32), (src, np.zeros(len(src), np.int64))), shape=(n, 1))
    Aug = sp.bmat([[A, col], [col.T, None]], format='csr')
    dist = csgraph.shortest_path(Aug, directed=False, unweighted=True, indices=[n])[0][:n] - 1
    return dist


def coverage_by_d(dist):
    return {d: float(np.mean(dist <= d - 1)) for d in D_LIST}


def crossing_c(cs, covs, P):
    """Smallest coverage c at which the coverage fraction reaches P (log-interpolated)."""
    cs = np.asarray(cs); covs = np.asarray(covs); o = np.argsort(cs); cs, covs = cs[o], covs[o]
    covs = np.maximum.accumulate(covs)
    if covs[-1] < P:
        return float('nan')
    j = int(np.argmax(covs >= P))
    if j == 0:
        return float(cs[0])
    x0, x1, y0, y1 = np.log(cs[j - 1]), np.log(cs[j]), covs[j - 1], covs[j]
    return float(np.exp(x0 + (P - y0) * (x1 - x0) / (y1 - y0)))


def grid_edge(E, A, xy, n, m):
    mid = (xy[E[:, 0]] + xy[E[:, 1]]) / 2
    from scipy.spatial import cKDTree
    tree = cKDTree(mid)
    lo, hi = mid.min(0), mid.max(0)
    rows = []
    for s in GRID_SPACINGS:
        gx, gy = np.meshgrid(np.arange(lo[0] + s / 2, hi[0], s), np.arange(lo[1] + s / 2, hi[1], s))
        pts = np.c_[gx.ravel(), gy.ravel()]
        dd, ii = tree.query(pts)
        sel = np.unique(ii[dd <= s / 2])          # lattice points that fall on the street network
        if len(sel) == 0:
            continue
        rows.append((len(sel) / m, coverage_by_d(hop_dist_from_edges(A, n, E, sel))))
    return {d: {P: crossing_c([r[0] for r in rows], [r[1][d] for r in rows], P) for P in P_LIST} for d in D_LIST}


def block_edge(E, A, xy, n, m, rng):
    mid = (xy[E[:, 0]] + xy[E[:, 1]]) / 2
    bid = np.floor((mid - mid.min(0)) / BLOCK_L).astype(np.int64)
    key = bid[:, 0] * 100000 + bid[:, 1]
    ukeys, inv = np.unique(key, return_inverse=True)
    out = {d: {P: [] for P in P_LIST} for d in D_LIST}
    for _ in range(N_BLOCK_ORDERS):
        order = rng.permutation(len(ukeys))
        pos = np.empty(len(ukeys), np.int64); pos[order] = np.arange(len(ukeys))
        edge_rank = pos[inv]                        # block position of each edge
        cnt = np.bincount(edge_rank, minlength=len(ukeys)).cumsum()
        ks = np.unique(np.clip(np.round(np.geomspace(1, len(ukeys), 40)).astype(int), 1, len(ukeys)))
        rows = []
        for k in ks:
            sel = np.where(edge_rank < k)[0]
            rows.append((cnt[k - 1] / m, coverage_by_d(hop_dist_from_edges(A, n, E, sel))))
        for d in D_LIST:
            for P in P_LIST:
                out[d][P].append(crossing_c([r[0] for r in rows], [r[1][d] for r in rows], P))
    return {d: {P: (float(np.nanmean(v)), float(np.nanstd(v))) for P, v in dd.items()} for d, dd in out.items()}


def alpha_fit(G, rng, ns=300):
    """Edge-count neighbourhood volume V(d) ~ alpha d^2 (same estimator as vd_alpha.py)."""
    nodes = list(G.nodes()); DMAX = 10
    Ve = np.zeros(DMAX + 2)
    for si in rng.choice(len(nodes), ns, replace=False):
        dist = nx.single_source_shortest_path_length(G, nodes[si], cutoff=DMAX)
        for u, du in dist.items():
            for w in G[u]:
                dw = dist.get(w)
                if dw is None or dw > du or (dw == du and u < w):
                    Ve[du + 1] += 1
    Ve = np.cumsum(Ve) / ns
    d = np.arange(2, 9)
    return float(np.sum(Ve[d] * d ** 2) / np.sum(d ** 4))


def run_city(name):
    t0 = time.time(); rng = np.random.default_rng(zlib.crc32(name.encode()))
    G = load_graph(name)
    nodes, E, A, xy = build(G); n, m = len(nodes), len(E)
    r = {'n': n, 'm': m, 'k': 2 * m / n, 'alpha': alpha_fit(G, rng)}
    r['random'] = random_edge(E, A, n, m, rng)
    r['designed'] = designed_edge(E, A, n, m)
    r['grid'] = grid_edge(E, A, xy, n, m)
    r['block'] = block_edge(E, A, xy, n, m, rng)
    print(f"{name:<14} n={n:>6} alpha={r['alpha']:.2f}  d=5: random {r['random'][5][0.9][0]:.4f} "
          f"designed {r['designed'][5].get(0.9, float('nan')):.4f} grid {r['grid'][5][0.9]:.4f} "
          f"block {r['block'][5][0.9][0]:.4f}   [{time.time() - t0:.0f}s]", flush=True)
    return name, r


if __name__ == '__main__':
    with Pool(5) as pool:
        results = dict(pool.map(run_city, CITIES))
    def jsonable(o):
        if isinstance(o, dict):
            return {str(k): jsonable(v) for k, v in o.items()}
        if isinstance(o, tuple):
            return [jsonable(v) for v in o]
        return o
    json.dump(jsonable(results), open(os.path.join(OUT, 'sparse_two_edges.json'), 'w'), indent=1)
    print('saved sparse_two_edges.json')
