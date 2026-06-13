"""
Optimal DAS Fiber-Optic Placement: Three Problems
====================================================================
Problem 1 (Traffic):  max Σ C_ij · A_ij   — knapsack-greedy by traffic/DAS-cost
Problem 2 (EEW):      min max_q[min_i d(s_i,q)] — minimax detection distance
Problem 3 (Activity): max population within coverage radius

Budget modes:
  - Edge count / DAS count
  - km of fiber: accumulated physical fiber length
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.colors as mcolors
import networkx as nx
import osmnx as ox
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist
import os
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
except ImportError:
    rasterio = None

# ================================================================
# Configuration
# ================================================================

CITIES = {
    'Tel Aviv':      'Tel Aviv, Israel',
    'Barcelona':     'Barcelona, Spain',
    'San Francisco': 'San Francisco, California, USA',
    'Amsterdam':     'Amsterdam, Netherlands',
    'Rome':          'Rome, Italy',
    'Manhattan':     'Manhattan, New York City, New York, USA',
    'Buenos Aires':  'Buenos Aires, Argentina',
    'Cape Town':     'City of Cape Town, South Africa',
    'Sydney':        'Sydney, New South Wales, Australia',
    'Tokyo':         'Shibuya, Tokyo, Japan',
    'Paris':         'Paris, France',
    'London':        'City of Westminster, London, United Kingdom',
    'Berlin':        'Berlin, Germany',
    'Singapore':     'Singapore',
    'Delhi':         'New Delhi, India',
    'Nairobi':       'Nairobi, Kenya',
    'Toronto':       'Old Toronto, Toronto, Ontario, Canada',
    'Seoul':         'Gangnam-gu, Seoul, South Korea',
    'Mexico City':   'Cuauhtémoc, Mexico City, Mexico',
    'Istanbul':      'Beyoğlu, Istanbul, Turkey',
}

TRAFFIC_WEIGHT = {
    'motorway': 2000, 'trunk': 1500, 'primary': 1000,
    'secondary': 500, 'tertiary': 250, 'unclassified': 100,
    'residential': 50, 'living_street': 20,
    'motorway_link': 1000, 'trunk_link': 750,
    'primary_link': 500, 'secondary_link': 300,
    'tertiary_link': 150, 'road': 50, 'service': 10,
}

T_FRACTIONS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]

TRAFFIC_EDGE_FRACS = [0.001, 0.01, 0.05, 0.10]
TRAFFIC_KM_BUDGETS = [1, 5, 10, 50, 100, 500]
EEW_T_VALUES = [4, 6, 8, 10, 12, 14, 16, 18, 20]
EEW_KM_BUDGETS = [1, 5, 10, 50]
EEW_MC_SAMPLES = 300
EEW_MIN_SPACING = 5_000          # 5 km minimum distance between DAS
ACTIVITY_T_VALUES = [4, 6, 8, 10]
ACTIVITY_KM_BUDGETS = [1, 5, 10, 50]
ACTIVITY_COVERAGE_RADIUS = 600    # 600 m DAS coverage radius
DAS_SENSING_RANGE = 500              # meters — max distance a single DAS can support
EQ_DEPTH = 20_000                    # meters — assumed earthquake hypocentre depth
V_P = 7_000                          # m/s — P-wave velocity
V_S = 4_000                          # m/s — S-wave velocity

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'city_cache')
STRATEGY_COLORS = {'traffic': '#D32F2F', 'eew': '#1565C0',
                   'hull': '#2E7D32', 'hull_pop': '#6A1B9A', 'uniform': '#1565C0'}
STRATEGY_LABELS = {'traffic': 'Traffic-optimal', 'eew': 'EEW-optimal',
                   'hull': 'Hull-optimal (area)',
                   'hull_pop': 'Hull-optimal (population)', 'uniform': 'Uniform random'}

CITY_POP_RASTER = {
    'Tel Aviv': 'isr', 'Barcelona': 'esp', 'San Francisco': 'usa',
    'Amsterdam': 'nld', 'Rome': 'ita', 'Manhattan': 'usa',
    'Buenos Aires': 'arg', 'Cape Town': 'zaf', 'Sydney': 'aus',
    'Tokyo': 'jpn', 'Paris': 'fra', 'London': 'gbr',
    'Berlin': 'deu', 'Singapore': 'sgp', 'Delhi': 'ind',
    'Nairobi': 'ken', 'Toronto': 'can', 'Seoul': 'kor',
    'Mexico City': 'mex', 'Istanbul': 'tur',
}

# ================================================================
# 1. City Loading
# ================================================================

def load_city(name, query):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f'{name.replace(" ", "_")}.pkl')

    if os.path.exists(cache_path):
        print(f"    [cache] {name}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"    [download] {name} ...")
    G = ox.graph_from_place(query, network_type='drive')
    G_proj = ox.project_graph(G)

    with open(cache_path, 'wb') as f:
        pickle.dump(G_proj, f)
    return G_proj


# ================================================================
# 2. Edge Data Extraction
# ================================================================

def extract_edges(G_proj):
    """Deduplicate to undirected; store midpoints, traffic weight, geometry."""
    seen = set()
    edges = []

    for u, v, _key, data in G_proj.edges(keys=True, data=True):
        ek = (min(u, v), max(u, v))
        if ek in seen:
            continue
        seen.add(ek)

        xu, yu = G_proj.nodes[u]['x'], G_proj.nodes[u]['y']
        xv, yv = G_proj.nodes[v]['x'], G_proj.nodes[v]['y']

        hw = data.get('highway', 'residential')
        if isinstance(hw, list):
            hw = hw[0]

        if 'geometry' in data:
            xs, ys = data['geometry'].xy
            line_coords = list(zip(xs, ys))
        else:
            line_coords = [(xu, yu), (xv, yv)]

        edges.append({
            'u': u, 'v': v,
            'mx': (xu + xv) / 2, 'my': (yu + yv) / 2,
            'weight': TRAFFIC_WEIGHT.get(hw, 50),
            'length_m': data.get('length', 0.0),
            'line': line_coords,
        })
    return edges


def get_base_lines(G_proj):
    """Line segments for background street plot (fast LineCollection)."""
    lines = []
    for u, v, data in G_proj.edges(data=True):
        if 'geometry' in data:
            xs, ys = data['geometry'].xy
            lines.append(list(zip(xs, ys)))
        else:
            lines.append([
                (G_proj.nodes[u]['x'], G_proj.nodes[u]['y']),
                (G_proj.nodes[v]['x'], G_proj.nodes[v]['y']),
            ])
    return lines


# ================================================================
# 2b. Population Data
# ================================================================

def _find_pop_dir():
    """Locate the WorldPop raster directory.

    Resolution order:
      1. $WORLDPOP_DIR environment variable, if it points to a directory.
      2. A local ``population_data/`` folder next to this script.
    Returns None if neither is found (EEW and Activity tasks are then skipped).
    See the README for download instructions.
    """
    env = os.environ.get('WORLDPOP_DIR')
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'population_data')
    if os.path.isdir(local):
        return os.path.abspath(local)
    return None


def load_population_per_edge(edges, G_proj, city_name):
    """Sample WorldPop 1km raster at each edge midpoint."""
    if rasterio is None:
        return None
    code = CITY_POP_RASTER.get(city_name)
    if code is None:
        return None
    pop_dir = _find_pop_dir()
    if pop_dir is None:
        return None
    tif_path = os.path.join(pop_dir, f'{code}_ppp_2020_1km_Aggregated_UNadj.tif')
    if not os.path.exists(tif_path):
        return None

    from pyproj import Transformer
    crs_proj = G_proj.graph['crs']
    transformer = Transformer.from_crs(crs_proj, 'EPSG:4326', always_xy=True)

    mx = np.array([e['mx'] for e in edges])
    my = np.array([e['my'] for e in edges])
    lons, lats = transformer.transform(mx, my)

    with rasterio.open(tif_path) as src:
        pop = np.array([v[0] for v in src.sample(zip(lons, lats))])
    pop = np.where(np.isnan(pop) | (pop < 0), 0.0, pop)
    return pop


# ================================================================
# 2c. Earthquake Location Sampling
# ================================================================

def sample_earthquake_locations(edges, pop, n_samples, rng, jitter_sigma=3000):
    """Sample earthquake epicentres from population-weighted distribution.

    Inside the city: probability proportional to WorldPop density at edge midpoints.
    The Gaussian jitter naturally places some samples outside the city boundary,
    providing the fast-decaying margin.
    """
    midpoints = np.array([[e['mx'], e['my']] for e in edges])
    p = np.maximum(pop, 0).astype(float)
    p_sum = p.sum()
    if p_sum > 0:
        p /= p_sum
    else:
        p = np.ones(len(p)) / len(p)

    edge_idx = rng.choice(len(edges), size=n_samples, replace=True, p=p)
    jitter = rng.normal(0, jitter_sigma, size=(n_samples, 2))
    return midpoints[edge_idx] + jitter


# ================================================================
# 2d. Budget Helpers
# ================================================================

def select_by_km_budget(edges, ranking, km_budget):
    """Select edges in ranking order until accumulated length ≥ km_budget (km)."""
    budget_m = km_budget * 1000.0
    selected = []
    cum = 0.0
    for idx in ranking:
        l = edges[idx]['length_m']
        if cum + l > budget_m and len(selected) > 0:
            break
        selected.append(idx)
        cum += l
    return selected


def optimize_traffic_km(edges, km_budget):
    """Knapsack-style traffic-optimal selection: rank by value per DAS cost."""
    lengths = np.array([e['length_m'] for e in edges])
    weights = np.array([e['weight'] for e in edges])
    das_cost = np.ceil(lengths / DAS_SENSING_RANGE)
    efficiency = weights / np.maximum(das_cost, 1)
    ranking = np.argsort(-efficiency)
    return select_by_km_budget(edges, ranking, km_budget)


# ================================================================
# 3. Optimization Algorithms
# ================================================================

def optimize_traffic(edges, T):
    """Knapsack-greedy: select T edges with highest traffic/DAS-cost ratio."""
    lengths = np.array([e['length_m'] for e in edges])
    weights = np.array([e['weight'] for e in edges])
    das_cost = np.ceil(lengths / DAS_SENSING_RANGE)
    efficiency = weights / np.maximum(das_cost, 1)
    return np.argsort(-efficiency)[:T].tolist()


def optimize_hull(edges, T):
    """Select hull-vertex edges first to maximize convex-hull area."""
    midpoints = np.array([[e['mx'], e['my']] for e in edges])
    n = len(midpoints)
    if T >= n:
        return list(range(n))
    if n < 3 or T < 3:
        # Fallback: farthest-pair heuristic
        from scipy.spatial.distance import cdist
        D = cdist(midpoints, midpoints)
        pair = np.unravel_index(D.argmax(), D.shape)
        return list(pair)[:T]

    full_hull = ConvexHull(midpoints)
    hv = list(full_hull.vertices)

    if T >= len(hv):
        selected = set(hv)
        remaining = [i for i in range(n) if i not in selected]
        np.random.seed(42)
        fill = min(T - len(selected), len(remaining))
        if fill > 0:
            selected.update(np.random.choice(remaining, fill, replace=False))
        return sorted(selected)[:T]

    # T < |hull vertices|: greedy subset that maximises sub-hull area
    from scipy.spatial.distance import cdist
    hv_pts = midpoints[hv]
    D = cdist(hv_pts, hv_pts)
    i0, j0 = np.unravel_index(D.argmax(), D.shape)
    local_sel = [i0, j0]
    local_rem = set(range(len(hv))) - {i0, j0}

    for _ in range(T - 2):
        if not local_rem:
            break
        best_idx, best_area = None, -1
        cur = hv_pts[local_sel]
        for idx in local_rem:
            cand = np.vstack([cur, hv_pts[idx]])
            try:
                a = ConvexHull(cand).volume
                if a > best_area:
                    best_area, best_idx = a, idx
            except Exception:
                continue
        if best_idx is not None:
            local_sel.append(best_idx)
            local_rem.remove(best_idx)
        else:
            break
    return [hv[i] for i in local_sel]


def optimize_uniform(edges, T, seed=42):
    """Uniform random edge sampling."""
    rng = np.random.RandomState(seed)
    return rng.choice(len(edges), min(T, len(edges)), replace=False).tolist()


def _pop_inside_hull(all_midpoints, pop, hull_pts):
    """Sum population of edges whose midpoints fall inside the convex hull of hull_pts."""
    if len(hull_pts) < 3:
        return 0.0
    try:
        tri = Delaunay(hull_pts)
        inside = tri.find_simplex(all_midpoints) >= 0
        return float(pop[inside].sum())
    except Exception:
        return 0.0


def optimize_hull_pop(edges, T, pop):
    """Select edges to maximize total population inside the convex hull."""
    midpoints = np.array([[e['mx'], e['my']] for e in edges])
    n = len(midpoints)
    if T >= n:
        return list(range(n))
    if n < 3 or T < 3:
        return np.argsort(-pop)[:T].tolist()

    full_hull = ConvexHull(midpoints)
    hv = list(full_hull.vertices)

    if T >= len(hv):
        selected = set(hv)
        remaining = [i for i in range(n) if i not in selected]
        rem_pop = sorted(remaining, key=lambda i: -pop[i])
        fill = min(T - len(selected), len(remaining))
        selected.update(rem_pop[:fill])
        return sorted(selected)[:T]

    from scipy.spatial.distance import cdist
    hv_pts = midpoints[hv]
    D = cdist(hv_pts, hv_pts)
    i0, j0 = np.unravel_index(D.argmax(), D.shape)
    local_sel = [i0, j0]
    local_rem = set(range(len(hv))) - {i0, j0}

    for _ in range(T - 2):
        if not local_rem:
            break
        best_idx, best_pop_val = None, -1
        cur = hv_pts[local_sel]
        for idx in local_rem:
            cand = np.vstack([cur, hv_pts[idx]])
            try:
                v = _pop_inside_hull(midpoints, pop, cand)
                if v > best_pop_val:
                    best_pop_val, best_idx = v, idx
            except Exception:
                continue
        if best_idx is not None:
            local_sel.append(best_idx)
            local_rem.remove(best_idx)
        else:
            break
    return [hv[i] for i in local_sel]


# ================================================================
# 3b. EEW Optimization
# ================================================================

def _greedy_eew_step(D_full, midpoints, selected, valid_mask, min_spacing):
    """Vectorised greedy step: pick candidate minimising max detection distance.

    For each earthquake q, detection distance = min distance to any DAS.
    Objective: minimise the maximum of this over all earthquakes.
    Greedy criterion uses mean for stability, tiebreaks by max.
    """
    D_sel = D_full[selected]                  # (k, Q)
    current_min = D_sel.min(axis=0)           # (Q,) nearest DAS per quake

    sel_pts = midpoints[selected]
    dists_to_sel = cdist(midpoints, sel_pts)  # (N, k)
    min_dists = dists_to_sel.min(axis=1)      # (N,)
    spacing_ok = min_dists >= min_spacing

    mask = valid_mask & spacing_ok
    mask[selected] = False
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        return None

    cand_d = D_full[candidates]                                    # (C, Q)
    new_min = np.minimum(current_min[np.newaxis, :], cand_d)       # (C, Q)
    scores = new_min.max(axis=1)                                   # (C,)

    best = int(candidates[np.argmin(scores)])
    return best


def optimize_eew(edges, T, eq_locations, min_spacing=EEW_MIN_SPACING):
    """Greedy selection of T DAS edges (vectorised).

    Objective:  minimise max_q [ min_i d(s_i, q) ]  — worst-case detection
    Constraint: every pair of selected DAS must be ≥ min_spacing apart
    """
    midpoints = np.array([[e['mx'], e['my']] for e in edges])
    D_full = np.sqrt(cdist(midpoints, eq_locations) ** 2 + EQ_DEPTH ** 2)

    eq_centroid = eq_locations.mean(axis=0)
    dist_to_centroid = np.sqrt(((midpoints - eq_centroid) ** 2).sum(axis=1))
    first = int(np.argmin(dist_to_centroid))

    selected = [first]
    valid_mask = np.ones(len(edges), dtype=bool)

    for _ in range(T - 1):
        best = _greedy_eew_step(D_full, midpoints, selected, valid_mask, min_spacing)
        if best is None:
            break
        selected.append(best)
    return selected


def optimize_eew_km(edges, km_budget, eq_locations, min_spacing=EEW_MIN_SPACING):
    """EEW-optimal greedy selection under a km-of-fiber budget (vectorised)."""
    midpoints = np.array([[e['mx'], e['my']] for e in edges])
    D_full = np.sqrt(cdist(midpoints, eq_locations) ** 2 + EQ_DEPTH ** 2)
    lengths = np.array([e['length_m'] for e in edges])

    budget_m = km_budget * 1000.0

    eq_centroid = eq_locations.mean(axis=0)
    dist_to_centroid = np.sqrt(((midpoints - eq_centroid) ** 2).sum(axis=1))
    first = int(np.argmin(dist_to_centroid))

    selected = [first]
    cum = lengths[first]

    while cum < budget_m:
        valid_mask = (lengths <= budget_m - cum) | (np.arange(len(edges)) == first)
        valid_mask[selected] = False
        best = _greedy_eew_step(D_full, midpoints, selected, valid_mask, min_spacing)
        if best is None:
            break
        selected.append(best)
        cum += lengths[best]
    return selected


def eval_eew(edges, selected, eq_locations):
    """Max detection distance: max over earthquakes of min-dist to nearest DAS. Lower = better."""
    if not selected:
        return np.inf
    midpoints = np.array([[edges[i]['mx'], edges[i]['my']] for i in selected])
    D = np.sqrt(cdist(midpoints, eq_locations) ** 2 + EQ_DEPTH ** 2)
    nearest = D.min(axis=0)                  # (Q,) nearest DAS per quake
    return float(nearest.max())              # worst-case 3D detection distance


# ================================================================
# 3c. Activity Monitoring Optimization
# ================================================================

def optimize_activity(edges, T, pop):
    """Greedy max-coverage: select T DAS edges maximising population within coverage radius."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix

    midpoints = np.array([[e['mx'], e['my']] for e in edges])
    pop_safe = np.maximum(pop, 0).astype(float)
    n = len(edges)

    tree = cKDTree(midpoints)
    pairs = tree.query_ball_tree(tree, r=ACTIVITY_COVERAGE_RADIUS)

    rows, cols = [], []
    for i, neighbors in enumerate(pairs):
        rows.extend([i] * len(neighbors))
        cols.extend(neighbors)
    cov = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))

    covered = np.zeros(n, dtype=bool)
    selected = []

    for _ in range(T):
        uncov_pop = pop_safe * (~covered)
        gains = np.asarray(cov.dot(uncov_pop)).ravel()
        gains[selected] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        selected.append(best)
        newly = np.asarray(cov[best].toarray()).ravel().astype(bool)
        covered |= newly

    return selected


def optimize_activity_km(edges, km_budget, pop):
    """Activity monitoring under km-of-fiber budget."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix

    midpoints = np.array([[e['mx'], e['my']] for e in edges])
    pop_safe = np.maximum(pop, 0).astype(float)
    lengths = np.array([e['length_m'] for e in edges])
    n = len(edges)
    budget_m = km_budget * 1000.0

    tree = cKDTree(midpoints)
    pairs = tree.query_ball_tree(tree, r=ACTIVITY_COVERAGE_RADIUS)

    rows, cols = [], []
    for i, neighbors in enumerate(pairs):
        rows.extend([i] * len(neighbors))
        cols.extend(neighbors)
    cov = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))

    covered = np.zeros(n, dtype=bool)
    selected = []
    cum = 0.0

    while cum < budget_m:
        uncov_pop = pop_safe * (~covered)
        gains = np.asarray(cov.dot(uncov_pop)).ravel()
        gains[selected] = -1
        over_budget = (lengths > budget_m - cum)
        gains[over_budget] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        selected.append(best)
        cum += lengths[best]
        newly = np.asarray(cov[best].toarray()).ravel().astype(bool)
        covered |= newly

    return selected


def eval_activity(edges, selected, pop):
    """Total population covered within ACTIVITY_COVERAGE_RADIUS of any selected DAS."""
    if not selected or pop is None:
        return 0.0
    from scipy.spatial import cKDTree
    midpoints_all = np.array([[e['mx'], e['my']] for e in edges])
    midpoints_sel = midpoints_all[selected]
    pop_safe = np.maximum(pop, 0).astype(float)

    tree = cKDTree(midpoints_all)
    covered_idx = tree.query_ball_point(midpoints_sel, r=ACTIVITY_COVERAGE_RADIUS)
    covered_set = set()
    for lst in covered_idx:
        covered_set.update(lst)
    return float(sum(pop_safe[i] for i in covered_set))


# ================================================================
# 3d. EEW Alert-Time Heatmap
# ================================================================

def compute_alert_time_grid(edges, selected, grid_res=200, pad_frac=0.35):
    """EEW excess detection delay: 100 × (t_P − t_P^min) / t_P^min  (%).

    t_P     = d_4 / v_P  — actual P-wave travel time to 4th nearest DAS.
    t_P^min = h  / v_P   — minimum (DAS directly above epicentre).
    Excess  = 100 × (d_4 − h) / h.   Best case = 0%.
    """
    midpoints_sel = np.array([[edges[i]['mx'], edges[i]['my']] for i in selected])
    all_mx = np.array([e['mx'] for e in edges])
    all_my = np.array([e['my'] for e in edges])
    dx = all_mx.max() - all_mx.min()
    dy = all_my.max() - all_my.min()
    pad = max(dx, dy) * pad_frac
    xmin, xmax = all_mx.min() - pad, all_mx.max() + pad
    ymin, ymax = all_my.min() - pad, all_my.max() + pad

    xi = np.linspace(xmin, xmax, grid_res)
    yi = np.linspace(ymin, ymax, grid_res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    grid_pts = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])

    horiz_dists = cdist(grid_pts, midpoints_sel)
    dists_3d = np.sqrt(horiz_dists ** 2 + EQ_DEPTH ** 2)

    k = min(4, len(selected))
    d_k = np.partition(dists_3d, k - 1, axis=1)[:, k - 1]

    h = float(EQ_DEPTH)
    excess_pct = 100.0 * (d_k - h) / h

    return excess_pct.reshape(grid_res, grid_res), (xmin, xmax, ymin, ymax)


# ================================================================
# 4. Metric Evaluation
# ================================================================

def eval_traffic(edges, selected):
    return sum(edges[i]['weight'] for i in selected)


def eval_hull_area(edges, selected):
    if len(selected) < 3:
        return 0.0
    pts = np.array([[edges[i]['mx'], edges[i]['my']] for i in selected])
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def eval_hull_population(edges, selected, pop):
    """Total population of ALL edges inside the convex hull of selected edges."""
    if len(selected) < 3:
        return 0.0
    midpoints = np.array([[e['mx'], e['my']] for e in edges])
    hull_pts = np.array([[edges[i]['mx'], edges[i]['my']] for i in selected])
    return _pop_inside_hull(midpoints, pop, hull_pts)


def max_hull_population(edges, pop):
    """Population inside the full convex hull of all edges."""
    midpoints = np.array([[e['mx'], e['my']] for e in edges])
    try:
        hull = ConvexHull(midpoints)
        hull_pts = midpoints[hull.vertices]
        return _pop_inside_hull(midpoints, pop, hull_pts)
    except Exception:
        return float(pop.sum())


def eval_uniformity_vmr(edges, selected, n_grid=20):
    """Variance-to-Mean Ratio (Index of Dispersion). VMR≈1 = Poisson; VMR>>1 = clustered."""
    if len(selected) < 2:
        return 0.0
    all_mx = np.array([e['mx'] for e in edges])
    all_my = np.array([e['my'] for e in edges])
    sel_mx = np.array([edges[i]['mx'] for i in selected])
    sel_my = np.array([edges[i]['my'] for i in selected])

    xb = np.linspace(all_mx.min(), all_mx.max(), n_grid + 1)
    yb = np.linspace(all_my.min(), all_my.max(), n_grid + 1)

    counts, _, _ = np.histogram2d(sel_mx, sel_my, bins=[xb, yb])
    mean_c = counts.mean()
    if mean_c < 0.01:
        return 0.0
    return float(counts.var() / mean_c)


def max_traffic(edges):
    return sum(e['weight'] for e in edges)


def max_hull_area(edges):
    pts = np.array([[e['mx'], e['my']] for e in edges])
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 1.0


# ================================================================
# 5. Analysis
# ================================================================

def analyze_city(name, G_proj, T_fractions=None):
    """Run traffic + EEW optimisation with both budget modes."""
    edges = extract_edges(G_proj)
    base_lines = get_base_lines(G_proj)
    n_edges = len(edges)
    total_km = sum(e['length_m'] for e in edges) / 1000.0
    mt = max_traffic(edges)

    pop = load_population_per_edge(edges, G_proj, name)

    print(f"  {name}: {n_edges} edges, {total_km:.1f} km network, max traffic={mt:,.0f}")

    results = {
        'n_edges': n_edges,
        'total_km': total_km,
        'max_traffic': mt,
        'pop_per_edge': pop,
        'edges': edges,
        'base_lines': base_lines,
    }

    # ----- Problem 1: Traffic -----
    traffic_edge = {}
    for frac in TRAFFIC_EDGE_FRACS:
        T = max(1, int(frac * n_edges))
        sel = optimize_traffic(edges, T)
        lengths_sel = [edges[i]['length_m'] for i in sel]
        traffic_edge[frac] = {
            'T': T, 'selected': sel,
            'traffic_frac': eval_traffic(edges, sel) / mt,
            'km_used': sum(lengths_sel) / 1000.0,
            'das_count': int(sum(np.ceil(l / DAS_SENSING_RANGE) for l in lengths_sel)),
        }

    traffic_km = {}
    for km in TRAFFIC_KM_BUDGETS:
        if km > total_km:
            continue
        sel = optimize_traffic_km(edges, km)
        lengths_sel = [edges[i]['length_m'] for i in sel]
        traffic_km[km] = {
            'n_edges': len(sel), 'selected': sel,
            'traffic_frac': eval_traffic(edges, sel) / mt,
            'km_used': sum(lengths_sel) / 1000.0,
            'das_count': int(sum(np.ceil(l / DAS_SENSING_RANGE) for l in lengths_sel)),
        }

    results['traffic'] = {'edge_budget': traffic_edge, 'km_budget': traffic_km}

    # ----- Problem 2: EEW (requires population data) -----
    if pop is not None:
        rng = np.random.RandomState(42)
        eq_locs = sample_earthquake_locations(edges, pop, EEW_MC_SAMPLES, rng)

        eew_edge = {}
        for T in EEW_T_VALUES:
            if T > n_edges:
                continue
            sel = optimize_eew(edges, T, eq_locs, EEW_MIN_SPACING)
            eew_edge[T] = {
                'selected': sel,
                'score': eval_eew(edges, sel, eq_locs),
                'km_used': sum(edges[i]['length_m'] for i in sel) / 1000.0,
            }

        eew_km = {}
        for km in EEW_KM_BUDGETS:
            if km > total_km:
                continue
            sel = optimize_eew_km(edges, km, eq_locs, EEW_MIN_SPACING)
            eew_km[km] = {
                'n_edges': len(sel), 'selected': sel,
                'score': eval_eew(edges, sel, eq_locs),
                'km_used': sum(edges[i]['length_m'] for i in sel) / 1000.0,
            }

        alert_grids = {}
        for T_val, d in eew_edge.items():
            if len(d['selected']) >= 4:
                grid, extent = compute_alert_time_grid(edges, d['selected'])
                alert_grids[T_val] = {'grid': grid, 'extent': extent}

        results['eew'] = {
            'edge_budget': eew_edge, 'km_budget': eew_km,
            'eq_locations': eq_locs,
            'alert_grids': alert_grids,
        }

    # ----- Problem 3: Activity Monitoring (requires population data) -----
    if pop is not None:
        total_pop = float(np.maximum(pop, 0).sum())

        act_edge = {}
        for T in ACTIVITY_T_VALUES:
            if T > n_edges:
                continue
            sel = optimize_activity(edges, T, pop)
            cov = eval_activity(edges, sel, pop)
            act_edge[T] = {
                'selected': sel,
                'pop_covered': cov,
                'pop_frac': cov / total_pop if total_pop > 0 else 0,
                'km_used': sum(edges[i]['length_m'] for i in sel) / 1000.0,
            }

        act_km = {}
        for km in ACTIVITY_KM_BUDGETS:
            if km > total_km:
                continue
            sel = optimize_activity_km(edges, km, pop)
            cov = eval_activity(edges, sel, pop)
            act_km[km] = {
                'n_edges': len(sel), 'selected': sel,
                'pop_covered': cov,
                'pop_frac': cov / total_pop if total_pop > 0 else 0,
                'km_used': sum(edges[i]['length_m'] for i in sel) / 1000.0,
            }

        results['activity'] = {
            'edge_budget': act_edge, 'km_budget': act_km,
            'total_pop': total_pop,
        }

    return results


# ================================================================
# 6. Plotting Helpers
# ================================================================

def plot_city_base(ax, base_lines, color='#d0d0d0', lw=0.3, alpha=0.5):
    lc = LineCollection(base_lines, colors=color, linewidths=lw, alpha=alpha, zorder=1)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')


def plot_selected_edges(ax, edges, selected, colors, lw=1.5):
    lines = [edges[i]['line'] for i in selected]
    if isinstance(colors, str):
        colors = [colors] * len(selected)
    lc = LineCollection(lines, colors=colors, linewidths=lw, alpha=0.85, zorder=5)
    ax.add_collection(lc)


def plot_pop_density(ax, edges, pop, grid_res=200, sigma=8):
    """Plot population density as a smooth background heatmap."""
    from scipy.ndimage import gaussian_filter
    if pop is None:
        return
    mx = np.array([e['mx'] for e in edges])
    my = np.array([e['my'] for e in edges])
    pad = 500
    xmin, xmax = mx.min() - pad, mx.max() + pad
    ymin, ymax = my.min() - pad, my.max() + pad

    grid = np.zeros((grid_res, grid_res))
    xi = np.linspace(xmin, xmax, grid_res)
    yi = np.linspace(ymin, ymax, grid_res)
    dx = (xmax - xmin) / grid_res
    dy = (ymax - ymin) / grid_res

    pop_safe = np.maximum(pop, 0)
    for k in range(len(edges)):
        if pop_safe[k] < 1:
            continue
        ix = int((mx[k] - xmin) / dx)
        iy = int((my[k] - ymin) / dy)
        ix = min(max(ix, 0), grid_res - 1)
        iy = min(max(iy, 0), grid_res - 1)
        grid[iy, ix] += pop_safe[k]

    grid = gaussian_filter(grid, sigma=sigma)
    grid[grid < grid.max() * 0.005] = np.nan

    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad(alpha=0)
    ax.imshow(grid, extent=(xmin, xmax, ymin, ymax), origin='lower',
              cmap=cmap, alpha=0.45, zorder=1, aspect='auto')


# ================================================================
# 7. Figure 1: Traffic Optimization Maps
# ================================================================

def figure_traffic(all_results, T_frac_show=0.1):
    cities = list(all_results.keys())
    n = len(cities)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.get_cmap('YlOrRd')

    for idx, city in enumerate(cities):
        ax = axes[idx]
        r = all_results[city]
        edges = r['edges']

        # Pick T closest to desired fraction
        T_idx = np.argmin(np.abs(np.array(r['T_fractions']) - T_frac_show))
        T = r['T_values'][T_idx]
        sel = r['traffic']['selected'][T]

        plot_city_base(ax, r['base_lines'])

        weights = np.array([edges[i]['weight'] for i in sel])
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            normed = (weights - w_min) / (w_max - w_min)
        else:
            normed = np.ones_like(weights)
        colors = [cmap(0.3 + 0.7 * v) for v in normed]

        plot_selected_edges(ax, edges, sel, colors, lw=1.8)

        frac_traffic = r['traffic']['metrics']['traffic_norm'][T_idx]
        ax.set_title(f'{city}\nT = {T} edges ({r["T_fractions"][T_idx]*100:.0f}%)  '
                     f'Traffic captured: {frac_traffic*100:.1f}%',
                     fontsize=12, fontweight='bold')

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:n], fraction=0.02, pad=0.02, shrink=0.6)
    cbar.set_label('Relative Traffic Weight', fontsize=11)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])

    fig.suptitle('Problem 1: Traffic-Optimal Interrogator Placement',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_traffic_optimization.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("  Saved fig_traffic_optimization.png")
    plt.close()


# ================================================================
# 8. Figure 2: Convex Hull Optimization
# ================================================================

def figure_hull(all_results, primary_city='Tel Aviv'):
    if primary_city not in all_results:
        primary_city = list(all_results.keys())[0]
    r = all_results[primary_city]
    edges = r['edges']
    nhv = r['n_hull_verts']

    hull_T_abs = [5, max(5, nhv // 3), max(10, 2 * nhv // 3), nhv]
    hull_T_abs = sorted(set(hull_T_abs))
    hull_colors = ['#D32F2F', '#FF9800', '#4CAF50', '#1565C0'][:len(hull_T_abs)]

    fig = plt.figure(figsize=(16, 6.5))

    # Left: map showing selected EDGES + hull polygon for the largest T
    ax_map = fig.add_axes([0.02, 0.05, 0.52, 0.85])
    plot_city_base(ax_map, r['base_lines'], color='#c8c8c8', lw=0.4)

    from matplotlib.lines import Line2D
    legend_handles = []

    # Draw nested hulls (largest first so smallest is on top)
    for t_abs, color in zip(reversed(hull_T_abs), reversed(hull_colors)):
        sel = optimize_hull(edges, t_abs)

        # Draw the actual selected fiber edges on the map
        plot_selected_edges(ax_map, edges, sel, color, lw=2.0)

        # Draw hull polygon boundary
        if len(sel) >= 3:
            pts = np.array([[edges[i]['mx'], edges[i]['my']] for i in sel])
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                poly = MplPolygon(hull_pts, closed=True, facecolor=color, alpha=0.08,
                                  edgecolor=color, linewidth=2, linestyle='--', zorder=4)
                ax_map.add_patch(poly)
                area_km2 = hull.volume / 1e6
                legend_handles.append(
                    Line2D([0], [0], color=color, linewidth=3, alpha=0.7,
                           label=f'T = {t_abs} edges → {area_km2:.1f} km²'))
            except Exception:
                pass

    ax_map.legend(handles=legend_handles[::-1], loc='upper left', fontsize=10,
                  framealpha=0.9, edgecolor='#666',
                  title=f'Selected fibers  (hull verts = {nhv})', title_fontsize=10)
    ax_map.set_title(f'{primary_city} — Hull Expansion with Budget',
                     fontsize=14, fontweight='bold')

    # Right: saturation curve
    ax_curve = fig.add_axes([0.60, 0.12, 0.36, 0.78])
    city_markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

    for ci, city in enumerate(all_results):
        r2 = all_results[city]
        hs = r2['hull_sweep']
        marker = city_markers[ci % len(city_markers)]
        ax_curve.plot(hs['T'], np.array(hs['area_norm']) * 100,
                      marker=marker, linewidth=2, markersize=5, label=city, alpha=0.85)
        hv = r2['n_hull_verts']
        ax_curve.axvline(hv, color='gray', linestyle=':', alpha=0.25)

    ax_curve.set_xlabel('Budget T  (number of interrogators)', fontsize=12)
    ax_curve.set_ylabel('Hull Area  (% of maximum)', fontsize=12)
    ax_curve.set_title('Coverage Saturation\n(dotted = hull vertex count per city)',
                       fontsize=12, fontweight='bold')
    ax_curve.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.set_ylim(0, 105)

    fig.suptitle('Problem 2: Coverage-Optimal Interrogator Placement',
                 fontsize=16, fontweight='bold')
    plt.savefig('fig_hull_optimization.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("  Saved fig_hull_optimization.png")
    plt.close()


# ================================================================
# 8b. Figure: Side-by-side comparison of all 3 strategies
# ================================================================

def figure_side_by_side(all_results, cities_to_show=None, T_frac=0.05):
    """Show the 3 strategies next to each other for direct visual comparison."""
    if cities_to_show is None:
        cities_to_show = [c for c in ['Tel Aviv', 'Barcelona', 'Manhattan']
                          if c in all_results]
    cities_to_show = [c for c in cities_to_show if c in all_results]
    if not cities_to_show:
        cities_to_show = list(all_results.keys())[:3]

    nrows = len(cities_to_show)
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 5.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    strat_names = ['traffic', 'hull', 'uniform']
    strat_titles = ['Traffic-Optimal', 'Hull-Optimal', 'Uniform Random']
    strat_colors_base = ['#D32F2F', '#2E7D32', '#1565C0']

    for row, city in enumerate(cities_to_show):
        r = all_results[city]
        edges = r['edges']
        T_idx = np.argmin(np.abs(np.array(r['T_fractions']) - T_frac))
        T = r['T_values'][T_idx]

        for col, (sname, stitle, scolor) in enumerate(
                zip(strat_names, strat_titles, strat_colors_base)):
            ax = axes[row, col]
            plot_city_base(ax, r['base_lines'], color='#d5d5d5', lw=0.3)

            sel = r[sname]['selected'][T]
            plot_selected_edges(ax, edges, sel, scolor, lw=1.4)

            # For hull strategy, also draw the hull polygon boundary
            if sname == 'hull' and len(sel) >= 3:
                pts = np.array([[edges[i]['mx'], edges[i]['my']] for i in sel])
                try:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    poly = MplPolygon(hull_pts, closed=True, facecolor='none',
                                      edgecolor=scolor, linewidth=2,
                                      linestyle='--', alpha=0.6, zorder=6)
                    ax.add_patch(poly)
                except Exception:
                    pass

            m = r[sname]['metrics']
            subtitle = (f"Traffic {m['traffic_norm'][T_idx]*100:.0f}%  |  "
                        f"Hull {m['hull_norm'][T_idx]*100:.0f}%  |  "
                        f"VMR {m['vmr'][T_idx]:.1f}")
            ax.set_title(f'{stitle}\n{subtitle}', fontsize=11,
                         fontweight='bold', color=scolor)

            if col == 0:
                ax.text(-0.08, 0.5, f'{city}\nT = {T}',
                        transform=ax.transAxes, fontsize=12, fontweight='bold',
                        va='center', ha='center', rotation=90)

    fig.suptitle(f'Three Routing Strategies Compared  (T ≈ {T_frac*100:.0f}% of edges)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_three_strategies.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("  Saved fig_three_strategies.png")
    plt.close()


# ================================================================
# 9. Figure 3: Strategy Comparison
# ================================================================

def figure_comparison(all_results, primary_city='Tel Aviv'):
    if primary_city not in all_results:
        primary_city = list(all_results.keys())[0]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    metrics_keys = ['traffic_norm', 'hull_norm', 'vmr']
    titles = ['Traffic Captured', 'Hull Area', 'Spatial Clustering (VMR)']
    ylabels = ['% of max traffic', '% of max area', 'VMR  (1 = Poisson-uniform)']

    r = all_results[primary_city]
    x = np.array(r['T_fractions']) * 100

    for ax, mk, title, ylabel in zip(axes, metrics_keys, titles, ylabels):
        for sname in ['traffic', 'hull', 'uniform']:
            y = np.array(r[sname]['metrics'][mk])
            if mk in ('traffic_norm', 'hull_norm'):
                y = y * 100
            ax.plot(x, y, marker='o', linewidth=2.5, markersize=5,
                    color=STRATEGY_COLORS[sname], label=STRATEGY_LABELS[sname], alpha=0.9)

        # Background: other cities as thin lines
        for city in all_results:
            if city == primary_city:
                continue
            r2 = all_results[city]
            x2 = np.array(r2['T_fractions']) * 100
            for sname in ['traffic', 'hull', 'uniform']:
                y2 = np.array(r2[sname]['metrics'][mk])
                if mk in ('traffic_norm', 'hull_norm'):
                    y2 = y2 * 100
                ax.plot(x2, y2, linewidth=0.7, color=STRATEGY_COLORS[sname], alpha=0.15)

        if mk == 'vmr':
            ax.axhline(1.0, color='black', linestyle=':', alpha=0.4, label='Poisson baseline')

        ax.set_xlabel('Budget T  (% of edges)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.9)

    fig.suptitle(f'Strategy Trade-offs — {primary_city} (bold) + all cities (faded)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('fig_strategy_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("  Saved fig_strategy_comparison.png")
    plt.close()


# ================================================================
# 10. Publication Figure — Nature Communications quality
# ================================================================

def figure_publication(all_results, primary_city='Tel Aviv', c_map=0.05):
    """Combined map + curves figure. Colorblind-safe Wong palette."""

    # Wong colorblind-safe palette
    C = {'traffic': '#D55E00', 'hull': '#009E73', 'uniform': '#0072B2'}
    L = {'traffic': 'Traffic-optimal', 'hull': 'Coverage-optimal', 'uniform': 'Uniform'}

    if primary_city not in all_results:
        primary_city = list(all_results.keys())[0]
    r = all_results[primary_city]
    edges = r['edges']

    T_idx = np.argmin(np.abs(np.array(r['T_fractions']) - c_map))
    T_show = r['T_values'][T_idx]
    c_actual = r['T_fractions'][T_idx]

    # ---- Layout ----
    fig = plt.figure(figsize=(7.2, 5.8), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1],
                          hspace=0.32, wspace=0.38,
                          left=0.06, right=0.97, top=0.92, bottom=0.08)

    strats = ['traffic', 'hull', 'uniform']
    map_titles = ['Traffic maximization', 'Coverage maximization', 'Uniform sensing']
    panel_ids = list('abcdef')

    # ===== ROW 1: Maps =====
    for col, (sn, mt_) in enumerate(zip(strats, map_titles)):
        ax = fig.add_subplot(gs[0, col])
        plot_city_base(ax, r['base_lines'], color='#e6e6e6', lw=0.18)

        sel = r[sn]['selected'][T_show]
        plot_selected_edges(ax, edges, sel, C[sn], lw=1.0)

        if sn == 'hull' and len(sel) >= 3:
            pts = np.array([[edges[i]['mx'], edges[i]['my']] for i in sel])
            try:
                hull = ConvexHull(pts)
                hp = pts[hull.vertices]
                ax.add_patch(MplPolygon(hp, closed=True, facecolor='none',
                             edgecolor=C[sn], lw=1.2, ls='--', alpha=0.55, zorder=6))
            except Exception:
                pass

        # Panel label
        ax.text(0.03, 0.97, f'({panel_ids[col]})', transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top')

        # Metrics badge
        m = r[sn]['metrics']
        badge = (f"Traffic {m['traffic_norm'][T_idx]*100:.0f}%   "
                 f"Hull {m['hull_norm'][T_idx]*100:.0f}%   "
                 f"VMR {m['vmr'][T_idx]:.1f}")
        ax.text(0.5, -0.02, badge, transform=ax.transAxes,
                fontsize=6, ha='center', color='#555555')

        ax.set_title(mt_, fontsize=8.5, fontweight='bold', color=C[sn], pad=4)

    # ===== ROW 2: Quantitative curves =====
    curve_cfg = [
        ('traffic_norm', r'Traffic captured  $\Sigma\, C_{ij} A_{ij}$',
         '% of maximum', False),
        ('hull_norm', r'Convex-hull area  $H(\mathrm{conv}\, g)$',
         '% of maximum', False),
        ('vmr', 'Spatial clustering  (VMR)',
         'Variance / Mean', True),
    ]

    for col, (mk, ctitle, ylabel, is_vmr) in enumerate(curve_cfg):
        ax = fig.add_subplot(gs[1, col])

        # Background: other cities
        for city in all_results:
            if city == primary_city:
                continue
            r2 = all_results[city]
            xo = np.array(r2['T_fractions'])
            for sn in strats:
                yo = np.array(r2[sn]['metrics'][mk])
                if not is_vmr:
                    yo *= 100
                ax.plot(xo, yo, lw=0.4, color=C[sn], alpha=0.10)

        # Primary city: bold
        xp = np.array(r['T_fractions'])
        for sn in strats:
            yp = np.array(r[sn]['metrics'][mk])
            if not is_vmr:
                yp *= 100
            ax.plot(xp, yp, marker='o', lw=1.8, ms=3, color=C[sn],
                    label=L[sn], alpha=0.92,
                    markeredgecolor='white', markeredgewidth=0.4, zorder=4)

        # Reference lines
        ax.axvline(c_actual, color='#999999', ls=':', lw=0.6, alpha=0.5)
        if is_vmr:
            ax.set_yscale('log')
            ax.axhline(1.0, color='#999999', ls='--', lw=0.5, alpha=0.4)
            ax.text(0.42, 0.75, 'Poisson', fontsize=5.5, color='#999999',
                    transform=ax.transAxes)
            ax.set_ylim(0.5, None)

        ax.text(0.03, 0.97, f'({panel_ids[3 + col]})', transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top')
        ax.set_xlabel(r'Coverage ratio  $c = T\,/\,|E|$', fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.set_title(ctitle, fontsize=8, fontweight='bold', pad=6)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=6.5, direction='in', length=3, width=0.4)
        ax.grid(True, alpha=0.12, lw=0.4)

        if col == 0:
            ax.legend(fontsize=6, framealpha=0.85, loc='lower right',
                      edgecolor='none', handlelength=1.5,
                      handletextpad=0.4, labelspacing=0.3)

    fig.text(0.5, 0.97,
             f'{primary_city}  —  $c = {c_actual:.3f}$  '
             f'({T_show} of {r["n_edges"]} edges)',
             ha='center', fontsize=10, fontweight='bold')
    fig.text(0.5, 0.942,
             f'Bold = {primary_city};  faded = '
             + ', '.join(c for c in all_results if c != primary_city),
             ha='center', fontsize=5.5, color='#888888')

    plt.savefig('fig_publication.png', dpi=300, bbox_inches='tight',
                facecolor='white', pad_inches=0.15)
    print("  Saved fig_publication.png")
    plt.close()


# ================================================================
# 11. Main Figure — Marginal Contribution Heatmap (1×3)
# ================================================================

def compute_marginal(edges, selected, task, pop_per_edge=None):
    """Per-edge marginal contribution (leave-one-out), normalized to [0,1]."""
    n_sel = len(selected)
    if n_sel == 0:
        return np.array([])

    if task == 'traffic':
        vals = np.array([edges[i]['weight'] for i in selected], dtype=float)

    elif task == 'hull':
        full_area = eval_hull_area(edges, selected)
        vals = np.zeros(n_sel)
        for j, idx in enumerate(selected):
            subset = [s for s in selected if s != idx]
            vals[j] = full_area - eval_hull_area(edges, subset)

    elif task == 'hull_pop':
        full_pop = eval_hull_population(edges, selected, pop_per_edge)
        vals = np.zeros(n_sel)
        for j, idx in enumerate(selected):
            subset = [s for s in selected if s != idx]
            vals[j] = full_pop - eval_hull_population(edges, subset, pop_per_edge)

    elif task == 'uniform':
        full_vmr = eval_uniformity_vmr(edges, selected)
        vals = np.zeros(n_sel)
        for j, idx in enumerate(selected):
            subset = [s for s in selected if s != idx]
            vals[j] = eval_uniformity_vmr(edges, subset) - full_vmr

    else:
        vals = np.ones(n_sel)

    vmin, vmax = vals.min(), vals.max()
    if vmax > vmin:
        vals = (vals - vmin) / (vmax - vmin)
    else:
        vals = np.ones(n_sel) * 0.5

    return vals


def figure_main(all_results, c=0.01):
    """2×3 figure: 6 cities from 6 continents, columns = tasks."""

    task_keys = ['traffic', 'hull', 'uniform']
    task_labels = ['Traffic maximization', 'Coverage maximization', 'Uniform sensing']

    city_grid = [
        ['Manhattan', 'Cape Town', 'Sydney'],
        ['Buenos Aires', 'Tel Aviv', 'Barcelona'],
    ]

    available = list(all_results.keys())
    for row in range(2):
        for col in range(3):
            if city_grid[row][col] not in all_results:
                used = {city_grid[r][cc] for r in range(2) for cc in range(3)
                        if (r, cc) != (row, col)}
                for av in available:
                    if av not in used:
                        city_grid[row][col] = av
                        break

    fig, axes = plt.subplots(2, 3, figsize=(20, 14), facecolor='white')

    base_cmap = plt.cm.YlOrRd
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'warmth', base_cmap(np.linspace(0.12, 0.92, 256)))

    opt_fns = {'traffic': optimize_traffic,
               'hull': optimize_hull,
               'uniform': optimize_uniform}

    print("    figure_main edge counts (c={:.3f}):".format(c))
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            ax.set_facecolor('white')

            city = city_grid[row][col]
            task = task_keys[col]
            r = all_results[city]
            edges = r['edges']
            T_val = max(3, int(c * r['n_edges']))
            print(f"      {city}: {T_val}/{r['n_edges']} edges")

            plot_city_base(ax, r['base_lines'], color='#c0c0c0',
                           lw=0.35, alpha=0.6)

            selected = opt_fns[task](edges, T_val)
            marginals = compute_marginal(edges, selected, task)

            lines = [edges[i]['line'] for i in selected]
            colors = cmap(marginals)
            lws = 2.0 + 3.5 * marginals

            lc = LineCollection(lines, colors=colors, linewidths=lws,
                                zorder=5, alpha=0.92)
            ax.add_collection(lc)

            ax.set_title(city, fontsize=15, fontweight='bold',
                         color='#222222', pad=8)
            ax.axis('off')
            ax.set_aspect('equal')

    for col, label in enumerate(task_labels):
        x = (col + 0.5) / 3.0 * 0.88 + 0.04
        fig.text(x, 0.98, label, ha='center', fontsize=17,
                 fontweight='bold', color='#333333')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Normalized marginal contribution', fontsize=13,
                   fontweight='bold', color='#333333', labelpad=12)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(colors='#333333', labelsize=10)
    cbar.outline.set_edgecolor('#aaaaaa')

    plt.subplots_adjust(left=0.02, right=0.91, top=0.93, bottom=0.02,
                        wspace=0.05, hspace=0.10)
    plt.savefig('fig_main_marginal.png', dpi=400, bbox_inches='tight',
                facecolor='white', pad_inches=0.3)
    print("  Saved fig_main_marginal.png")
    plt.close()


def figure_panels(all_results, c=0.01):
    """Individual standalone panels per city × task, for manual composition."""

    base_tasks = ['traffic', 'hull', 'uniform']
    task_label = {'traffic': 'Traffic maximization',
                  'hull': 'Coverage maximization (area)',
                  'hull_pop': 'Coverage maximization (population)',
                  'uniform': 'Uniform sensing'}

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'edge_importance',
        ['#2979FF', '#FF6D00', '#B71C1C'], N=256)

    opt_fns = {'traffic': optimize_traffic,
               'hull': optimize_hull,
               'uniform': optimize_uniform}

    panel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'panels')
    os.makedirs(panel_dir, exist_ok=True)

    for city, r in all_results.items():
        edges = r['edges']
        T_val = max(3, int(c * r['n_edges']))
        pop = r.get('pop_per_edge')

        tasks = base_tasks + (['hull_pop'] if pop is not None else [])

        for task in tasks:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')

            plot_city_base(ax, r['base_lines'], color='#909090',
                           lw=0.5, alpha=0.65)

            if task in opt_fns:
                selected = opt_fns[task](edges, T_val)
            else:
                selected = optimize_hull_pop(edges, T_val, pop)

            marginals = compute_marginal(edges, selected, task,
                                         pop_per_edge=pop)

            lines_sel = [edges[i]['line'] for i in selected]
            colors = cmap(marginals)
            lws = 2.5 + 3.0 * marginals

            lc = LineCollection(lines_sel, colors=colors, linewidths=lws,
                                zorder=5, alpha=0.95)
            ax.add_collection(lc)

            ax.set_title(f'{city}\n{task_label[task]}',
                         fontsize=18, fontweight='bold', pad=12)
            ax.axis('off')
            ax.set_aspect('equal')

            fname = os.path.join(
                panel_dir,
                f'{city.replace(" ", "_")}_{task}.png')
            plt.savefig(fname, dpi=400, bbox_inches='tight',
                        facecolor='white', pad_inches=0.15)
            plt.close()
            print(f"    {fname}")

    # Standalone colorbar (recreate cmap to be safe)
    cmap_cb = mcolors.LinearSegmentedColormap.from_list(
        'edge_importance', ['#2979FF', '#FF6D00', '#B71C1C'], N=256)
    fig, ax = plt.subplots(1, 1, figsize=(1.2, 6), facecolor='white')
    sm = plt.cm.ScalarMappable(cmap=cmap_cb, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation='vertical')
    cbar.set_label('Normalized marginal contribution',
                   fontsize=12, fontweight='bold')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar_path = os.path.join(panel_dir, 'colorbar.png')
    plt.savefig(cbar_path, dpi=400, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()
    print(f"    {cbar_path}")


# ================================================================
# 12. New Figures — Traffic + EEW with Both Budget Modes
# ================================================================

def figure_traffic_new(all_results):
    """Traffic captured vs. edge-count budget and km budget, across all cities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for city, r in all_results.items():
        t = r['traffic']
        # Edge-count budget
        fracs = sorted(t['edge_budget'].keys())
        traffic_pct = [t['edge_budget'][f]['traffic_frac'] * 100 for f in fracs]
        ax1.plot([f * 100 for f in fracs], traffic_pct,
                 marker='o', linewidth=2, markersize=5, label=city, alpha=0.8)

        # km budget
        kms = sorted(t['km_budget'].keys())
        if kms:
            traffic_pct_km = [t['km_budget'][k]['traffic_frac'] * 100 for k in kms]
            ax2.plot(kms, traffic_pct_km,
                     marker='s', linewidth=2, markersize=5, label=city, alpha=0.8)

    ax1.set_xlabel('Budget  (% of edges)', fontsize=13)
    ax1.set_ylabel('Traffic captured  (%)', fontsize=13)
    ax1.set_title('Traffic Optimisation — Edge-count Budget', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Budget  (km of fiber)', fontsize=13)
    ax2.set_ylabel('Traffic captured  (%)', fontsize=13)
    ax2.set_title('Traffic Optimisation — km Budget', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Problem 1: Traffic Activity Monitoring', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('fig_traffic_budgets.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("  Saved fig_traffic_budgets.png")
    plt.close()


def figure_eew_new(all_results):
    """EEW score vs. edge-count and km budget, across all cities."""
    cities_with_eew = {c: r for c, r in all_results.items() if 'eew' in r}
    if not cities_with_eew:
        print("  Skipping EEW figure — no population data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for city, r in cities_with_eew.items():
        e = r['eew']
        # Edge-count budget
        Ts = sorted(e['edge_budget'].keys())
        scores = [e['edge_budget'][t]['score'] for t in Ts]
        ax1.plot(Ts, scores, marker='o', linewidth=2, markersize=5,
                 label=city, alpha=0.8)

        # km budget
        kms = sorted(e['km_budget'].keys())
        if kms:
            scores_km = [e['km_budget'][k]['score'] for k in kms]
            ax2.plot(kms, scores_km, marker='s', linewidth=2, markersize=5,
                     label=city, alpha=0.8)

    ax1.set_xlabel('Budget T  (number of DAS)', fontsize=13)
    ax1.set_ylabel('Max detection distance  (m, lower = better)', fontsize=13)
    ax1.set_title('EEW Optimisation — DAS-count Budget', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Budget  (km of fiber)', fontsize=13)
    ax2.set_ylabel('Distance spread  (m, lower = better)', fontsize=13)
    ax2.set_title('EEW Optimisation — km Budget', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Problem 2: Earthquake Early Warning', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('fig_eew_budgets.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("  Saved fig_eew_budgets.png")
    plt.close()


def figure_maps_traffic(all_results, frac=0.01):
    """Map panels: traffic-optimal DAS placement for each city at given edge fraction."""
    cities = list(all_results.keys())
    n = len(cities)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.get_cmap('YlOrRd')

    for idx, city in enumerate(cities):
        ax = axes[idx]
        r = all_results[city]
        edges = r['edges']
        t = r['traffic']['edge_budget']
        closest_frac = min(t.keys(), key=lambda f: abs(f - frac))
        sel = t[closest_frac]['selected']
        T = t[closest_frac]['T']
        tf = t[closest_frac]['traffic_frac']

        plot_city_base(ax, r['base_lines'])
        weights = np.array([edges[i]['weight'] for i in sel])
        if weights.max() > weights.min():
            normed = (weights - weights.min()) / (weights.max() - weights.min())
        else:
            normed = np.ones_like(weights)
        colors = [cmap(0.3 + 0.7 * v) for v in normed]
        plot_selected_edges(ax, edges, sel, colors, lw=1.8)
        ax.set_title(f'{city}\nT={T} ({closest_frac*100:.1f}%)  '
                     f'Traffic: {tf*100:.1f}%', fontsize=11, fontweight='bold')

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Traffic-Optimal DAS Placement (c ≈ {frac*100:.1f}%)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_traffic_maps.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("  Saved fig_traffic_maps.png")
    plt.close()


def figure_maps_eew(all_results, T_show=5):
    """Map panels: EEW-optimal DAS placement with population density heatmap."""
    cities_with_eew = {c: r for c, r in all_results.items()
                       if 'eew' in r and T_show in r['eew']['edge_budget']}
    if not cities_with_eew:
        print("  Skipping EEW maps — no data")
        return

    cities = list(cities_with_eew.keys())
    n = len(cities)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, city in enumerate(cities):
        ax = axes[idx]
        r = all_results[city]
        edges = r['edges']
        pop = r.get('pop_per_edge')
        eew = r['eew']
        sel = eew['edge_budget'][T_show]['selected']
        score = eew['edge_budget'][T_show]['score']
        km_used = eew['edge_budget'][T_show]['km_used']

        plot_pop_density(ax, edges, pop)
        plot_city_base(ax, r['base_lines'], color='#c0c0c0', lw=0.25, alpha=0.4)

        plot_selected_edges(ax, edges, sel, '#1565C0', lw=3.0)

        for i in sel:
            circle = plt.Circle((edges[i]['mx'], edges[i]['my']),
                                EEW_MIN_SPACING, fill=False,
                                edgecolor='#1565C0', linewidth=1.0,
                                linestyle='--', alpha=0.5, zorder=6)
            ax.add_patch(circle)
            ax.scatter(edges[i]['mx'], edges[i]['my'],
                       c='#1565C0', s=60, zorder=7, edgecolors='white', linewidths=1)

        ax.set_title(f'{city}\nT={T_show} DAS, {km_used:.1f} km, score={score:.0f}',
                     fontsize=11, fontweight='bold')

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'EEW-Optimal DAS Placement (T = {T_show})',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_eew_maps.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("  Saved fig_eew_maps.png")
    plt.close()


# ================================================================
# 12b. Individual Panels (for paper figure composition)
# ================================================================

def figure_panels_new(all_results):
    """Individual standalone panels per city × task × budget for paper selection."""
    panel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'panels')
    os.makedirs(panel_dir, exist_ok=True)
    count = 0

    cmap_traffic = plt.get_cmap('YlOrRd')

    for city, r in all_results.items():
        edges = r['edges']
        pop = r.get('pop_per_edge')
        safe_name = city.replace(' ', '_')

        # --- Traffic panels (edge-count budgets) ---
        for frac, d in r['traffic']['edge_budget'].items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
            plot_city_base(ax, r['base_lines'], color='#909090', lw=0.5, alpha=0.65)
            sel = d['selected']
            weights = np.array([edges[i]['weight'] for i in sel])
            if weights.max() > weights.min():
                normed = (weights - weights.min()) / (weights.max() - weights.min())
            else:
                normed = np.ones_like(weights)
            colors = [cmap_traffic(0.3 + 0.7 * v) for v in normed]
            plot_selected_edges(ax, edges, sel, colors, lw=2.5)
            ax.set_title(f'{city} — Traffic\n'
                         f'T={d["T"]} ({frac*100:.1f}%), {d["km_used"]:.1f} km, '
                         f'{d.get("das_count", "?")} DAS, '
                         f'traffic={d["traffic_frac"]*100:.1f}%',
                         fontsize=16, fontweight='bold', pad=12)
            ax.axis('off'); ax.set_aspect('equal')
            fname = os.path.join(panel_dir, f'{safe_name}_traffic_{frac*100:.1f}pct.png')
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
            plt.close(); count += 1

        # --- Traffic panels (km budgets) ---
        for km, d in r['traffic']['km_budget'].items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
            plot_city_base(ax, r['base_lines'], color='#909090', lw=0.5, alpha=0.65)
            sel = d['selected']
            weights = np.array([edges[i]['weight'] for i in sel])
            if weights.max() > weights.min():
                normed = (weights - weights.min()) / (weights.max() - weights.min())
            else:
                normed = np.ones_like(weights)
            colors = [cmap_traffic(0.3 + 0.7 * v) for v in normed]
            plot_selected_edges(ax, edges, sel, colors, lw=2.5)
            ax.set_title(f'{city} — Traffic\n'
                         f'{d["n_edges"]} edges, {d["km_used"]:.1f}/{km} km, '
                         f'{d.get("das_count", "?")} DAS, '
                         f'traffic={d["traffic_frac"]*100:.1f}%',
                         fontsize=16, fontweight='bold', pad=12)
            ax.axis('off'); ax.set_aspect('equal')
            fname = os.path.join(panel_dir, f'{safe_name}_traffic_{km}km.png')
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
            plt.close(); count += 1

        # --- EEW panels ---
        if 'eew' not in r:
            continue
        eew = r['eew']

        for T, d in eew['edge_budget'].items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
            plot_pop_density(ax, edges, pop)
            plot_city_base(ax, r['base_lines'], color='#b0b0b0', lw=0.35, alpha=0.5)
            sel = d['selected']
            plot_selected_edges(ax, edges, sel, '#1565C0', lw=3.5)
            for i in sel:
                circle = plt.Circle((edges[i]['mx'], edges[i]['my']),
                                    EEW_MIN_SPACING, fill=False,
                                    edgecolor='#1565C0', lw=1.2, ls='--', alpha=0.5, zorder=6)
                ax.add_patch(circle)
                ax.scatter(edges[i]['mx'], edges[i]['my'],
                           c='#1565C0', s=80, zorder=7, edgecolors='white', linewidths=1.2)
            ax.set_title(f'{city} — EEW\n'
                         f'T={T} DAS, {d["km_used"]:.1f} km, score={d["score"]:.0f}',
                         fontsize=16, fontweight='bold', pad=12)
            ax.axis('off'); ax.set_aspect('equal')
            fname = os.path.join(panel_dir, f'{safe_name}_eew_T{T}.png')
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
            plt.close(); count += 1

        for km, d in eew['km_budget'].items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
            plot_pop_density(ax, edges, pop)
            plot_city_base(ax, r['base_lines'], color='#b0b0b0', lw=0.35, alpha=0.5)
            sel = d['selected']
            plot_selected_edges(ax, edges, sel, '#1565C0', lw=3.5)
            for i in sel:
                circle = plt.Circle((edges[i]['mx'], edges[i]['my']),
                                    EEW_MIN_SPACING, fill=False,
                                    edgecolor='#1565C0', lw=1.2, ls='--', alpha=0.5, zorder=6)
                ax.add_patch(circle)
                ax.scatter(edges[i]['mx'], edges[i]['my'],
                           c='#1565C0', s=80, zorder=7, edgecolors='white', linewidths=1.2)
            ax.set_title(f'{city} — EEW\n'
                         f'{d["n_edges"]} DAS, {d["km_used"]:.1f}/{km} km, '
                         f'score={d["score"]:.0f}',
                         fontsize=16, fontweight='bold', pad=12)
            ax.axis('off'); ax.set_aspect('equal')
            fname = os.path.join(panel_dir, f'{safe_name}_eew_{km}km.png')
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
            plt.close(); count += 1

        # --- EEW Excess Detection Delay Heatmap panels ---
        alert_grids = eew.get('alert_grids', {})
        if alert_grids:
            from matplotlib.colors import LinearSegmentedColormap
            _nature_colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
                              '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
            cmap_nature = LinearSegmentedColormap.from_list(
                'nature_delay', _nature_colors, N=256)

            all_mx = np.array([e['mx'] for e in edges])
            all_my = np.array([e['my'] for e in edges])
            span = max(all_mx.max() - all_mx.min(), all_my.max() - all_my.min())
            buf = 0.03 * span
            city_xlim = (all_mx.min() - buf, all_mx.max() + buf)
            city_ylim = (all_my.min() - buf, all_my.max() + buf)

            for T_val, ag in alert_grids.items():
                grid = ag['grid']
                xmin, xmax, ymin, ymax = ag['extent']
                sel = eew['edge_budget'][T_val]['selected']

                fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
                im = ax.imshow(grid, extent=(xmin, xmax, ymin, ymax),
                               origin='lower', cmap=cmap_nature, alpha=0.85,
                               vmin=0, vmax=50, zorder=1, aspect='auto')
                plot_city_base(ax, r['base_lines'], color='#333333', lw=0.3, alpha=0.5)
                for i in sel:
                    ax.scatter(edges[i]['mx'], edges[i]['my'],
                               c='white', s=120, zorder=7, edgecolors='black',
                               linewidths=1.5, marker='^')
                cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, shrink=0.8)
                cbar.set_label('$100 \\times (t_P - t_P^{\\mathrm{min}})\\,/\\, t_P^{\\mathrm{min}}$  (%)',
                               fontsize=13, fontweight='bold')
                cbar.set_ticks([0, 10, 20, 30, 40, 50])
                cbar.set_ticklabels(['0', '10', '20', '30', '40', '50+'])

                ax.set_xlim(city_xlim)
                ax.set_ylim(city_ylim)

                mean_pct = grid.mean()
                ax.set_title(f'{city} — EEW Excess Detection Delay\n'
                             f'T={len(sel)} DAS, '
                             f'mean={mean_pct:.0f}%',
                             fontsize=16, fontweight='bold', pad=12)
                ax.axis('off'); ax.set_aspect('equal')
                fname = os.path.join(panel_dir, f'{safe_name}_alert_T{T_val}.png')
                plt.savefig(fname, dpi=300, bbox_inches='tight',
                            facecolor='white', pad_inches=0.1)
                plt.close(); count += 1

        # --- Activity Monitoring panels ---
        if 'activity' not in r:
            continue
        act = r['activity']

        for T, d in act['edge_budget'].items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
            plot_pop_density(ax, edges, pop)
            plot_city_base(ax, r['base_lines'], color='#b0b0b0', lw=0.35, alpha=0.5)
            sel = d['selected']
            plot_selected_edges(ax, edges, sel, '#2E7D32', lw=3.5)
            for i in sel:
                circle = plt.Circle((edges[i]['mx'], edges[i]['my']),
                                    ACTIVITY_COVERAGE_RADIUS, fill=True,
                                    facecolor='#2E7D32', edgecolor='#2E7D32',
                                    lw=1.2, alpha=0.15, zorder=4)
                ax.add_patch(circle)
                ax.scatter(edges[i]['mx'], edges[i]['my'],
                           c='#2E7D32', s=80, zorder=7, edgecolors='white', linewidths=1.2)
            ax.set_title(f'{city} — Activity Monitoring\n'
                         f'T={T} DAS, {d["km_used"]:.1f} km, '
                         f'pop covered={d["pop_frac"]*100:.1f}%',
                         fontsize=16, fontweight='bold', pad=12)
            ax.axis('off'); ax.set_aspect('equal')
            fname = os.path.join(panel_dir, f'{safe_name}_activity_T{T}.png')
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
            plt.close(); count += 1

        for km, d in act['km_budget'].items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
            plot_pop_density(ax, edges, pop)
            plot_city_base(ax, r['base_lines'], color='#b0b0b0', lw=0.35, alpha=0.5)
            sel = d['selected']
            plot_selected_edges(ax, edges, sel, '#2E7D32', lw=3.5)
            for i in sel:
                circle = plt.Circle((edges[i]['mx'], edges[i]['my']),
                                    ACTIVITY_COVERAGE_RADIUS, fill=True,
                                    facecolor='#2E7D32', edgecolor='#2E7D32',
                                    lw=1.2, alpha=0.15, zorder=4)
                ax.add_patch(circle)
                ax.scatter(edges[i]['mx'], edges[i]['my'],
                           c='#2E7D32', s=80, zorder=7, edgecolors='white', linewidths=1.2)
            ax.set_title(f'{city} — Activity Monitoring\n'
                         f'{d["n_edges"]} DAS, {d["km_used"]:.1f}/{km} km, '
                         f'pop covered={d["pop_frac"]*100:.1f}%',
                         fontsize=16, fontweight='bold', pad=12)
            ax.axis('off'); ax.set_aspect('equal')
            fname = os.path.join(panel_dir, f'{safe_name}_activity_{km}km.png')
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
            plt.close(); count += 1

    # --- EEW colorbar ---
    fig, ax = plt.subplots(figsize=(1.2, 6), facecolor='white')
    cmap_pop = plt.get_cmap('YlOrRd').copy()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_pop),
                      cax=ax, orientation='vertical')
    cb.set_label('Population density (relative)', fontsize=12)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(['Low', '', 'Medium', '', 'High'])
    fname = os.path.join(panel_dir, 'eew_colorbar.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close(); count += 1

    print(f"  Saved {count} panels to {panel_dir}/")


# ================================================================
# 13. Summary Table
# ================================================================

def print_summary(all_results):
    w = 100
    print("\n" + "=" * w)
    print("TRAFFIC OPTIMISATION SUMMARY")
    print("=" * w)
    hdr = f"{'City':<18} {'Budget Mode':<16} {'Budget':>10} {'T':>6} {'km used':>10} {'DAS':>8} {'Traffic%':>10}"
    print(hdr)
    print("-" * w)

    for city, r in all_results.items():
        t = r['traffic']
        for frac in sorted(t['edge_budget'].keys()):
            d = t['edge_budget'][frac]
            print(f"{city:<18} {'edges':<16} {frac*100:>9.1f}% {d['T']:>6} "
                  f"{d['km_used']:>9.1f}km {d.get('das_count','?'):>8} "
                  f"{d['traffic_frac']*100:>9.1f}%")
        for km in sorted(t['km_budget'].keys()):
            d = t['km_budget'][km]
            print(f"{city:<18} {'km':<16} {km:>9}km {d['n_edges']:>6} "
                  f"{d['km_used']:>9.1f}km {d.get('das_count','?'):>8} "
                  f"{d['traffic_frac']*100:>9.1f}%")
        print()

    cities_with_eew = {c: r for c, r in all_results.items() if 'eew' in r}
    if cities_with_eew:
        print("=" * w)
        print("EEW OPTIMISATION SUMMARY")
        print("=" * w)
        hdr = f"{'City':<18} {'Budget Mode':<16} {'Budget':>10} {'T':>6} {'km used':>10} {'Score':>12}"
        print(hdr)
        print("-" * w)
        for city, r in cities_with_eew.items():
            e = r['eew']
            for T in sorted(e['edge_budget'].keys()):
                d = e['edge_budget'][T]
                print(f"{city:<18} {'DAS count':<16} {T:>10} {len(d['selected']):>6} "
                      f"{d['km_used']:>9.1f}km {d['score']:>11.0f}")
            for km in sorted(e['km_budget'].keys()):
                d = e['km_budget'][km]
                print(f"{city:<18} {'km':<16} {km:>9}km {d['n_edges']:>6} "
                      f"{d['km_used']:>9.1f}km {d['score']:>11.0f}")
            print()

    cities_with_act = {c: r for c, r in all_results.items() if 'activity' in r}
    if cities_with_act:
        print("=" * w)
        print("ACTIVITY MONITORING SUMMARY")
        print("=" * w)
        hdr = f"{'City':<18} {'Budget Mode':<16} {'Budget':>10} {'T':>6} {'km used':>10} {'Pop Cov%':>12}"
        print(hdr)
        print("-" * w)
        for city, r in cities_with_act.items():
            a = r['activity']
            for T in sorted(a['edge_budget'].keys()):
                d = a['edge_budget'][T]
                print(f"{city:<18} {'DAS count':<16} {T:>10} {len(d['selected']):>6} "
                      f"{d['km_used']:>9.1f}km {d['pop_frac']*100:>11.1f}%")
            for km in sorted(a['km_budget'].keys()):
                d = a['km_budget'][km]
                print(f"{city:<18} {'km':<16} {km:>9}km {d['n_edges']:>6} "
                      f"{d['km_used']:>9.1f}km {d['pop_frac']*100:>11.1f}%")
            print()


# ================================================================
# Main
# ================================================================

def main(city_filter=None):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time()

    print("=" * 60)
    print("OPTIMAL DAS PLACEMENT — TRAFFIC + EEW + ACTIVITY")
    print("=" * 60)

    # Optionally restrict to a subset of cities (names must match CITIES keys).
    if city_filter:
        cities = {k: v for k, v in CITIES.items() if k in city_filter}
        missing = [c for c in city_filter if c not in CITIES]
        if missing:
            print(f"    WARNING: unknown cities ignored: {missing}")
        if not cities:
            print("    No valid cities selected. Available: "
                  + ", ".join(CITIES.keys()))
            return
    else:
        cities = CITIES

    # 1. Load cities
    print("\n1. Loading city street networks...")
    graphs = {}
    for name, query in cities.items():
        try:
            graphs[name] = load_city(name, query)
        except Exception as e:
            print(f"    SKIP {name}: {e}")

    # 2. Analyse with both budget modes
    print("\n2. Running optimisations (traffic + EEW, edge-count + km budgets)...")
    all_results = {}
    for name, G_proj in graphs.items():
        try:
            all_results[name] = analyze_city(name, G_proj)
        except Exception as e:
            print(f"    SKIP {name}: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("No cities processed successfully.")
        return

    # 3. Summary
    print_summary(all_results)

    # 4. Figures
    print("\n3. Generating figures...")
    figure_traffic_new(all_results)
    figure_maps_traffic(all_results, frac=0.01)
    figure_eew_new(all_results)
    figure_maps_eew(all_results, T_show=6)
    figure_panels_new(all_results)

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Optimal DAS fiber placement for traffic, EEW, and activity sensing.")
    parser.add_argument(
        '--cities', nargs='+', default=None, metavar='CITY',
        help="Subset of cities to process (names must match the CITIES dict, "
             "e.g. --cities 'Tel Aviv' Manhattan). Default: all cities. "
             "The bundled sample (Tel Aviv, Manhattan, Barcelona) runs offline; "
             "any other city is downloaded from OpenStreetMap on first use.")
    args = parser.parse_args()
    main(city_filter=args.cities)
