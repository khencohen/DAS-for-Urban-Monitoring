"""Shared earthquake-early-warning design used by all revision scripts (round 7).

Objective (co-author decision, Oct 2026): minimize the population-weighted mean alarm time, where
the alarm is raised when the M-th distinct segment (M = 4) detects the P wave:
    L = sum_q w_q * delta_(M)(q) / vP ,   w_q = exposure weight of candidate source q,
subject to a minimum spacing between segments. Candidate sources are a fixed set of points with
weights (population-weighted: 300 epicentres sampled in proportion to population, unit weights;
uniform: uniform over the footprint; hazard: catalogue events with harm weights).
Greedy: segments are added one at a time; while fewer than M segments are selected the objective
uses the k-th nearest with k = number selected.

Primary scenario: h = 10 km, vP = 5 km/s, vS = 2.9 km/s. Sensitivity: h = 20 km, vP = 7, vS = 4.
Latency: detection 0.5 s after the M-th P arrival plus 0.5 s processing and alert = 1 s (baseline),
with 3 s as a pessimistic case.
"""
import numpy as np
from scipy.spatial.distance import cdist

H_PRIMARY, VP_PRIMARY, VS_PRIMARY = 10000.0, 5000.0, 2900.0
H_SENS, VP_SENS, VS_SENS = 20000.0, 7000.0, 4000.0
M_ALARM = 4
T_LAT_BASE, T_LAT_PESS = 1.0, 3.0
MIN_SPACING = 5000.0


def design_mean_alarm(mid, eq, w, T, h, vp=1.0, M=M_ALARM, min_spacing=MIN_SPACING, allowed=None):
    """Greedy design minimizing sum_q w_q * delta_(M)(q) (travel time if vp is given).

    mid: (N, 2) candidate segment midpoints; eq: (Q, 2) sources; w: (Q,) weights (normalized here);
    h: source depth (scalar or (Q,) array). Returns the list of selected segment indices.
    """
    w = np.asarray(w, float); w = w / w.sum()
    h = np.broadcast_to(np.asarray(h, float), (len(eq),))
    D = (np.sqrt(cdist(mid, eq) ** 2 + h[None, :] ** 2) / vp).astype(np.float32)   # (N, Q)
    N = len(mid)
    allowed = np.ones(N, bool) if allowed is None else allowed.copy()
    sel = []
    sel_D = np.empty((0, len(eq)), np.float32)
    for step in range(T):
        ok = allowed.copy(); ok[sel] = False
        if sel:
            ok &= cdist(mid, mid[sel]).min(1) >= min_spacing
        cand = np.where(ok)[0]
        if len(cand) == 0:                       # spacing infeasible: relax it
            ok = allowed.copy(); ok[sel] = False; cand = np.where(ok)[0]
            if len(cand) == 0:
                break
        k = min(M, len(sel) + 1)                 # order statistic used at this step
        if len(sel) == 0:
            score = D[cand] @ w
        else:
            # k-th smallest among current selection plus the candidate, for every q
            # current sorted distances (len(sel), Q); insert candidate row
            base = np.sort(sel_D, axis=0)
            if k <= len(sel):
                kth_base = base[k - 1]            # k-th without candidate
                km1 = base[k - 2] if k >= 2 else np.full(len(eq), -np.inf, np.float32)
                # with candidate c: new k-th = c if km1 <= c < kth_base ; else kth_base if c >= kth_base; else kth_base? (c < km1 -> k-th becomes base[k-2])
                C = D[cand]
                new_k = np.where(C >= kth_base, kth_base, np.where(C >= km1, C, km1 if k >= 2 else C))
            else:                                 # k == len(sel) + 1: k-th is max(candidate, current max)
                C = D[cand]; new_k = np.maximum(C, base[-1])
            score = new_k @ w
        b = int(cand[np.argmin(score)])
        sel.append(b); sel_D = np.vstack([sel_D, D[b][None, :]])
    return sel


def alarm_times(mid, sel, eq, h, vp, M=M_ALARM):
    """P-wave arrival at the M-th nearest selected segment for each source (seconds)."""
    h = np.broadcast_to(np.asarray(h, float), (len(eq),))
    D = np.sqrt(cdist(mid[sel], eq) ** 2 + h[None, :] ** 2) / vp
    D = np.sort(D, axis=0)
    return D[min(M, len(sel)) - 1], D[0]


def random_spaced(mid, T, rng, min_spacing=MIN_SPACING):
    sel = []
    for i in rng.permutation(len(mid)):
        if all(np.hypot(*(mid[i] - mid[j])) >= min_spacing for j in sel):
            sel.append(i)
            if len(sel) == T:
                break
    return np.array(sel)
