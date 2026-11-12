"""Round 10: EEW architecture row of Table S8 with the 4% backbone (replaces the major-road zones of round 7).
Uses the round-7 population-prior design (T = 6, h = 10 km, vP = 5 km/s) stored in eew_round7.json."""
import os, sys, json, pickle
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
OUT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, OUT)
import optimal_routing as orx
from eew_common import H_PRIMARY, VP_PRIMARY, MIN_SPACING, alarm_times
from backbone import backbone
from infra_architectures import load

G, edges, meta = load('Mexico City')
pop = np.maximum(orx.load_population_per_edge(edges, G, 'Mexico City'), 0).astype(float)
mid = np.array([[e['mx'], e['my']] for e in edges])
eq = orx.sample_earthquake_locations(edges, pop, orx.EEW_MC_SAMPLES, np.random.RandomState(42))
w = np.full(len(eq), 1 / len(eq))
R = json.load(open(os.path.join(OUT, 'eew_round7.json')))
g6 = np.array(R['designs']['P_T6'])
pieces, mask = backbone(edges, [x[0] for x in meta]); bidx = np.where(mask)[0]
pts = []
for i in np.random.default_rng(5).permutation(bidx):
    if not pts or np.min(np.hypot(*(mid[pts] - mid[i]).T)) >= MIN_SPACING:
        pts.append(i)
def metr(sel):
    t4, t1 = alarm_times(mid, np.asarray(sel), eq, H_PRIMARY, VP_PRIMARY)
    return {'t4_mean_s': float((w * t4).sum()), 'worst4_s': float(t4.max()), 'worst_nearest_s': float(t1.max())}
lr, d6, hy = metr(pts), metr(g6), metr(list(pts) + list(g6))
old = R.get('architectures_eew')
R['architectures_eew_majorroads_round7'] = old
R['architectures_eew'] = {'zones': len(pts), 'backbone_km': float(sum(edges[i]['length_m'] for i in bidx) / 1000), 'long_range': lr, 'greedy6': d6, 'hybrid6': hy}
json.dump(R, open(os.path.join(OUT, 'eew_round7.json'), 'w'), indent=1, default=float)
print(f"backbone {R['architectures_eew']['backbone_km']:.0f} km, zones {len(pts)}: long-range {lr['t4_mean_s']:.2f} s, greedy6 {d6['t4_mean_s']:.2f} s, hybrid {hy['t4_mean_s']:.2f} s  (old major-road zones {old['zones']}: {old['long_range']['t4_mean_s']:.2f} s)")
