"""Round 7: all earthquake-early-warning results with the co-author's objective and scenario.

Objective: population-weighted mean alarm time of the M = 4 segment (eew_common.design_mean_alarm).
Primary scenario h = 10 km, vP = 5 km/s, vS = 2.9 km/s; sensitivity h = 20 km, vP = 7, vS = 4.
Latency: 1 s baseline (0.5 s detection + 0.5 s processing/alert), 3 s pessimistic.

Outputs (SciRep folder + here):
  p2_v2.pdf                 Fig. 3 middle row (c placements, d alarm time of 4th segment, primary scenario)
  fig_eew_priors.pdf        SI figure: catalogue + designs under uniform / population / hazard priors
  fig_eew_warning.pdf       SI figure: warning vs distance, warned share vs latency
  fig_eew_warned_map.pdf    SI map: warned share by epicentre
  eew_round7.json, si_table_priors_body.tex, si_table_s11_body.tex (random vs design), si_table_s12_body.tex (warning)
  tasks_two_edges.json      eew entries replaced (partial access, mean alarm time)
  infra eew row             printed for Table S9
"""
import os, sys, csv, json, pickle
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyproj import Transformer
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimal_routing as orx
from eew_common import (design_mean_alarm, alarm_times, random_spaced, H_PRIMARY, VP_PRIMARY, VS_PRIMARY,
                        H_SENS, VP_SENS, VS_SENS, M_ALARM, T_LAT_BASE, T_LAT_PESS, MIN_SPACING)

OUT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'); os.makedirs(PAPER, exist_ok=True)
MAJOR = {'motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link', 'primary_link'}
FS_LABEL, FS_TICK, FS_LEG, FS_PANEL = 8, 7, 6, 10


def rasterize(fig):
    for ax in fig.axes:
        for c in ax.collections: c.set_rasterized(True)
        for im in ax.images: im.set_rasterized(True)


# ---------------------------------------------------------------- data
G = pickle.load(open(os.path.join(REPO, 'city_cache', 'Mexico_City.pkl'), 'rb'))
edges = orx.extract_edges(G); base_lines = orx.get_base_lines(G)
pop = np.maximum(orx.load_population_per_edge(edges, G, 'Mexico City'), 0).astype(float)
wpop = pop / pop.sum()
mid = np.array([[e['mx'], e['my']] for e in edges]); m = len(mid); cen = mid.mean(0)
eq_pop = orx.sample_earthquake_locations(edges, pop, orx.EEW_MC_SAMPLES, np.random.RandomState(42))
w_pop = np.full(len(eq_pop), 1 / len(eq_pop))
SCEN = {'P': (H_PRIMARY, VP_PRIMARY, VS_PRIMARY), 'S': (H_SENS, VP_SENS, VS_SENS)}
res = {}

# ---------------------------------------------------------------- designs (population prior)
designs = {}
for sc, (h, vp, vs) in SCEN.items():
    for T in (4, 6, 8, 10, 20):
        designs[(sc, T)] = design_mean_alarm(mid, eq_pop, w_pop, T, h, vp)
res['designs'] = {f'{k[0]}_T{k[1]}': [int(i) for i in v] for k, v in designs.items()}
same = {T: designs[('P', T)] == designs[('S', T)] for T in (6, 10)}
print('identical placement at both depths:', same, flush=True)

# ---------------------------------------------------------------- Fig. 3 middle row (primary scenario)
h, vp, vs = SCEN['P']
lo, hi = mid.min(0), mid.max(0); pad = (hi - lo).max() * 0.35
xi = np.linspace(lo[0] - pad, hi[0] + pad, 200); yi = np.linspace(lo[1] - pad, hi[1] + pad, 200)
X, Y = np.meshgrid(xi, yi); gpts = np.c_[X.ravel(), Y.ravel()]; ext = (xi[0], xi[-1], yi[0], yi[-1])
T_LIST = [4, 6, 8, 10]
tdet = {T: alarm_times(mid, designs[('P', T)], gpts, h, vp)[0].reshape(200, 200) for T in T_LIST}
vmin, vmax = h / vp, 4.5
fig = plt.figure(figsize=(841.89 / 72, 595.276 / 72))
fig.suptitle('Mexico City', fontsize=13, y=0.90)
w, hgt, x0, y0, gap = 0.19, 0.27, 0.05, 0.56, 0.005
for k, T in enumerate(T_LIST):
    ax = fig.add_axes([x0 + k * (w + gap), y0, w, hgt])
    orx.plot_pop_density(ax, edges, pop); orx.plot_city_base(ax, base_lines, color='#c0c0c0', lw=0.15, alpha=0.4)
    for i in designs[('P', T)]:
        ax.add_patch(plt.Circle((edges[i]['mx'], edges[i]['my']), MIN_SPACING, fill=False, edgecolor='#1565C0', lw=0.6, ls='--', alpha=0.6, zorder=6))
        ax.scatter(edges[i]['mx'], edges[i]['my'], c='#1565C0', s=12, zorder=7, edgecolors='white', linewidths=0.5)
    ax.set_aspect('equal'); ax.axis('off')
    if k == 0: ax.text(-0.08, 1.0, 'c', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
cax = fig.add_axes([x0 + 4 * (w + gap) + 0.01, y0 + 0.02, 0.012, hgt - 0.04])
sm = plt.cm.ScalarMappable(cmap='YlOrRd'); sm.set_array([0, 1]); cb = fig.colorbar(sm, cax=cax); cb.set_ticks([]); cb.set_label('Population Density', fontsize=8)
for k, T in enumerate(T_LIST):
    ax = fig.add_axes([x0 + k * (w + gap), y0 - hgt - 0.01, w, hgt])
    im = ax.imshow(tdet[T], extent=ext, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='bilinear')
    orx.plot_city_base(ax, base_lines, color='#555555', lw=0.12, alpha=0.5)
    for i in designs[('P', T)]:
        ax.scatter(edges[i]['mx'], edges[i]['my'], c='white', s=10, zorder=7, edgecolors='k', linewidths=0.5)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect('equal'); ax.axis('off')
    ax.text(0.5, -0.08, str(T), transform=ax.transAxes, fontsize=11, ha='center', va='top')
    if k == 0: ax.text(-0.08, 1.0, 'd', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
cax = fig.add_axes([x0 + 4 * (w + gap) + 0.01, y0 - hgt + 0.01, 0.012, hgt - 0.04])
cb = fig.colorbar(im, cax=cax, extend='max'); cb.set_label('P-wave alarm time, 4th segment (s)', fontsize=8); cb.ax.tick_params(labelsize=7)
fig.text(x0 + 4 * (w + gap) + 0.01, y0 - hgt - 0.045, 'Number of\nInterrogators', fontsize=9, va='top')
fig.patches.append(plt.matplotlib.patches.FancyArrowPatch((x0, y0 - hgt - 0.045), (x0 + 4 * (w + gap), y0 - hgt - 0.045), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=10, color='k', lw=0.8))
rasterize(fig); fig.savefig(os.path.join(PAPER, 'p2_v2.pdf'), dpi=300); fig.savefig(os.path.join(OUT, 'p2_v2.png'), dpi=110); plt.close(fig)
print('Fig. 3 row saved', flush=True)

# ---------------------------------------------------------------- random vs design (Table S11)
rng = np.random.default_rng(29)
rows11 = []; res['random_vs_design'] = {}
for sc, (h, vp, vs) in (('P', SCEN['P']), ('S', SCEN['S'])):
    for T in (6, 10):
        sel = designs[(sc, T)]
        t4, t1 = alarm_times(mid, sel, eq_pop, h, vp)
        full = {'mean4': float((w_pop * t4).sum()), 'worst4': float(t4.max())}
        rs, ru = [], []
        for _ in range(200):
            a4, _ = alarm_times(mid, random_spaced(mid, T, rng), eq_pop, h, vp); rs.append(((w_pop * a4).sum(), a4.max()))
            a4, _ = alarm_times(mid, rng.choice(m, T, replace=False), eq_pop, h, vp); ru.append(((w_pop * a4).sum(), a4.max()))
        rs, ru = np.array(rs), np.array(ru)
        res['random_vs_design'][f'{sc}_T{T}'] = {'full': full, 'spaced_mean4': rs[:, 0].mean(), 'spaced_mean4_sd': rs[:, 0].std(), 'spaced_worst4': rs[:, 1].mean(),
                                                  'unspaced_mean4': ru[:, 0].mean(), 'unspaced_mean4_sd': ru[:, 0].std(), 'unspaced_worst4': ru[:, 1].mean(), 'tmin': h / vp}
        rows11.append(f"    {h/1000:.0f} & {vp/1000:.0f} & {T} & {h/vp:.2f} & {full['mean4']:.2f} & {full['worst4']:.2f} & {rs[:,0].mean():.2f} $\\pm$ {rs[:,0].std():.2f} & {rs[:,1].mean():.2f} & {ru[:,0].mean():.2f} $\\pm$ {ru[:,0].std():.2f} & {ru[:,1].mean():.2f} \\\\")
        print(f"{sc} T={T}: design mean4 {full['mean4']:.2f}s worst4 {full['worst4']:.2f}s | random spaced {rs[:,0].mean():.2f} | unspaced {ru[:,0].mean():.2f}", flush=True)
open(os.path.join(OUT, 'si_table_s11_body.tex'), 'w').write('\n'.join(rows11) + '\n')

# ---------------------------------------------------------------- priors (Table S5 / Fig. S6), primary scenario
h, vp, vs = SCEN['P']
tree = cKDTree(mid); pts = []
while len(pts) < 300:
    p = lo + rng.random((2000, 2)) * (hi - lo); d, _ = tree.query(p); pts.extend(p[d <= 1000].tolist())
eq_uni = np.array(pts[:300]); w_uni = np.full(300, 1 / 300)
tr = Transformer.from_crs('EPSG:4326', G.graph['crs'], always_xy=True)
cat = [r for r in csv.DictReader(open(os.path.join(OUT, 'usgs_mexico_M5_500km_1960_2026.csv'))) if r['type'] == 'earthquake']
cx, cy = tr.transform([float(r['longitude']) for r in cat], [float(r['latitude']) for r in cat])
eq_haz = np.c_[cx, cy]; h_haz = np.array([float(r['depth']) for r in cat]) * 1000.0; mag = np.array([float(r['mag']) for r in cat])
sub = rng.choice(m, 4000, p=wpop)
Rh = np.sqrt(cdist(eq_haz, mid[sub]) ** 2 + h_haz[:, None] ** 2)
w_haz = 10 ** (1.5 * (mag - 5.0)) * (1.0 / Rh ** 2).mean(1); w_haz /= w_haz.sum()
PRI = {'uniform': (eq_uni, h, w_uni), 'population': (eq_pop, h, w_pop), 'hazard': (eq_haz, h_haz, w_haz)}
pdes = {}
for name, (q, hh, ww) in PRI.items():
    for T in (6, 10):
        pdes[(name, T)] = design_mean_alarm(mid, q, ww, T, hh, vp)
rows5 = []; res['priors'] = {}
for T in (6, 10):
    for name in ('uniform', 'population', 'hazard'):
        sel = pdes[(name, T)]; row = {}
        for ename, (q, hh, ww) in PRI.items():
            t4, _ = alarm_times(mid, sel, q, hh, vp); row[ename] = {'mean4': float((ww * t4).sum()), 'worst4': float(t4.max())}
        res['priors'][f'{name}_T{T}'] = row
        rows5.append(f"    {name.capitalize()} & {T} & {row['uniform']['mean4']:.2f} / {row['uniform']['worst4']:.2f} & {row['population']['mean4']:.2f} / {row['population']['worst4']:.2f} & {row['hazard']['mean4']:.1f} / {row['hazard']['worst4']:.1f} \\\\")
        print(f"prior {name:<10} T={T}: " + ' | '.join(f"{e}: {v['mean4']:.2f}/{v['worst4']:.2f}" for e, v in row.items()), flush=True)
    if T == 6: rows5.append("    \\midrule")
open(os.path.join(OUT, 'si_table_priors_body.tex'), 'w').write('\n'.join(rows5) + '\n')
res['hazard_weight_within_200km'] = float(w_haz[np.hypot(*(eq_haz - cen).T) < 200e3].sum())
fig = plt.figure(figsize=(7.0, 2.6))
axA = fig.add_axes([0.095, 0.17, 0.245, 0.70])
axA.scatter((mid[::50, 0] - cen[0]) / 1000, (mid[::50, 1] - cen[1]) / 1000, s=0.3, color='#9e9e9e', lw=0, rasterized=True)
sc_ = axA.scatter((eq_haz[:, 0] - cen[0]) / 1000, (eq_haz[:, 1] - cen[1]) / 1000, s=2 + 3 * (mag - 5) ** 2, c=np.log10(w_haz), cmap='YlOrRd', edgecolors='k', linewidths=0.2, zorder=5)
axA.set_aspect('equal'); axA.set_xlim(-520, 520); axA.set_ylim(-520, 520)
axA.set_xlabel('East of city centre (km)', fontsize=FS_LABEL); axA.set_ylabel('North (km)', fontsize=FS_LABEL); axA.tick_params(labelsize=FS_TICK)
axA.text(0.03, 0.97, f'M $\\geq$ 5, 1960--2026\n{len(cat)} events', transform=axA.transAxes, fontsize=FS_LEG, va='top')
cax = fig.add_axes([0.345, 0.17, 0.012, 0.70]); cb = fig.colorbar(sc_, cax=cax); cb.set_label('log$_{10}$ harm weight', fontsize=FS_LEG); cb.ax.tick_params(labelsize=FS_TICK)
fig.text(0.01, 0.93, 'A', fontsize=FS_PANEL, fontweight='bold', ha='left', va='bottom')
for k, (pname, title) in enumerate((('uniform', 'Uniform prior'), ('population', 'Population prior'), ('hazard', 'Hazard prior'))):
    ax = fig.add_axes([0.41 + k * 0.195, 0.04, 0.19, 0.86])
    orx.plot_pop_density(ax, edges, pop); orx.plot_city_base(ax, base_lines, color='#c0c0c0', lw=0.15, alpha=0.4)
    for i in pdes[(pname, 6)]:
        ax.add_patch(plt.Circle((edges[i]['mx'], edges[i]['my']), MIN_SPACING, fill=False, edgecolor='#1565C0', lw=0.6, ls='--', alpha=0.6, zorder=6))
        ax.scatter(edges[i]['mx'], edges[i]['my'], c='#1565C0', s=14, zorder=7, edgecolors='white', linewidths=0.5)
    ax.set_aspect('equal'); ax.axis('off')
    fig.text(0.41 + k * 0.195 + 0.095, 0.93, title, fontsize=FS_LABEL, ha='center', va='bottom')
    fig.text(0.41 + k * 0.195 + 0.005, 0.93, 'BCD'[k], fontsize=FS_PANEL, fontweight='bold', ha='left', va='bottom')
rasterize(fig); fig.savefig(os.path.join(PAPER, 'fig_eew_priors.pdf'), dpi=300); fig.savefig(os.path.join(OUT, 'fig_eew_priors.png'), dpi=160); plt.close(fig)
print('priors done', flush=True)

# ---------------------------------------------------------------- warning time (Table S12, Fig. S9, warned map)
def offset(dist_km, bearing_deg):
    b = np.radians(bearing_deg); return cen + 1000 * dist_km * np.array([np.sin(b), np.cos(b)])
WSCEN = {'local-10': dict(eqs=eq_pop, h=H_PRIMARY, vp=VP_PRIMARY, vs=VS_PRIMARY, sc='P'),
         'local-20': dict(eqs=eq_pop, h=H_SENS, vp=VP_SENS, vs=VS_SENS, sc='S'),
         'inslab': dict(eqs=offset(120, 150)[None, :], h=50e3, vp=7e3, vs=4e3, sc='S'),
         'interface': dict(eqs=offset(300, 200)[None, :], h=20e3, vp=7e3, vs=4e3, sc='S')}
rand6 = [random_spaced(mid, 6, rng) for _ in range(100)]


def wstats(sensors_idx, sc, tlat, near_source=False, every=False):
    h, vp, vs = sc['h'], sc['vp'], sc['vs']; out = {k: [] for k in ('t_det', 'f_pos', 'f_5', 'f_10', 'median', 'blind_km')}
    for q in sc['eqs']:
        if near_source: td = np.sqrt(20e3 ** 2 + h ** 2) / vp
        elif every: td = np.sqrt(np.partition(((mid - q) ** 2).sum(1), M_ALARM - 1)[:M_ALARM].max() + h ** 2) / vp
        else: td = float(alarm_times(mid, sensors_idx, q[None, :], h, vp)[0][0])
        tS = np.sqrt(((mid - q) ** 2).sum(1) + h ** 2) / vs; warn = tS - td - tlat
        o = np.argsort(warn); cw = np.cumsum(wpop[o])
        out['t_det'].append(td); out['f_pos'].append(float(wpop[warn > 0].sum())); out['f_5'].append(float(wpop[warn >= 5].sum())); out['f_10'].append(float(wpop[warn >= 10].sum()))
        out['median'].append(float(warn[o][np.searchsorted(cw, 0.5)])); rb2 = (vs * (td + tlat)) ** 2 - h ** 2; out['blind_km'].append(float(np.sqrt(max(rb2, 0)) / 1000))
    return {k: float(np.mean(v)) for k, v in out.items()}


res['warning'] = {}; rows12 = []
LAB = {'local-10': 'Beneath city, $h$ = 10 km', 'local-20': 'Beneath city, $h$ = 20 km', 'inslab': 'Intraslab, 120 km, $h$ = 50 km', 'interface': 'Interface, 300 km, $h$ = 20 km'}
for sname, sc in WSCEN.items():
    res['warning'][sname] = {}
    nets = [(f'Greedy, $T={T}$', designs[(sc['sc'], T)], {}) for T in (6, 10, 20)]
    first = True
    for lab, sel, kw in nets + [('Random, $T=6$', None, {}), ('Every street', None, {'every': True})] + ([('Source region', None, {'near_source': True})] if sname in ('inslab', 'interface') else []):
        vals = {}
        for tl in (T_LAT_BASE, T_LAT_PESS):
            if lab.startswith('Random'):
                L = [wstats(r, sc, tl) for r in rand6]; vals[tl] = {k: float(np.mean([x[k] for x in L])) for k in L[0]}
            else:
                vals[tl] = wstats(sel, sc, tl, **kw)
        res['warning'][sname][lab] = {str(k): v for k, v in vals.items()}
        v1, v3 = vals[T_LAT_BASE], vals[T_LAT_PESS]
        rows12.append(f"    {LAB[sname] if first else ''} & {lab} & {v1['t_det']:.1f} & {v1['median']:.1f} / {v3['median']:.1f} & {v1['f_pos']*100:.0f} / {v3['f_pos']*100:.0f} & {v1['f_5']*100:.0f} / {v3['f_5']*100:.0f} & {v1['blind_km']:.0f} / {v3['blind_km']:.0f} \\\\")
        first = False
        print(f"{sname:<10} {lab:<16} tdet {v1['t_det']:5.1f}  median {v1['median']:5.1f}/{v3['median']:5.1f}  pos {v1['f_pos']*100:3.0f}/{v3['f_pos']*100:3.0f}%  >=5s {v1['f_5']*100:3.0f}/{v3['f_5']*100:3.0f}%  blind {v1['blind_km']:4.1f}/{v3['blind_km']:4.1f} km", flush=True)
    rows12.append("    \\midrule")
open(os.path.join(OUT, 'si_table_s12_body.tex'), 'w').write('\n'.join(rows12[:-1]) + '\n')

# Fig. S9: A median warning vs distance (primary scenario, 1 s latency); B warned share vs latency (local-10)
h, vp, vs = SCEN['P']
D = np.arange(0, 301, 10); b = np.radians(200); curves = {'Urban DAS, greedy $T$ = 6': [], 'Urban DAS, every street': [], 'Source-region network': []}
def med_warn(q, td, tl):
    tS = np.sqrt(((mid - q) ** 2).sum(1) + h ** 2) / vs; o = np.argsort(tS); cw = np.cumsum(wpop[o]); return tS[o][np.searchsorted(cw, 0.5)] - td - tl
for dk in D:
    q = offset(dk, 200)
    curves['Urban DAS, greedy $T$ = 6'].append(med_warn(q, float(alarm_times(mid, designs[('P', 6)], q[None, :], h, vp)[0][0]), T_LAT_BASE))
    curves['Urban DAS, every street'].append(med_warn(q, np.sqrt(np.partition(((mid - q) ** 2).sum(1), 3)[:4].max() + h ** 2) / vp, T_LAT_BASE))
    curves['Source-region network'].append(med_warn(q, np.sqrt(20e3 ** 2 + h ** 2) / vp, T_LAT_BASE))
LAT = np.arange(0, 6.01, 0.25); pre = []
for q in eq_pop:
    tS = np.sqrt(((mid - q) ** 2).sum(1) + h ** 2) / vs; o = np.argsort(tS); pre.append((tS[o], np.cumsum(wpop[o])))
def frac_pos(tdets, lat):
    v = []
    for (tS, cw), td in zip(pre, tdets):
        i = np.searchsorted(tS, td + lat, 'right'); v.append(1.0 - (cw[i - 1] if i > 0 else 0.0))
    return float(np.mean(v))
tsets = {f'Greedy design, $T$ = {T}': alarm_times(mid, designs[('P', T)], eq_pop, h, vp)[0] for T in (6, 20)}
tsets['Every street'] = np.array([np.sqrt(np.partition(((mid - q) ** 2).sum(1), 3)[:4].max() + h ** 2) / vp for q in eq_pop])
rtd = [alarm_times(mid, r, eq_pop, h, vp)[0] for r in rand6[:50]]
fracs = {k: [frac_pos(v, l) for l in LAT] for k, v in tsets.items()}; fracs['Random, $T$ = 6'] = [float(np.mean([frac_pos(v, l) for v in rtd])) for l in LAT]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
cols = {'Urban DAS, greedy $T$ = 6': '#1565C0', 'Urban DAS, every street': '#64b5f6', 'Source-region network': '#555555'}
for k, v in curves.items(): a1.plot(D, v, '-' if 'Urban' in k else '--', color=cols[k], lw=1.4, label=k)
a1.axhline(0, color='k', lw=0.6); a1.set_xlabel('Epicentral distance from the city centre (km)', fontsize=FS_LABEL); a1.set_ylabel('Median warning time (s)', fontsize=FS_LABEL); a1.set_xlim(0, 300); a1.legend(fontsize=FS_LEG, loc='upper left', framealpha=0.92)
bcols = {'Greedy design, $T$ = 6': '#1565C0', 'Greedy design, $T$ = 20': '#0d47a1', 'Random, $T$ = 6': '#8a8a8a', 'Every street': '#64b5f6'}
for k, v in fracs.items(): a2.plot(LAT, np.array(v) * 100, color=bcols[k], lw=1.4, ls='--' if 'Random' in k else '-', label=k)
a2.axvline(T_LAT_BASE, color='k', lw=0.6, ls=':'); a2.set_xlabel('Detection, processing and alert latency (s)', fontsize=FS_LABEL); a2.set_ylabel('Population with positive warning (%)', fontsize=FS_LABEL); a2.set_xlim(0, 6); a2.set_ylim(0, 102); a2.legend(fontsize=FS_LEG, loc='lower left', framealpha=0.92)
for ax, lab in zip((a1, a2), 'AB'):
    ax.grid(True, alpha=0.22, lw=0.5); ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.6)
    for s_ in ax.spines.values(): s_.set_linewidth(0.7)
    ax.text(-0.15, 1.08, lab, transform=ax.transAxes, fontsize=FS_PANEL, fontweight='bold', va='top')
plt.tight_layout(pad=0.4, w_pad=2.0); fig.savefig(os.path.join(PAPER, 'fig_eew_warning.pdf'), dpi=300); fig.savefig(os.path.join(OUT, 'fig_eew_warning.png'), dpi=160); plt.close(fig)
res['warning_curves'] = {'D_km': D.tolist(), 'median': {k: [float(x) for x in v] for k, v in curves.items()}, 'lat': LAT.tolist(), 'frac_pos': fracs}
# warned map (T = 6, primary, 1 s)
cell = 1000.0; bid = np.floor((mid - mid.min(0)) / cell).astype(int); key = bid[:, 0] * 10000 + bid[:, 1]
uk, inv = np.unique(key, return_inverse=True); cpop = np.bincount(inv, weights=pop); cxy = np.c_[np.bincount(inv, weights=mid[:, 0]) / np.bincount(inv), np.bincount(inv, weights=mid[:, 1]) / np.bincount(inv)]; cw_ = cpop / cpop.sum()
pad2 = (hi - lo).max() * 0.15; xi2 = np.linspace(lo[0] - pad2, hi[0] + pad2, 90); yi2 = np.linspace(lo[1] - pad2, hi[1] + pad2, 90); X2, Y2 = np.meshgrid(xi2, yi2); g2 = np.c_[X2.ravel(), Y2.ravel()]
D4 = alarm_times(mid, designs[('P', 6)], g2, h, vp)[0]; tS = np.sqrt(cdist(g2, cxy) ** 2 + h ** 2) / vs; warned = ((tS - D4[:, None] - T_LAT_BASE) > 0) @ cw_
fig, ax = plt.subplots(figsize=(3.4, 3.2))
im = ax.imshow(warned.reshape(90, 90) * 100, extent=(xi2[0], xi2[-1], yi2[0], yi2[-1]), origin='lower', cmap='viridis', vmin=0, vmax=100, interpolation='bilinear')
orx.plot_city_base(ax, base_lines, color='white', lw=0.12, alpha=0.5)
for i in designs[('P', 6)]: ax.scatter(edges[i]['mx'], edges[i]['my'], c='white', s=12, zorder=7, edgecolors='k', linewidths=0.5)
ax.set_aspect('equal'); ax.axis('off'); cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label('Population warned before S-wave arrival (%)', fontsize=7); cb.ax.tick_params(labelsize=7)
rasterize(fig); fig.savefig(os.path.join(PAPER, 'fig_eew_warned_map.pdf'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'fig_eew_warned_map.png'), dpi=160, bbox_inches='tight'); plt.close(fig)
res['warned_map_mean'] = float(warned.mean()); res['warned_map_min'] = float(warned.min())
print('warning done', flush=True)

# ---------------------------------------------------------------- partial access (Fig. S8C), primary scenario
h, vp, vs = SCEN['P']
BLOCK_L = 2000.0; bid = np.floor((mid - mid.min(0)) / BLOCK_L).astype(np.int64); key = bid[:, 0] * 100000 + bid[:, 1]
ukeys, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
def access_mask(a, model):
    k = max(1, int(round(a * m)))
    if a >= 1: return np.ones(m, bool)
    if model == 'street':
        mk = np.zeros(m, bool); mk[rng.choice(m, k, replace=False)] = True; return mk
    order = rng.permutation(len(ukeys)); cum = np.cumsum(counts[order]); nb = int(np.searchsorted(cum, k)) + 1
    chosen = np.zeros(len(ukeys), bool); chosen[order[:nb]] = True; return chosen[inv]
KAPPAS = [1, 1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 1e4, 2e4]
T_path = os.path.join(OUT, 'tasks_two_edges.json'); TASKS = json.load(open(T_path)) if os.path.exists(T_path) else []; TASKS = [r for r in TASKS if r['task'] != 'eew']
def metr(sel):
    t4, t1 = alarm_times(mid, sel, eq_pop, h, vp); return {'t4_mean_s': float((w_pop * t4).sum()), 'worst4_s': float(t4.max()), 'worst_nearest_s': float(t1.max())}
for T in (6, 10):
    c = T / m; full = metr(designs[('P', T)]); rnd = [metr(r) for r in (rand6 if T == 6 else [random_spaced(mid, T, rng) for _ in range(100)])]
    agg = lambda L, k: (float(np.mean([x[k] for x in L])), float(np.std([x[k] for x in L])))
    out = {'city': 'Mexico City', 'task': 'eew', 'budget': T, 'm': m, 'c': c, 'full': full, 'random': {k: agg(rnd, k) for k in full}, 'partial': {}}
    for model in ('street', 'block'):
        rows = []
        for kap in KAPPAS:
            a = kap * c
            if a > 1: break
            if kap == 1 and model == 'street': rows.append((kap, {k: agg(rnd, k) for k in full}, 0)); continue
            L = [metr(design_mean_alarm(mid, eq_pop, w_pop, T, h, vp, allowed=access_mask(a, model))) for _ in range(10)]
            rows.append((kap, {k: agg(L, k) for k in full}, 0))
        out['partial'][model] = rows
    TASKS.append(out)
    print(f"partial T={T}: full {full['t4_mean_s']:.2f}s random {out['random']['t4_mean_s'][0]:.2f}s block@k<=100 {min(x[1]['t4_mean_s'][0] for x in out['partial']['block'] if x[0]<=100):.2f}-{max(x[1]['t4_mean_s'][0] for x in out['partial']['block'] if x[0]<=100):.2f}s", flush=True)
json.dump(TASKS, open(T_path, 'w'), indent=1)

# ---------------------------------------------------------------- architectures row (Table S9): long-range zones on major roads
hw, seen = [], set()
for u, v, k, d in G.edges(keys=True, data=True):
    ek = (min(u, v), max(u, v))
    if ek in seen: continue
    seen.add(ek); hh = d.get('highway', 'residential'); hw.append(hh[0] if isinstance(hh, list) else hh)
major = np.array([x in MAJOR for x in hw]); midx = np.where(major)[0]
pts = []
for i in np.random.default_rng(5).permutation(midx):
    if not pts or np.min(np.hypot(*(mid[pts] - mid[i]).T)) >= MIN_SPACING: pts.append(i)
lr = metr(np.array(pts)); g6 = metr(designs[('P', 6)]); hy = metr(np.array(list(pts) + list(designs[('P', 6)])))
res['architectures_eew'] = {'zones': len(pts), 'long_range': lr, 'greedy6': g6, 'hybrid6': hy}
print(f"arch EEW: zones {len(pts)} long-range mean4 {lr['t4_mean_s']:.2f}s greedy6 {g6['t4_mean_s']:.2f}s hybrid {hy['t4_mean_s']:.2f}s", flush=True)
json.dump(res, open(os.path.join(OUT, 'eew_round7.json'), 'w'), indent=1, default=float)
print('saved eew_round7.json')
