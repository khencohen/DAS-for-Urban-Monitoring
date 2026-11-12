"""Fig. 3, middle row (Mexico City early warning), round 12: panel (d) shows the alarm delay, i.e. the
P-wave arrival at the fourth segment minus the earliest possible arrival h / vP (a segment directly
above the source), as asked by A. Lellouch. Designs are the stored round-7 designs (eew_round7.json)."""
import os, sys, json, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
OUT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, OUT)
import optimal_routing as orx
from eew_common import H_PRIMARY as h, VP_PRIMARY as vp, MIN_SPACING, alarm_times
PAPER = os.path.join(OUT, 'figures'); os.makedirs(PAPER, exist_ok=True)

G = pickle.load(open(os.path.join(REPO, 'city_cache', 'Mexico_City.pkl'), 'rb'))
edges = orx.extract_edges(G); base_lines = orx.get_base_lines(G)
pop = np.maximum(orx.load_population_per_edge(edges, G, 'Mexico City'), 0).astype(float)
mid = np.array([[e['mx'], e['my']] for e in edges])
eq = orx.sample_earthquake_locations(edges, pop, orx.EEW_MC_SAMPLES, np.random.RandomState(42))
D = json.load(open(os.path.join(OUT, 'eew_round7.json')))['designs']
T_LIST = [4, 6, 8, 10]; designs = {T: D[f'P_T{T}'] for T in T_LIST}
lo, hi = mid.min(0), mid.max(0); pad = (hi - lo).max() * 0.35
xi = np.linspace(lo[0] - pad, hi[0] + pad, 200); yi = np.linspace(lo[1] - pad, hi[1] + pad, 200)
X, Y = np.meshgrid(xi, yi); gpts = np.c_[X.ravel(), Y.ravel()]; ext = (xi[0], xi[-1], yi[0], yi[-1])
delay = {T: alarm_times(mid, designs[T], gpts, h, vp)[0].reshape(200, 200) - h / vp for T in T_LIST}
for T in T_LIST:
    t4 = alarm_times(mid, designs[T], eq, h, vp)[0] - h / vp
    print(f'T={T}: population-weighted mean alarm delay {t4.mean():.2f} s, worst {t4.max():.2f} s')
fig = plt.figure(figsize=(841.89 / 72, 595.276 / 72))
fig.suptitle('Mexico City', fontsize=13, y=0.90)
w, hgt, x0, y0, gap = 0.19, 0.27, 0.05, 0.56, 0.005
for k, T in enumerate(T_LIST):
    ax = fig.add_axes([x0 + k * (w + gap), y0, w, hgt])
    orx.plot_pop_density(ax, edges, pop); orx.plot_city_base(ax, base_lines, color='#c0c0c0', lw=0.15, alpha=0.4)
    for i in designs[T]:
        ax.add_patch(plt.Circle((edges[i]['mx'], edges[i]['my']), MIN_SPACING, fill=False, edgecolor='#1565C0', lw=0.6, ls='--', alpha=0.6, zorder=6))
        ax.scatter(edges[i]['mx'], edges[i]['my'], c='#1565C0', s=12, zorder=7, edgecolors='white', linewidths=0.5)
    ax.set_aspect('equal'); ax.axis('off')
    if k == 0: ax.text(-0.08, 1.0, 'c', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
cax = fig.add_axes([x0 + 4 * (w + gap) + 0.01, y0 + 0.02, 0.012, hgt - 0.04])
sm = plt.cm.ScalarMappable(cmap='YlOrRd'); sm.set_array([0, 1]); cb = fig.colorbar(sm, cax=cax); cb.set_ticks([]); cb.set_label('Population Density', fontsize=8)
for k, T in enumerate(T_LIST):
    ax = fig.add_axes([x0 + k * (w + gap), y0 - hgt - 0.01, w, hgt])
    im = ax.imshow(delay[T], extent=ext, origin='lower', cmap='coolwarm', vmin=0, vmax=2.5, interpolation='bilinear')
    orx.plot_city_base(ax, base_lines, color='#555555', lw=0.12, alpha=0.5)
    for i in designs[T]:
        ax.scatter(edges[i]['mx'], edges[i]['my'], c='white', s=10, zorder=7, edgecolors='k', linewidths=0.5)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect('equal'); ax.axis('off')
    ax.text(0.5, -0.08, str(T), transform=ax.transAxes, fontsize=11, ha='center', va='top')
    if k == 0: ax.text(-0.08, 1.0, 'd', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
cax = fig.add_axes([x0 + 4 * (w + gap) + 0.01, y0 - hgt + 0.01, 0.012, hgt - 0.04])
cb = fig.colorbar(im, cax=cax, extend='max'); cb.set_label('Alarm delay beyond earliest possible (s)', fontsize=8); cb.ax.tick_params(labelsize=7)
fig.text(x0 + 4 * (w + gap) + 0.01, y0 - hgt - 0.045, 'Number of\nInterrogators', fontsize=9, va='top')
fig.patches.append(matplotlib.patches.FancyArrowPatch((x0, y0 - hgt - 0.045), (x0 + 4 * (w + gap), y0 - hgt - 0.045), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=10, color='k', lw=0.8))
for ax in fig.axes:
    for c in ax.collections: c.set_rasterized(True)
    for im_ in ax.images: im_.set_rasterized(True)
fig.savefig(os.path.join(PAPER, 'p2_v2.pdf'), dpi=300); fig.savefig(os.path.join(OUT, 'p2_v2.png'), dpi=110)
print('saved p2_v2')
