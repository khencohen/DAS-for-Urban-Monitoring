"""Figure 2 in two versions for the authors to choose from: distance in hops (as in the current figure)
and in metres (80 m per hop, Methods). Eq. 3 of the paper: d = sqrt(log_{1-c}(1-P) / 2), P = 90%."""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.join(HERE, 'figures'); os.makedirs(PAPER, exist_ok=True)
P, M_PER_HOP, CSTAR = 0.9, 80.0, 0.516
c = np.concatenate([np.linspace(1e-5, 0.01, 400), np.linspace(0.01, 0.99, 2000), 1 - np.logspace(-2, -12, 600)])
d = np.sqrt(np.log(1 - P) / np.log(1 - c) / 2)
c = np.append(c, 1.0); d = np.append(d, 0.03)                # the curve reaches c = 100% as d -> 0
dstar = np.sqrt(np.log(1 - P) / np.log(1 - CSTAR) / 2)
BANDS = [  # (lo, hi, colour, label)
    (10, 25, '#c9d6ea', 'Earthquake Early Warning'), (7, 12, '#d9cba3', 'Geological Units Mapping'),
    (5, 10, '#f6c77d', 'Groundwater Monitoring'), (3, 8, '#dcebd3', 'Urban Activity Tracking'),
    (0.1, dstar, '#f9bdbd', None), (0.05, 0.1, '#d5c3e3', 'Infrastructure Monitoring'), (0.035, 0.05, '#c8c8c8', 'Individual Vehicle Tracking / Pedestrian Monitoring')]
LABEL_Y = {'Earthquake Early Warning': 16, 'Geological Units Mapping': 11.0, 'Groundwater Monitoring': 7.6, 'Urban Activity Tracking': 4.2}

def fmt_m(v):
    return (f'{v/1000:g} km' if v >= 1000 else f'{v:g} m')

def draw(unit):
    k = 1.0 if unit == 'hops' else M_PER_HOP
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    for lo, hi, col, lab in BANDS:
        ax.axhspan(lo * k, hi * k, color=col, alpha=0.75, lw=0, zorder=1)
        if lab is None: continue
        if unit == 'hops': rng = f'({lo:g}-{hi:g})' if lo >= 0.1 else f'(0-{hi:g})'
        else: rng = f'({fmt_m(lo * k)} - {fmt_m(hi * k)})' if lo >= 0.1 else f'(0 - {fmt_m(hi * k)})'
        y = LABEL_Y.get(lab, np.sqrt(lo * hi)) * k
        ax.text(51, y, f'{lab} {rng}', ha='center', va='center', fontsize=12.5 if lo >= 0.05 else 10.5, zorder=4)
    ax.text(51, 0.4 * k, r'City Scale Monitoring ($c > c^{\star}$)', ha='center', va='center', fontsize=12.5, zorder=4)
    ax.plot(c * 100, d * k, color='#1f77b4', lw=2.4, zorder=5)
    ax.plot(CSTAR * 100, dstar * k, 'o', color='red', mec='k', mew=1, ms=8, zorder=6)
    ax.text(CSTAR * 100 - 0.5, dstar * k * 1.25, 'Transition Point', fontsize=8.5, fontweight='bold', zorder=6)
    ax.set_yscale('log'); ax.set_xlim(0, 102); ax.set_ylim(0.035 * k, 35 * k)
    ax.set_xticks(range(0, 101, 10)); ax.set_xlabel('Coverage Ratio (%)', fontsize=13)
    ticks = [0.1, 1, dstar, 3, 5, 7, 8, 10, 12, 25]
    ax.set_yticks([t * k for t in ticks]); ax.minorticks_off()
    if unit == 'hops':
        ax.set_yticklabels(['0.1', '1', f'{dstar:.2f}', '3', '5', '7', '8', '10', '12', '25'], fontsize=8.5); ax.set_ylabel('Critical Distance  d  (hops)', fontsize=13)
    else:
        ax.set_yticklabels([fmt_m(round(t * k)) for t in ticks], fontsize=8.5); ax.set_ylabel('Critical Distance  (m)', fontsize=13)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    name = f'fig2_d_vs_c_{unit}'
    for dd in (HERE, PAPER): fig.savefig(os.path.join(dd, name + '.pdf'))
    fig.savefig(os.path.join(HERE, name + '.png'), dpi=110); plt.close(fig)
def draw_dual():
    """Meters on the left axis, hops on the right axis (the version used in the paper)."""
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    for lo, hi, col, lab in BANDS:
        ax.axhspan(lo, hi, color=col, alpha=0.75, lw=0, zorder=1)
        if lab is None: continue
        rng = f'($d$ = {lo:g}-{hi:g})' if lo >= 0.1 else f'($d$ = 0-{hi:g})'
        ax.text(51, LABEL_Y.get(lab, np.sqrt(lo * hi)), f'{lab} {rng}', ha='center', va='center', fontsize=12.5 if lo >= 0.05 else 10.5, zorder=4)
    ax.text(51, 0.4, r'City Scale Monitoring ($c > c^{\star}$)', ha='center', va='center', fontsize=12.5, zorder=4)
    ax.plot(c * 100, d, color='#1f77b4', lw=2.4, zorder=5)
    ax.plot(CSTAR * 100, dstar, 'o', color='red', mec='k', mew=1, ms=8, zorder=6)
    ax.text(CSTAR * 100 - 0.5, dstar * 1.25, 'Transition Point', fontsize=8.5, fontweight='bold', zorder=6)
    ax.set_yscale('log'); ax.set_xlim(0, 102); ax.set_ylim(0.035, 35)
    ax.set_xticks(range(0, 101, 10)); ax.set_xlabel('Coverage Ratio (%)', fontsize=13)
    ticks = [0.1, 1, dstar, 3, 5, 7, 8, 10, 12, 25]
    ax.set_yticks(ticks); ax.minorticks_off()
    ax.set_yticklabels(['0.1', '1', f'{dstar:.2f}', '3', '5', '7', '8', '10', '12', '25'], fontsize=8.5)
    ax.set_ylabel('Critical Distance  d  (hops)', fontsize=13)
    ax.yaxis.tick_right(); ax.yaxis.set_label_position('right')
    ax2 = ax.secondary_yaxis('left', functions=(lambda x: x * M_PER_HOP, lambda x: x / M_PER_HOP))
    mt = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    ax2.set_yticks(mt); ax2.set_yticklabels([fmt_m(v) for v in mt], fontsize=8.5); ax2.minorticks_off()
    ax2.set_ylabel('Critical Distance  (80 m per hop)', fontsize=13)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    for dd in (HERE, PAPER): fig.savefig(os.path.join(dd, 'fig2_d_vs_c.pdf'))
    fig.savefig(os.path.join(HERE, 'fig2_d_vs_c.png'), dpi=110); plt.close(fig)

if __name__ == '__main__':
    draw_dual(); print('saved fig2_d_vs_c (paper version); draw("hops") and draw("metres") give the single-axis alternatives')

