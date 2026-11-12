"""Round 9 figures.

fig_placement_tasks.pdf  : main-text Fig. 4, traffic (A) and activity (B) versus coverage,
                           mean +/- s.d. over ten cities, three placements.
fig_sparse_cities.pdf    : SI figure, Fig. 2 axes (linear c, log d) with the Monte Carlo
                           random and designed placement curves of ten real cities.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'); os.makedirs(PAPER, exist_ok=True)
R = json.load(open(os.path.join(HERE, 'tasks_many_cities.json')))
S = json.load(open(os.path.join(HERE, 'sparse_two_edges.json')))

FS_LABEL, FS_TICK, FS_LEG, FS_PANEL = 8, 7, 6, 10
GRAY, BLUE, ORANGE = '#8a8a8a', '#1f5fa8', '#e07b00'
FIG2_BLUE = '#1f77b4'
D = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]


def style(ax, lab):
    ax.grid(True, alpha=0.22, lw=0.5, which='major')
    ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.6)
    ax.tick_params(which='minor', length=1.5, width=0.4)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    if lab:
        ax.text(-0.2, 1.08, lab, transform=ax.transAxes, fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')


def stats(task):
    rows = [r for r in R if r['task'] == task]
    cs = [x['c'] for x in rows[0]['rows']]
    out = {}
    for key in ('random', 'partial', 'designed'):
        arr = np.array([[ (x[key][0] if isinstance(x[key], list) else x[key]) for x in r['rows']] for r in rows]) * 100
        out[key] = (arr.mean(0), arr.std(0))
    return np.array(cs) * 100, out, len(rows)


def main_figure():
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4))
    specs = [('traffic', 'Traffic captured (%)', 'A'), ('activity', 'Population covered (%)', 'B')]
    series = (('random', GRAY, 'o', 'Random placement'),
              ('partial', ORANGE, 's', '1% of streets designed,\nthe rest random'),
              ('designed', BLUE, 'D', 'Designed placement'))
    for ax, (task, ylab, lab) in zip(axes, specs):
        c, st, n = stats(task)
        for key, col, mk, name in series:
            m, s = st[key]
            if key == 'partial':
                ax.errorbar(c, m, yerr=s, color=col, lw=1.2, ls='--', marker=mk, ms=3.5, mfc='white', mew=0.9,
                            capsize=1.5, elinewidth=0.7, label=name, zorder=5)
            else:
                ax.errorbar(c, m, yerr=s, color=col, lw=1.4, marker=mk, ms=3, mew=0, capsize=1.5,
                            elinewidth=0.7, label=name, zorder=3 if key == 'designed' else 2)
        ax.set_xscale('log')
        ax.set_xlabel('Coverage ratio $c$ (%)', fontsize=FS_LABEL)
        ax.set_ylabel(ylab, fontsize=FS_LABEL)
        ax.set_xlim(0.007, 140); ax.set_ylim(0, 105)
        ax.set_xticks([0.01, 0.1, 1, 10, 100]); ax.set_xticklabels(['0.01', '0.1', '1', '10', '100'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        style(ax, lab)
    axes[0].legend(fontsize=FS_LEG, loc='upper left', frameon=False, handlelength=1.6, labelspacing=0.5)
    plt.tight_layout(pad=0.4, w_pad=1.6)
    for d in (HERE, PAPER):
        fig.savefig(os.path.join(d, 'fig_placement_tasks.pdf'), dpi=300)
    fig.savefig(os.path.join(HERE, 'fig_placement_tasks.png'), dpi=170)
    plt.close(fig)


def band(key, P='0.9'):
    arr = np.array([[ (S[c][key][str(d)][P][0] if key == 'random' else S[c][key][str(d)].get(P, np.nan))
                      for d in D] for c in S], float)
    return np.nanmedian(arr, 0), np.nanmin(arr, 0), np.nanmax(arr, 0)


def si_figure():
    """Same axes as Fig. 2 of the main text: linear coverage (%) and logarithmic d."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    P = 0.9
    cc = np.concatenate([np.linspace(1e-4, 0.99, 3000), np.linspace(0.99, 0.999999, 500)])
    dd_theory = np.sqrt(np.log(1 - P) / np.log(1 - cc) / 2)      # Eq. 3: P = 1-(1-c)^(2 d^2)
    ax.plot(cc * 100, dd_theory, color=FIG2_BLUE, lw=2.0, label='Eq. 3, $\\alpha = 2$ (Figure 2)', zorder=3)
    cstar = 0.516; ax.plot(cstar * 100, np.sqrt(np.log(1 - P) / np.log(1 - cstar) / 2), 'o', color='red', mec='k', mew=0.6, ms=5, zorder=5)
    dd = np.array(D, float)
    for key, col, lab in (('random', GRAY, 'Random placement, ten cities'), ('designed', ORANGE, 'Designed placement, ten cities')):
        med, mn, mx = band(key)
        ax.fill_betweenx(dd, mn * 100, mx * 100, color=col, alpha=0.22, lw=0)
        ax.plot(med * 100, dd, 'o-', color=col, lw=1.3, ms=3, label=lab)
    ax.set_yscale('log')
    ax.set_xlim(0, 100); ax.set_ylim(0.08, 30)
    ax.set_xticks(range(0, 101, 10))
    ax.set_yticks([0.1, 1, 3, 5, 10, 25]); ax.set_yticklabels(['0.1', '1', '3', '5', '10', '25'])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Coverage ratio $c$ (%)', fontsize=FS_LABEL)
    ax.set_ylabel('Critical distance $d$ (hops)', fontsize=FS_LABEL)
    ax.legend(fontsize=FS_LEG + 0.5, loc='upper right', frameon=False)
    style(ax, None)
    plt.tight_layout(pad=0.4)
    for d in (HERE, PAPER):
        fig.savefig(os.path.join(d, 'fig_sparse_cities.pdf'), dpi=300)
    fig.savefig(os.path.join(HERE, 'fig_sparse_cities.png'), dpi=170)
    plt.close(fig)


def numbers():
    for task in ('traffic', 'activity'):
        c, st, n = stats(task)
        print(task, n, 'cities')
        for i, cv in enumerate(c):
            r, p, d = st['random'][0][i], st['partial'][0][i], st['designed'][0][i]
            gain = (p - r) / (d - r) if d > r else np.nan
            print(f"  c={cv:6.2f}%  random {r:5.1f}+-{st['random'][1][i]:4.1f}  partial {p:5.1f}+-{st['partial'][1][i]:4.1f}"
                  f"  designed {d:5.1f}+-{st['designed'][1][i]:4.1f}  ratio {d/r:4.1f}  partial gain {gain*100:4.0f}%")


if __name__ == '__main__':
    main_figure()
    si_figure()
    numbers()
