"""
Targeted panel generation:
  - Traffic panels for SF (50/250 km), Berlin & Singapore (proportionally scaled)
  - Activity panels for Paris with 600m radius
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimal_routing import (
    extract_edges, get_base_lines, load_population_per_edge,
    optimize_traffic_km, eval_traffic,
    optimize_activity, optimize_activity_km, eval_activity,
    plot_city_base, plot_selected_edges, plot_pop_density,
    ACTIVITY_COVERAGE_RADIUS, DAS_SENSING_RANGE,
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'city_cache')
PANEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'panels')

SF_KM_BUDGETS = [50, 250]
TRAFFIC_CITIES = ['San Francisco', 'Berlin', 'Singapore']
ACTIVITY_CITIES = ['Paris']

def load_city(name):
    safe = name.replace(' ', '_')
    path = os.path.join(CACHE_DIR, f'{safe}.pkl')
    with open(path, 'rb') as f:
        G_proj = pickle.load(f)
    edges = extract_edges(G_proj)
    base_lines = get_base_lines(G_proj)
    pop = load_population_per_edge(edges, G_proj, name)
    return edges, base_lines, pop

def run_traffic_all():
    city_data = {}
    for city in TRAFFIC_CITIES:
        edges, base_lines, pop = load_city(city)
        total_km = sum(e['length_m'] for e in edges) / 1000.0
        city_data[city] = (edges, base_lines, pop, total_km)
        print(f"  {city}: {len(edges)} edges, {total_km:.1f} km")

    sf_total = city_data['San Francisco'][3]
    print(f"\nReference: SF = {sf_total:.1f} km, budgets = {SF_KM_BUDGETS} km")

    for city in TRAFFIC_CITIES:
        edges, base_lines, pop, total_km = city_data[city]
        safe = city.replace(' ', '_')
        ratio = total_km / sf_total
        km_budgets = [round(b * ratio, 1) for b in SF_KM_BUDGETS]
        mt = sum(e['weight'] for e in edges)

        print(f"\n{city} (ratio={ratio:.2f}): budgets = {km_budgets} km")

        for km_budget in km_budgets:
            sel = optimize_traffic_km(edges, km_budget)
            km_used = sum(edges[i]['length_m'] for i in sel) / 1000.0
            tf = eval_traffic(edges, sel) / mt
            das_count = int(sum(np.ceil(edges[i]['length_m'] / DAS_SENSING_RANGE) for i in sel))
            pct_used = km_used / total_km * 100

            print(f"  {km_budget:.0f}km: {len(sel)} edges, "
                  f"{km_used:.1f}km used ({pct_used:.1f}%), {das_count} DAS, traffic={tf*100:.1f}%")

            fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
            plot_city_base(ax, base_lines, color='#999999', lw=0.5, alpha=0.7)

            if sel:
                weights = np.array([edges[i]['weight'] for i in sel])
                wmin, wmax = weights.min(), weights.max()
                if wmax > wmin:
                    norm = (weights - wmin) / (wmax - wmin)
                else:
                    norm = np.ones_like(weights)
                cmap = plt.get_cmap('YlOrRd')
                colors = [cmap(0.3 + 0.7 * v) for v in norm]
                plot_selected_edges(ax, edges, sel, colors, lw=2.5)

            ax.set_title(f'{city} — Traffic\n'
                         f'{len(sel)} edges, {km_used:.1f}/{km_budget:.0f} km, '
                         f'{das_count} DAS, traffic={tf*100:.1f}%',
                         fontsize=16, fontweight='bold', pad=12)
            ax.axis('off'); ax.set_aspect('equal')
            fname = os.path.join(PANEL_DIR, f'{safe}_traffic_{int(km_budget)}km.png')
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
            plt.close()
            print(f"    Saved {os.path.basename(fname)}")

def run_activity(city_name):
    safe = city_name.replace(' ', '_')
    edges, base_lines, pop = load_city(city_name)
    total_pop = pop.sum() if pop is not None else 1

    print(f"\n{city_name}: activity panels (radius={ACTIVITY_COVERAGE_RADIUS}m)")

    for T in [4, 6, 8, 10]:
        sel = optimize_activity(edges, T, pop)
        covered = eval_activity(edges, sel, pop)
        pop_frac = covered / total_pop
        km_used = sum(edges[i]['length_m'] for i in sel) / 1000.0

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
        plot_pop_density(ax, edges, pop)
        plot_city_base(ax, base_lines, color='#b0b0b0', lw=0.35, alpha=0.5)
        plot_selected_edges(ax, edges, sel, '#2E7D32', lw=3.5)
        for i in sel:
            circle = plt.Circle((edges[i]['mx'], edges[i]['my']),
                                ACTIVITY_COVERAGE_RADIUS, fill=True,
                                facecolor='#2E7D32', edgecolor='#2E7D32',
                                lw=1.2, alpha=0.08, zorder=4)
            ax.add_patch(circle)
            ax.scatter(edges[i]['mx'], edges[i]['my'],
                       c='#2E7D32', s=80, zorder=7, edgecolors='white', linewidths=1.2)
        ax.set_title(f'{city_name} — Activity Monitoring\n'
                     f'T={T} DAS, {km_used:.1f} km, '
                     f'pop covered={pop_frac*100:.1f}%',
                     fontsize=16, fontweight='bold', pad=12)
        ax.axis('off'); ax.set_aspect('equal')
        fname = os.path.join(PANEL_DIR, f'{safe}_activity_T{T}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        plt.close()
        print(f"  Saved T={T}: pop={pop_frac*100:.1f}%")

    for km in [1, 5, 10, 50]:
        sel = optimize_activity_km(edges, km, pop)
        covered = eval_activity(edges, sel, pop)
        pop_frac = covered / total_pop
        km_used = sum(edges[i]['length_m'] for i in sel) / 1000.0

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
        plot_pop_density(ax, edges, pop)
        plot_city_base(ax, base_lines, color='#b0b0b0', lw=0.35, alpha=0.5)
        plot_selected_edges(ax, edges, sel, '#2E7D32', lw=3.5)
        for i in sel:
            circle = plt.Circle((edges[i]['mx'], edges[i]['my']),
                                ACTIVITY_COVERAGE_RADIUS, fill=True,
                                facecolor='#2E7D32', edgecolor='#2E7D32',
                                lw=1.2, alpha=0.08, zorder=4)
            ax.add_patch(circle)
            ax.scatter(edges[i]['mx'], edges[i]['my'],
                       c='#2E7D32', s=80, zorder=7, edgecolors='white', linewidths=1.2)
        ax.set_title(f'{city_name} — Activity Monitoring\n'
                     f'{len(sel)} DAS, {km_used:.1f} km, '
                     f'pop covered={pop_frac*100:.1f}%',
                     fontsize=16, fontweight='bold', pad=12)
        ax.axis('off'); ax.set_aspect('equal')
        fname = os.path.join(PANEL_DIR, f'{safe}_activity_{km}km.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        plt.close()
        print(f"  Saved {km}km: pop={pop_frac*100:.1f}%")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run_traffic_all()

    for city in ACTIVITY_CITIES:
        run_activity(city)

    print("\nDone!")
