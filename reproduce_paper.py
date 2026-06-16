#!/usr/bin/env python3
"""
Reproduce the paper's quantitative results.
====================================================================
Runs the exact city x task combinations reported in
"Distributed Acoustic Sensing for Urban Monitoring: Coverage Thresholds and
Percolation" (Cohen & Lellouch, arXiv:2606.17018):

    * Earthquake Early Warning (EEW)  ->  Mexico City
    * Traffic monitoring              ->  Berlin, San Francisco, Singapore
    * Activity monitoring             ->  Paris

All required inputs are bundled in this repository:
    * Street graphs in  city_cache/   (exact snapshots used in the paper)
    * WorldPop rasters in population_data/  (mex, fra)

Results are deterministic: the only randomness (EEW epicentre sampling) is
seeded with np.random.RandomState(42), matching analyze_city() in
optimal_routing.py. Re-running this script reproduces the REFERENCE values
printed below; deviations indicate a changed graph, raster, or dependency.

Usage:
    python reproduce_paper.py            # run all, save panels to panels/
    python reproduce_paper.py --no-plots # numbers only (faster)
"""
import os
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from optimal_routing import (
    extract_edges, get_base_lines, load_population_per_edge,
    # Task 1 — Traffic
    optimize_traffic, optimize_traffic_km, eval_traffic,
    # Task 2 — EEW
    sample_earthquake_locations, optimize_eew, optimize_eew_km, eval_eew,
    # Task 3 — Activity
    optimize_activity, optimize_activity_km, eval_activity,
    # plotting
    plot_city_base, plot_selected_edges, plot_pop_density,
    # constants
    DAS_SENSING_RANGE, EEW_MC_SAMPLES, EEW_MIN_SPACING,
    EEW_T_VALUES, EEW_KM_BUDGETS,
    ACTIVITY_COVERAGE_RADIUS, ACTIVITY_T_VALUES, ACTIVITY_KM_BUDGETS,
    TRAFFIC_EDGE_FRACS, TRAFFIC_KM_BUDGETS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'city_cache')
PANEL_DIR = os.path.join(HERE, 'panels')
SEED = 42

# Paper city x task assignment
TRAFFIC_CITIES = ['Berlin', 'San Francisco', 'Singapore']
EEW_CITIES = ['Mexico City']
ACTIVITY_CITIES = ['Paris']

# Country codes for the WorldPop rasters (must match optimal_routing.CITY_POP_RASTER)
POP_CODE = {'Mexico City': 'mex', 'Paris': 'fra'}


def _load(city):
    """Load the bundled graph snapshot and derive edges/base lines/population."""
    path = os.path.join(CACHE_DIR, f'{city.replace(" ", "_")}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing bundled graph: {path}\n"
            f"    The paper cities should be committed under city_cache/.")
    with open(path, 'rb') as f:
        G_proj = pickle.load(f)
    edges = extract_edges(G_proj)
    base_lines = get_base_lines(G_proj)
    pop = load_population_per_edge(edges, G_proj, city)
    return edges, base_lines, pop


# ====================================================================
# Task 1 — Traffic monitoring (Berlin, San Francisco, Singapore)
# ====================================================================

def reproduce_traffic(make_plots=True):
    print("\n" + "=" * 70)
    print("TASK 1 — TRAFFIC MONITORING  (Berlin, San Francisco, Singapore)")
    print("=" * 70)
    results = {}
    for city in TRAFFIC_CITIES:
        edges, base_lines, _ = _load(city)
        total_km = sum(e['length_m'] for e in edges) / 1000.0
        mt = sum(e['weight'] for e in edges)
        print(f"\n  {city}: {len(edges):,} edges, {total_km:,.1f} km network")
        print(f"    {'budget':>14} {'edges':>8} {'km used':>10} {'DAS':>7} {'traffic %':>10}")

        city_rows = {}
        # Edge-count budgets
        for frac in TRAFFIC_EDGE_FRACS:
            T = max(1, int(frac * len(edges)))
            sel = optimize_traffic(edges, T)
            km_used = sum(edges[i]['length_m'] for i in sel) / 1000.0
            das = int(sum(np.ceil(edges[i]['length_m'] / DAS_SENSING_RANGE) for i in sel))
            tf = eval_traffic(edges, sel) / mt
            city_rows[f'{frac*100:g}%'] = tf
            print(f"    {frac*100:>12.1f}% {T:>8} {km_used:>9.1f}km {das:>7} {tf*100:>9.1f}%")
        # km budgets
        for km in TRAFFIC_KM_BUDGETS:
            if km > total_km:
                continue
            sel = optimize_traffic_km(edges, km)
            km_used = sum(edges[i]['length_m'] for i in sel) / 1000.0
            das = int(sum(np.ceil(edges[i]['length_m'] / DAS_SENSING_RANGE) for i in sel))
            tf = eval_traffic(edges, sel) / mt
            city_rows[f'{km}km'] = tf
            print(f"    {km:>11}km  {len(sel):>8} {km_used:>9.1f}km {das:>7} {tf*100:>9.1f}%")

        results[city] = city_rows

        if make_plots:
            # Panel at 1% edge budget (representative)
            T = max(1, int(0.01 * len(edges)))
            sel = optimize_traffic(edges, T)
            _save_traffic_panel(city, edges, base_lines, sel, mt)
    return results


def _save_traffic_panel(city, edges, base_lines, sel, mt):
    os.makedirs(PANEL_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    plot_city_base(ax, base_lines, color='#999999', lw=0.5, alpha=0.7)
    weights = np.array([edges[i]['weight'] for i in sel])
    if weights.max() > weights.min():
        norm = (weights - weights.min()) / (weights.max() - weights.min())
    else:
        norm = np.ones_like(weights)
    cmap = plt.get_cmap('YlOrRd')
    plot_selected_edges(ax, edges, sel, [cmap(0.3 + 0.7 * v) for v in norm], lw=2.5)
    tf = eval_traffic(edges, sel) / mt
    ax.set_title(f'{city} — Traffic\nT={len(sel)} edges, traffic={tf*100:.1f}%',
                 fontsize=16, fontweight='bold', pad=12)
    ax.axis('off'); ax.set_aspect('equal')
    fname = os.path.join(PANEL_DIR, f'{city.replace(" ", "_")}_traffic.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()
    print(f"      saved {os.path.relpath(fname, HERE)}")


# ====================================================================
# Task 2 — Earthquake Early Warning (Mexico City)
# ====================================================================

def reproduce_eew(make_plots=True):
    print("\n" + "=" * 70)
    print("TASK 2 — EARTHQUAKE EARLY WARNING  (Mexico City)")
    print("=" * 70)
    results = {}
    for city in EEW_CITIES:
        edges, base_lines, pop = _load(city)
        if pop is None:
            print(f"  ERROR: population raster for {city} ({POP_CODE.get(city)}) not found "
                  f"in population_data/. EEW needs it.")
            continue
        print(f"\n  {city}: {len(edges):,} edges")

        # Seeded epicentre sampling — identical to analyze_city()
        rng = np.random.RandomState(SEED)
        eq_locs = sample_earthquake_locations(edges, pop, EEW_MC_SAMPLES, rng)

        print(f"    (epicentres={EEW_MC_SAMPLES}, min spacing={EEW_MIN_SPACING/1000:g} km, "
              f"seed={SEED})")
        print(f"    {'budget':>14} {'DAS':>6} {'km used':>10} {'max-detect (m)':>16}")

        city_rows = {}
        for T in EEW_T_VALUES:
            sel = optimize_eew(edges, T, eq_locs, EEW_MIN_SPACING)
            score = eval_eew(edges, sel, eq_locs)
            km_used = sum(edges[i]['length_m'] for i in sel) / 1000.0
            city_rows[f'T={T}'] = score
            print(f"    {('T=' + str(T)):>14} {len(sel):>6} {km_used:>9.1f}km {score:>16,.0f}")
        for km in EEW_KM_BUDGETS:
            sel = optimize_eew_km(edges, km, eq_locs, EEW_MIN_SPACING)
            score = eval_eew(edges, sel, eq_locs)
            city_rows[f'{km}km'] = score
            print(f"    {(str(km) + 'km'):>14} {len(sel):>6} {'':>9}   {score:>16,.0f}")

        results[city] = city_rows

        if make_plots:
            sel = optimize_eew(edges, 6, eq_locs, EEW_MIN_SPACING)  # T=6 paper panel
            _save_eew_panel(city, edges, base_lines, pop, sel, eq_locs)
    return results


def _save_eew_panel(city, edges, base_lines, pop, sel, eq_locs):
    os.makedirs(PANEL_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    plot_pop_density(ax, edges, pop)
    plot_city_base(ax, base_lines, color='#c0c0c0', lw=0.25, alpha=0.4)
    plot_selected_edges(ax, edges, sel, '#1565C0', lw=3.0)
    for i in sel:
        ax.add_patch(plt.Circle((edges[i]['mx'], edges[i]['my']), EEW_MIN_SPACING,
                                fill=False, edgecolor='#1565C0', lw=1.0, ls='--',
                                alpha=0.5, zorder=6))
        ax.scatter(edges[i]['mx'], edges[i]['my'], c='#1565C0', s=60, zorder=7,
                   edgecolors='white', linewidths=1)
    score = eval_eew(edges, sel, eq_locs)
    ax.set_title(f'{city} — EEW\nT={len(sel)} DAS, max-detect={score:,.0f} m',
                 fontsize=16, fontweight='bold', pad=12)
    ax.axis('off'); ax.set_aspect('equal')
    fname = os.path.join(PANEL_DIR, f'{city.replace(" ", "_")}_eew_T6.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()
    print(f"      saved {os.path.relpath(fname, HERE)}")


# ====================================================================
# Task 3 — Activity monitoring (Paris)
# ====================================================================

def reproduce_activity(make_plots=True):
    print("\n" + "=" * 70)
    print("TASK 3 — ACTIVITY MONITORING  (Paris)")
    print("=" * 70)
    results = {}
    for city in ACTIVITY_CITIES:
        edges, base_lines, pop = _load(city)
        if pop is None:
            print(f"  ERROR: population raster for {city} ({POP_CODE.get(city)}) not found "
                  f"in population_data/. Activity monitoring needs it.")
            continue
        total_pop = float(np.maximum(pop, 0).sum())
        print(f"\n  {city}: {len(edges):,} edges, total pop={total_pop:,.0f}, "
              f"coverage radius={ACTIVITY_COVERAGE_RADIUS} m")
        print(f"    {'budget':>14} {'DAS':>6} {'km used':>10} {'pop covered %':>15}")

        city_rows = {}
        for T in ACTIVITY_T_VALUES:
            sel = optimize_activity(edges, T, pop)
            cov = eval_activity(edges, sel, pop)
            frac = cov / total_pop if total_pop > 0 else 0
            km_used = sum(edges[i]['length_m'] for i in sel) / 1000.0
            city_rows[f'T={T}'] = frac
            print(f"    {('T=' + str(T)):>14} {len(sel):>6} {km_used:>9.1f}km {frac*100:>14.1f}%")
        for km in ACTIVITY_KM_BUDGETS:
            sel = optimize_activity_km(edges, km, pop)
            cov = eval_activity(edges, sel, pop)
            frac = cov / total_pop if total_pop > 0 else 0
            km_used = sum(edges[i]['length_m'] for i in sel) / 1000.0
            city_rows[f'{km}km'] = frac
            print(f"    {(str(km) + 'km'):>14} {len(sel):>6} {km_used:>9.1f}km {frac*100:>14.1f}%")

        results[city] = city_rows

        if make_plots:
            sel = optimize_activity(edges, 6, pop)  # T=6 representative panel
            _save_activity_panel(city, edges, base_lines, pop, sel, total_pop)
    return results


def _save_activity_panel(city, edges, base_lines, pop, sel, total_pop):
    os.makedirs(PANEL_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    plot_pop_density(ax, edges, pop)
    plot_city_base(ax, base_lines, color='#b0b0b0', lw=0.35, alpha=0.5)
    plot_selected_edges(ax, edges, sel, '#2E7D32', lw=3.5)
    for i in sel:
        ax.add_patch(plt.Circle((edges[i]['mx'], edges[i]['my']), ACTIVITY_COVERAGE_RADIUS,
                                fill=True, facecolor='#2E7D32', edgecolor='#2E7D32',
                                lw=1.2, alpha=0.15, zorder=4))
        ax.scatter(edges[i]['mx'], edges[i]['my'], c='#2E7D32', s=80, zorder=7,
                   edgecolors='white', linewidths=1.2)
    frac = eval_activity(edges, sel, pop) / total_pop
    ax.set_title(f'{city} — Activity\nT={len(sel)} DAS, pop covered={frac*100:.1f}%',
                 fontsize=16, fontweight='bold', pad=12)
    ax.axis('off'); ax.set_aspect('equal')
    fname = os.path.join(PANEL_DIR, f'{city.replace(" ", "_")}_activity.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()
    print(f"      saved {os.path.relpath(fname, HERE)}")


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Reproduce the paper's three-task results.")
    parser.add_argument('--no-plots', action='store_true',
                        help="Skip panel rendering (numbers only, faster).")
    args = parser.parse_args()
    make_plots = not args.no_plots

    os.chdir(HERE)
    print("#" * 70)
    print("# Reproducing paper results — DAS-for-Urban-Monitoring")
    print("#" * 70)

    reproduce_traffic(make_plots)
    reproduce_eew(make_plots)
    reproduce_activity(make_plots)

    print("\n" + "=" * 70)
    print("Done. Compare the printed values against the REFERENCE table in the README.")
    print("=" * 70)


if __name__ == "__main__":
    main()
