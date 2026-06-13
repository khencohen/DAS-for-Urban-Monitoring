# DAS-for-Urban-Monitoring

Optimal placement of **Distributed Acoustic Sensing (DAS)** interrogators on existing
urban fiber-optic networks, for three city-scale sensing tasks. This repository
accompanies the paper *"Percolation Thresholds for Urban Sensing with Existing
Fiber-Optics Networks"* (Khen Cohen).

Existing telecom fiber under city streets can be repurposed as a dense array of
acoustic sensors. Given a limited sensing **budget** — a number of interrogators, or
a length of fiber in km — *which street segments should be instrumented?* The answer
depends on what you want to sense. This code formulates and greedily solves three
distinct placement problems on real street networks downloaded from OpenStreetMap.

---

## The three tasks

| # | Task | Objective | Optimiser |
|---|------|-----------|-----------|
| 1 | **Traffic monitoring** | maximise captured traffic, $\max \sum_{ij} C_{ij} A_{ij}$ | knapsack-greedy by traffic-per-DAS-cost |
| 2 | **Earthquake Early Warning (EEW)** | minimise worst-case detection distance, $\min \max_q \min_i d(s_i, q)$ | greedy minimax with a minimum-spacing constraint |
| 3 | **Activity monitoring** | maximise population within DAS coverage radius | greedy maximum-coverage |

**Task 1 — Traffic.** Each street edge carries a traffic weight derived from its
OpenStreetMap `highway` class (motorway → 2000, …, residential → 50). Its sensing
cost is the number of DAS units needed to cover it, $\lceil \ell / R \rceil$ for
sensing range $R$. Edges are ranked by traffic-per-cost and selected until the budget
is exhausted (a fractional-knapsack heuristic).

**Task 2 — EEW.** Earthquake epicentres are Monte-Carlo sampled from a
population-weighted distribution. Interrogators are placed greedily to minimise the
maximum over all epicentres of the distance to the nearest DAS (3-D, accounting for
hypocentre depth), subject to a minimum spacing between sensors.

**Task 3 — Activity.** Population is sampled from a WorldPop raster at each edge
midpoint. Interrogators are placed by greedy maximum-coverage to maximise the
population falling within a fixed coverage radius of some sensor.

Both **edge-count** and **km-of-fiber** budget modes are implemented for every task.

---

## Repository layout

```
DAS-for-Urban-Monitoring/
├── optimal_routing.py     # all three tasks: optimisers, evaluation, figures, CLI
├── run_targeted.py        # helper: targeted panel generation for selected cities
├── city_cache/            # bundled sample graphs (Tel Aviv, Manhattan, Barcelona)
├── population_data/        # (you provide) WorldPop rasters — see "Data" below
├── requirements.txt
├── LICENSE                # MIT
└── README.md
```

---

## Installation

```bash
git clone https://github.com/khencohen/DAS-for-Urban-Monitoring.git
cd DAS-for-Urban-Monitoring
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11, `osmnx` 2.1.0, `networkx` 3.3, `numpy` 1.26, `scipy` 1.17.

---

## Data

**Street networks (automatic).** Graphs are downloaded from OpenStreetMap via
[OSMnx](https://osmnx.readthedocs.io) and cached as pickles in `city_cache/`. A small
sample (Tel Aviv, Manhattan, Barcelona) is bundled so the code runs offline out of the
box; any other city is fetched on first use.

**Population rasters (manual, optional).** Tasks 2 and 3 require gridded population
data. Download the **WorldPop** *Unconstrained global mosaics, 2020, UN-adjusted,
1 km* GeoTIFFs from <https://www.worldpop.org/> and place them in `population_data/`
(or point `$WORLDPOP_DIR` at them). Files must follow WorldPop's naming convention:

```
population_data/
├── isr_ppp_2020_1km_Aggregated_UNadj.tif   # Israel  (Tel Aviv)
├── usa_ppp_2020_1km_Aggregated_UNadj.tif   # USA     (Manhattan, San Francisco)
├── esp_ppp_2020_1km_Aggregated_UNadj.tif   # Spain   (Barcelona)
└── ...
```

The ISO-3 country code for each city is listed in `CITY_POP_RASTER` in
`optimal_routing.py`. Task 1 (Traffic) runs **without** any population data.

> WorldPop data are distributed under CC-BY 4.0 by WorldPop, University of
> Southampton (Tatem, A.J. *WorldPop, open data for spatial demographics.*
> Scientific Data 4, 170004, 2017). They are **not** redistributed here.

---

## Usage

**Quick start — the offline sample:**

```bash
python optimal_routing.py --cities "Tel Aviv" Manhattan Barcelona
```

This runs all three tasks (Traffic always; EEW/Activity if population rasters are
present), prints a summary table, and writes figures (`fig_*.png`) plus per-panel
images to `panels/`.

**Full run — all 20 cities** (downloads any not already cached):

```bash
python optimal_routing.py
```

**Targeted panels** for specific cities/budgets (Traffic in km; Activity at 600 m
radius):

```bash
python run_targeted.py
```

### Using the optimisers programmatically

```python
import osmnx as ox
from optimal_routing import extract_edges, optimize_traffic, eval_traffic

G = ox.project_graph(ox.graph_from_place("Tel Aviv, Israel", network_type="drive"))
edges = extract_edges(G)

# Place 100 traffic-optimal interrogators
selected = optimize_traffic(edges, T=100)
print("captured traffic fraction:", eval_traffic(edges, selected) / sum(e["weight"] for e in edges))
```

Key entry points in `optimal_routing.py`:

| Function | Task | Budget |
|----------|------|--------|
| `optimize_traffic(edges, T)` / `optimize_traffic_km(edges, km)` | Traffic | edges / km |
| `optimize_eew(edges, T, eq_locs)` / `optimize_eew_km(...)` | EEW | edges / km |
| `optimize_activity(edges, T, pop)` / `optimize_activity_km(...)` | Activity | edges / km |

---

## Key parameters

Defined at the top of `optimal_routing.py`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `DAS_SENSING_RANGE` | 500 m | max length one DAS unit covers |
| `EEW_MIN_SPACING` | 5000 m | minimum distance between EEW sensors |
| `ACTIVITY_COVERAGE_RADIUS` | 600 m | DAS coverage radius for activity monitoring |
| `EQ_DEPTH` | 20 km | assumed earthquake hypocentre depth |
| `V_P`, `V_S` | 7000, 4000 m/s | P/S-wave velocities |
| `TRAFFIC_WEIGHT` | dict | per-`highway`-class traffic weights |

---

## Citation

If you use this code, please cite:

```bibtex
@article{cohen_das_urban_sensing,
  title   = {Percolation Thresholds for Urban Sensing with Existing Fiber-Optics Networks},
  author  = {Cohen, Khen},
  year    = {2026},
  note    = {Preprint}
}
```

*(arXiv identifier to be added on submission.)*

## License

MIT — see [LICENSE](LICENSE). Note that WorldPop population data are governed
separately by CC-BY 4.0.
