# Navigation under uncertainty

Extension of the default project's navigation. The baseline in `../navigation.py` (known map, exact
pose, A*) is left untouched; this package removes those two oracles one at a time and measures what
each removal costs.

| Oracle in the default env | Replaced by | Algorithm |
|---|---|---|
| `cube_positions` (the full map) | 2D lidar (PyBullet ray casts) | log-odds occupancy grid + **D\* Lite** replanning |
| exact base pose in `qpos` | noisy odometry + the same lidar | **particle filter** (MCL, likelihood-field beam model) |

With both removed, the robot localises against the map it is building ("SLAM-lite": the map uses the
filter's estimate; it is not full SLAM, which would keep one map per particle).

## Workflow

```
every env step:   odometry -> particle filter predict (or dead reckoning)
every 5th step:   lidar scan -> PF weight + resample -> pose estimate
                              -> occupancy update -> cost grid -> D* Lite repair -> path
every step:       pure-pursuit carrot on the path -> velocity command (dx, dy)
stop when the estimate is within 3 cm of the goal; score with the true pose
```

## The modified task (what the robot is and isn't given)

- **Given:** goal position, the arena's outer bounds, its start pose with 5 cm error.
- **Not given:** wall positions, its pose after the start.
- **Lidar:** 180 beams over 360 degrees, 4 m range, sigma 2 cm, 1% dropouts, at 0.3 m height.
- **Odometry:** per-episode scale error (sigma 3% per axis) plus per-step noise. Dead reckoning
  therefore drifts by tens of cm over the ~25 m route, enough to hit walls in 1 m corridors.
- **Motion:** the agent sends a displacement per step, not absolute joint targets. The wrapper turns
  it into a joint target relative to the true joint state, like a velocity controller would. This
  matters: absolute targets would let the joint controller fix the pose for free.
- Yaw is held at 0 and treated as known; the estimated state is (x, y).

## Design choices worth defending in the report

- **D\* Lite over repeated A\*:** the map changes every scan, mostly far from the robot. D* Lite
  repairs only the affected part of its search (it searches from the goal, so the robot's moves don't
  invalidate it). `mapping_astar` runs the same episode replanning from scratch, for comparison.
- **Unknown cells are treated as free** (optimistic planning), so the robot heads for the goal and
  replans when walls appear.
- **Cost grid, three tiers:** walls are impassable; cells where the 0.57 x 0.53 m base box would touch
  a wall cost 200 (allowed only to escape after an estimation error); a soft penalty within 0.8 m of
  walls keeps paths in the middle of corridors.
- **Likelihood-field beam model:** cheap and smooth compared with ray-casting each particle's scan.
- **The maze's 1 m corridors** leave only 0.4 m of space for the base centre, so localisation errors
  above ~0.2 m show up as wall contacts.

## Files

| File | Purpose |
|---|---|
| `sim.py` | headless env, ground truth (walls, entrance, exact ray casts); never read by the agent |
| `world.py` | the agent's interface: lidar, odometry, displacement commands |
| `occupancy.py` | grid geometry, log-odds mapping, cost grid, likelihood field |
| `planner.py` | D* Lite and A* on the same graph |
| `localization.py` | particle filter over (x, y, odometry scale x, scale y) |
| `agent.py` | the loop above; `AgentConfig` switches each oracle on or off |
| `run_experiments.py` | runs conditions x seeds, writes `results/runs.csv` and plots |
| `tests/` | unit and integration tests |

## Running

From the project root, in the `pnr` conda environment:

```bash
python -m nav_uncertainty.tests.run_all
python -m nav_uncertainty.run_experiments --check-astar
python -m nav_uncertainty.run_experiments --conditions slam_lite --seeds 0 --gui
```

## Validation

**Unit tests** (`tests/`):
- D\* Lite equals A\* on random grids, and still equals it after random blocking/unblocking while the
  start moves (including uniform-cost grids, where heuristic ties are everywhere).
- A scan marks the wall surface occupied, the space in front free, and leaves the space behind unknown.
- The cost grid leaves exactly 4 passable cells across a 1 m corridor, cheapest in the middle.
- The particle filter stays within 8 cm over 8 m with 4% odometry bias, while dead reckoning drifts
  > 20 cm, and recovers the 4% bias to within 0.02.

**Integration tests** (PyBullet, headless):
- PyBullet lidar ranges match exact ray/box intersection with the true walls to 2 cm.
- A mapping episode where D\* Lite is checked against A\* from scratch on every replan (regression for
  a float-rounding bug in D\* Lite's priority keys, found this way; see `DStarLite._key`).
- The oracle condition reaches the goal with no wall contact.

**Experiments** (`run_experiments.py`), one row per condition, 5 seeds each:

| condition | map | pose | planner | what it shows |
|---|---|---|---|---|
| `oracle` | known | true | D* Lite | reference, same information as `navigation.py` |
| `mapping` | lidar | true | D* Lite | cost of not knowing the map |
| `mapping_astar` | lidar | true | A* from scratch | planner efficiency comparison |
| `dead_reckoning` | known | odometry | D* Lite | why localisation is needed |
| `mcl_no_bias` | known | particle filter over (x, y) | D* Lite | plain MCL |
| `mcl` | known | particle filter + odometry scale | D* Lite | MCL that also models the bias |
| `slam_lite` | lidar | particle filter + scale | D* Lite | both oracles removed |

Metrics: success (true base within 10 cm of the goal), steps, wall-contact steps, localisation RMSE
and max error, error in the learned odometry scale, planning time and node expansions, map
precision, and the largest gap between D* Lite's cost and A* from scratch over all replans.

## Results (seeds 0-9, `--check-astar`)

| condition | success | steps | contact steps | loc RMSE (m) | loc max (m) | goal err (m) | plan time (s) | expansions | map precision |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 10/10 | 1962 | 0.0 | 0 | 0 | 0.026 | 0.81 | 3,168 | - |
| mapping | 10/10 | 1962 | 0.0 | 0 | 0 | 0.026 | 8.40 | 58,708 | 0.984 |
| mapping_astar | 10/10 | 1959 | 0.0 | 0 | 0 | 0.027 | 1.78 | 107,051 | 0.984 |
| dead_reckoning | 1/10 | 4014 | 773.9 | 0.291 | 0.396 | 0.352 | 0.75 | 3,202 | - |
| mcl_no_bias | 9/10 | 1959 | 0.0 | 0.041 | 0.085 | 0.046 | 0.72 | 3,191 | - |
| mcl | 10/10 | 1962 | 0.0 | 0.032 | 0.078 | 0.017 | 0.68 | 3,201 | - |
| slam_lite | 5/10 | 1968 | 0.0 | 0.132 | 0.180 | 0.130 | 10.58 | 86,648 | 0.689 |

D* Lite agreed with A* from scratch on every replan of every unknown-map run (largest gap 6e-13).
Per-run plots (map, true vs estimated path, error over time) are in `results/`.

### Reading the results

- **Unknown map costs nothing in path quality.** Mapping with lidar and replanning optimistically
  reaches the goal in the same number of steps as the oracle. Lidar range (4 m) is larger than the
  maze's dead ends, so the robot rarely commits to a wrong corridor.
- **D\* Lite did fewer expansions (45% fewer) but took 4.7x longer than replanning from scratch.**
  Each D* Lite expansion updates all 8 neighbours, and each of those recomputes 8 edge costs in
  Python; the soft cost band also turns every new wall cell into hundreds of changed cells. On this
  map size, from-scratch A* wins on wall time. This is the honest answer to "why D* Lite": it wins on
  work done, and would win on time only with a cheaper cost update or a much larger map.
- **Dead reckoning fails** (1/10, 774 wall-contact steps on average, one timeout): a 3% scale error
  is about 30-90 cm over the route, more than the 0.2 m clearance in a corridor.
- **The particle filter fixes it** (10/10, no contact, 3 cm RMSE).
- **Modelling the odometry bias** in each particle fixed the one plain-MCL failure (seed 3: a 10 m
  straight corridor where position along the corridor is unobservable, and plain MCL stayed
  confident while drifting 27 cm). Goal error dropped from 4.6 cm to 1.7 cm on average. But the
  learned scale is not clearly closer to the true one (scale error 0.031 vs 0.032 for no estimate),
  so the gain may come from the extra spread the scale adds along the direction of travel rather
  than from learning the bias. That needs a per-axis check before claiming it.
- **SLAM-lite is the open problem** (5/10). Localising against a map built from the filter's own
  estimate lets errors feed into the map (precision 0.69 vs 0.98 with true pose), which then
  misleads the filter. The failures arrive at the goal by their own estimate but are 15-30 cm away.
  Full SLAM (a map per particle, i.e. FastSLAM, or scan matching with loop closure) is the next step.
