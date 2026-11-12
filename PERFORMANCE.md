# Simulation speed and grasp-geometry diagnosis

The task runner now executes without deliberate sleeps in both GUI and headless mode. Use `--realtime` when you want physics paced for watching:

```bash
python run_task.py --object 6 --seed 0
python run_task.py --object 6 --seed 0 --headless
python run_task.py --object 6 --seed 0 --realtime
```

## Measured speed

These are profiled official-drop runs of strawberry / seed 0 on this cloud machine. They use the same four internal physics substeps, 200 solver iterations, motor forces, friction coefficients, action counts and collision geometry.

| Measurement | Before | After |
| --- | ---: | ---: |
| Full headless episode, excluding Python import time | 84.934 s | 77.054 s |
| Bullet exact distance queries | 10.755 s | 0.717 s |
| Collision validation, including its bookkeeping | 11.558 s | 2.131 s |
| Cartesian planning | 11.677 s | 3.322 s |
| Physics integration | 65.600 s | 66.372 s |
| Environment actions | 4,501 | 4,501 |
| Final 3D goal error | 0.1574 mm | 0.1574 mm |
| Full task passed | Yes | Yes |

Headless wall time fell by about 9.3% in this comparison. Timing depends on hardware, load and profiling overhead. Physics integration remains the main cost.

Previously a GUI run also requested five sleeps of 1/240 s per environment action. This episode's 4,501 actions therefore requested **93.77 seconds of additional sleep**, before rendering overhead. Those sleeps are now opt-in. A complete before/after GUI timing was not measured; the headless timings above must not be presented as GUI timings.

The planning speedup comes from conservative bounds before exact collision queries. Static maze boxes are indexed individually in the shadow world; distant walls are skipped. Nonadjacent self-link pairs are checked individually after excluding separated AABBs and semantic exclusions. Exact Bullet distances still decide collision feasibility. The live maze representation is unchanged.

The new collision tests compare against exhaustive Bullet self queries and the original merged maze over 100 configurations each, including colliding and free cases and a payload extending beyond the base. All 48 component tests passed.

Two alternatives were investigated and left out: splitting the live maze was slower in a loaded stationary benchmark; reducing the solver budget to 50 iterations sped up stepping but caused a previously successful strawberry run to lose its grasp. The shipped runner retains the original numerical settings.

## Object 0: master chef can

The apparent opening clearance depends on which geometry is measured. The loaded assets give:

| Quantity | Value |
| --- | ---: |
| Maximum physical jaw gap | 102.850 mm |
| Visible mesh horizontal X/Y extents | 102.529 / 102.377 mm |
| Collision mesh horizontal X/Y extents | 103.815 / 103.785 mm |
| Bullet object collision margin | 1.000 mm |
| Open-finger clearance requested by the planner | 0.200 mm |

Thus the visible X extent is only **0.321 mm** smaller than the maximum opening. A nominal 0.2 cm difference would be 2 mm, but it is not the visible-mesh gap in this checkout. The larger collision mesh and contact padding further reduce the available space. A broadphase AABB is also padded; it must not be interpreted as an exact cylinder diameter.

For seed 0, the geometric diagnostic evaluated 160 antipodal pairs and all 1,569 generated hand approaches. None passed the open-finger mesh checks. Even the best tested pose had approximately **0.187 mm of penetration**, whereas the check requires positive clearance. A sampled contact-pair width can be narrower than the full collision surface occupying the pad area.

This explains the current `finger_geometry` rejection. The runner now reports the signed separation and jaw gap in its error. These measurements do not establish that every possible manipulation or grasp of the can is impossible; changing the pose search or using preparatory manipulation is a separate algorithmic problem. The shipped change does not enlarge the gripper, shrink the object or bypass collision checks.

## Object 11: plate

The visible plate is about 260 mm across, but a rim grasp would close across its local thickness. The issue is therefore more than the plate's full diameter.

With the supplied 1,024 samples and seed 0, the current antipodal sampler retains one pair, approximately **3.115 mm** apart near the rim. Its hand-approach calculation requires at least **60.679 mm** of TCP/contact offset to clear the projected object geometry. The allowed offset within the finger pad is **30.5 mm**. No hand approach survives, even before the arm IK stage.

This is a limitation of the current sampling/approach policy. A more suitable rim approach or preparatory repositioning remains necessary. The runner now records `hand_approach` and a readable cause instead of the previous empty rejection dictionary.

## Evidence and reproduction

The earlier all-object policy benchmark remains in [TASK_VALIDATION.md](TASK_VALIDATION.md). This performance update does not claim all-object success.

To profile a complete headless run:

```bash
python -m cProfile -o outputs/strawberry.prof run_task.py \
  --headless --object 6 --seed 0 --result outputs/strawberry.json
python -m pstats outputs/strawberry.prof
python -m unittest discover -s tests -v
```

Frozen regression source, per-trial logs and raw profiles are retained under `outputs/speed-regression/`, `outputs/performance-baseline/` and `outputs/performance-updated/` in the execution workspace. Portable measurement summaries and the completed regression results are in `validation/performance/`.

## Regression results

**All nine previously successful full-task cases passed again.** Objects 0 and 11 remained planning rejections on seeds 0, 1 and 3. These six failed tasks are reported explicitly; they are not task passes. No trial timed out.

This targeted regression is separate from the earlier 45-case all-object benchmark. It does not establish all-object reliability.

| Seed | Object | Result | Actions | Goal error |
| --- | --- | --- | ---: | ---: |
| 0 | Master chef can | PICKUP REJECTED | 392 | — |
| 0 | Soup can | PASS | 4817 | 0.743 mm |
| 0 | Strawberry | PASS | 4501 | 0.157 mm |
| 0 | Plate | PICKUP REJECTED | 407 | — |
| 0 | Large marker | PASS | 5052 | 0.042 mm |
| 1 | Master chef can | PICKUP REJECTED | 392 | — |
| 1 | Banana | PASS | 5052 | 2.859 mm |
| 1 | Plate | PICKUP REJECTED | 407 | — |
| 1 | Large marker | PASS | 5489 | 0.545 mm |
| 3 | Master chef can | PICKUP REJECTED | 392 | — |
| 3 | Banana | PASS | 4070 | 0.473 mm |
| 3 | Strawberry | PASS | 4212 | 0.784 mm |
| 3 | Mug | PASS | 4014 | 0.088 mm |
| 3 | Plate | PICKUP REJECTED | 407 | — |
| 3 | Large marker | PASS | 5581 | 0.041 mm |

Frozen regression source SHA-256: `95ff6c118d19ff6e90d761c744ca94b54bfb56b07fb3d1b10251e951d5e1f5b0`.

Portable evidence: [regression metrics](validation/performance/regressions.json), [manifest](validation/performance/manifest.json), [timing comparison](validation/performance/comparison.json), [geometry diagnosis](validation/performance/grasp_geometry.json), and [48 component checks](validation/performance/component-tests.txt).
