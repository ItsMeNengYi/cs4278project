# Full-task validation

**9/45 complete tasks succeeded. All-object reliability has not been achieved.**

The sweep used all 15 supplied objects and task seeds 0, 1 and 3 with the official drop. 32/45 trials verified a physical lift and 60-action hold; 29/45 did so within the lift deadline. There were 36 completed unsuccessful trials and 0 timeouts.

The component checks passed: **46/46** grasping/routing tests and **10/10** navigation-extension tests, with PyBullet installed. These are distinct from the physical full-task trials.

## Reproduction and scope

- Frozen executable source SHA-256: 9248c04c6cd4ae733c0994eab51cd737aa567ae085134ea6c459d45cd794df95.
- Base repository commit used for validation: 3836ccf14c1249c2c0977217e87f0c09f2e3734d.
- The frozen snapshot is in outputs/final-evaluation/source-9248c04c6cd4.
- Python 3.10.22; numpy 2.2.6, scipy 1.15.3, pybullet 3.2.7, trimesh 5.2.0, matplotlib 3.10.9.
- Four worker processes, one BLAS/OpenMP thread per trial, 300-second wall timeout per case. Soup can / seed 0 was retried with 600 seconds after its first timeout; both attempts are retained.
- Each action uses five 1/240-second physics updates, with four internal substeps and up to 200 solver iterations.
- The task seed changes point sampling and maze entrance/exit randomness; the supplied internal maze-generation seed remains 5.
- No upright-placement shortcut, live object attachment constraint, friction-coefficient increase, or pickup teleport was used.

    source /workspace/venvs/pnr/bin/activate
    cd /workspace/cs4278project
    python evaluate_tasks.py --seeds 0 1 3 --workers 4 --timeout 600 --output outputs/recheck

Full success requires a verified initial pickup, lift by environment action 3,000, retained grasp through navigation, a measured 3D goal error below 10 cm, and at most 15,000 total environment actions. Budgets use env.step_count, not individual internal physics updates. A timeout is unresolved execution, not a physical pass or proof of infeasibility.

The recorded lift action is the end of the full commanded 30 cm lift, before its stationary hold. This conservatively checks the deadline; it does not measure the earlier first crossing of the rubric's 10 cm threshold. The original 300-second sweep and timeout are retained in outputs/final-evaluation/results-initial-sweep.json; results.json includes the unchanged-source retry.

## Every object and seed

PICK includes failed planning, closure, execution or recovery before a verified pickup. CARRY means transfer/back-out failed after pickup. NAV means navigation or its contact monitor failed. REACH means final positioning failed. BUDGET means a deadline was missed.

| Index | Object | Seed 0 | Seed 1 | Seed 3 | In-budget pickups | Complete tasks |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Master chef can | PICK | PICK | PICK | 0/3 | 0/3 |
| 1 | Sugar box | NAV | NAV | CARRY | 3/3 | 0/3 |
| 2 | Soup can | PASS | CARRY | NAV | 3/3 | 1/3 |
| 3 | Mustard bottle | CARRY | CARRY | BUDGET | 2/3 | 0/3 |
| 4 | Potted meat can | CARRY | CARRY | CARRY | 3/3 | 0/3 |
| 5 | Banana | NAV | PASS | PASS | 3/3 | 2/3 |
| 6 | Strawberry | PASS | BUDGET | PASS | 2/3 | 2/3 |
| 7 | Apple | CARRY | CARRY | NAV | 3/3 | 0/3 |
| 8 | Bleach cleanser | PICK | PICK | PICK | 0/3 | 0/3 |
| 9 | Bowl | PICK | PICK | PICK | 0/3 | 0/3 |
| 10 | Mug | PICK | NAV | PASS | 2/3 | 1/3 |
| 11 | Plate | PICK | PICK | PICK | 0/3 | 0/3 |
| 12 | Large marker | PASS | PASS | PASS | 3/3 | 3/3 |
| 13 | Hammer | NAV | NAV | NAV | 3/3 | 0/3 |
| 14 | Foam brick | NAV | BUDGET | NAV | 2/3 | 0/3 |

## Completed tasks

| Object | Seed | Lift action | Minimum bottom clearance | Hold drift | Navigation actions | Total actions | Goal error | Robot-wall / object-obstacle actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Soup can | 0 | 1614 | 29.4 cm | 0.011 mm | 2230 | 4817 | 0.743 mm | 0 / 0 |
| Strawberry | 0 | 1653 | 27.5 cm | 0.044 mm | 2172 | 4501 | 0.157 mm | 0 / 0 |
| Large marker | 0 | 2235 | 28.7 cm | 0.005 mm | 2259 | 5052 | 0.042 mm | 0 / 0 |
| Banana | 1 | 2235 | 25.9 cm | 0.183 mm | 1835 | 5052 | 2.859 mm | 0 / 0 |
| Large marker | 1 | 2877 | 25.9 cm | 0.286 mm | 1690 | 5489 | 0.545 mm | 0 / 0 |
| Banana | 3 | 1514 | 25.5 cm | 0.424 mm | 1992 | 4070 | 0.473 mm | 0 / 0 |
| Strawberry | 3 | 1578 | 27.6 cm | 0.003 mm | 1923 | 4212 | 0.784 mm | 0 / 0 |
| Mug | 3 | 1534 | 29.6 cm | 0.009 mm | 1920 | 4014 | 0.088 mm | 0 / 0 |
| Large marker | 3 | 2957 | 28.0 cm | 0.009 mm | 1639 | 5581 | 0.041 mm | 0 / 0 |

## What was fixed

- The visual-only maximal-coordinate goal marker is excluded before collision-shape queries in both planning and support detection.
- Headless/realtime options are explicit, maze files are private per simulation, and point sampling is seeded.
- IK uses a tested analytic Jacobian and both elbow branches; measured joint-stop slop and continuous-angle execution are handled.
- Flat pads follow opposing surface normals rather than staggered sample positions. The search covers more contact regions, side approaches and table stances.
- Table routes enter normally from a clear approach point, and the loaded carriage has a checked connection to the maze grid.
- Carriage planning uses the complete asymmetric footprint; holonomic travel keeps yaw fixed and follows the footprint centre.
- Whole-body goal positioning uses the torso and base before a residual arm correction.
- Failed final reach/navigation now fails the process; metrics, checkpoints, timeouts, dependency versions and the frozen source are retained.

## Remaining failures

The remaining failures are algorithmic policy limitations, not missing packages. A useful antipodal pair and a collision-free rigid-grasp trajectory do not guarantee torque resistance or retention under physical contact changes. Some objects have no executable proposals under the present sampler/hand geometry; others lift and hold successfully but slip during transfer or transport. A failed candidate search does not establish that an object is physically impossible to manipulate.

| Object / seed | Last stage | Recorded reason |
| --- | --- | --- |
| Master chef can / 0 | pickup | RuntimeError: No candidate has a complete checked arm trajectory: {'finger_geometry': 1409}; last:  |
| Sugar box / 0 | navigation | RuntimeError: Navigation failed or the object was dropped: loaded pads [] |
| Mustard bottle / 0 | carry | RuntimeError: Lost loaded pad contact or object slipped during motion: loaded pads [21] |
| Potted meat can / 0 | carry | RuntimeError: Lost loaded pad contact or object slipped during motion: loaded pads [] |
| Banana / 0 | navigation | RuntimeError: Navigation failed or the object was dropped: loaded pads [] |
| Apple / 0 | carry | RuntimeError: Lost loaded pad contact or object slipped during motion: loaded pads [] |
| Bleach cleanser / 0 | pickup | RuntimeError: No verified pickup in 5 attempts: Object moved after planning; refresh the grasp |
| Bowl / 0 | pickup | RuntimeError: No candidate has a complete checked arm trajectory: {}; last:  |
| Mug / 0 | pickup | RuntimeError: Pickup attempt failed (Insertion planning failed: robot link 20 near object (0.0002 m)); recovery failed (No checked retreat after release) |
| Plate / 0 | pickup | RuntimeError: No candidate has a complete checked arm trajectory: {}; last:  |
| Hammer / 0 | navigation | RuntimeError: Navigation failed or the object was dropped: loaded pads [] |
| Foam brick / 0 | navigation | RuntimeError: Navigation failed or the object was dropped: loaded pads [] |
| Master chef can / 1 | pickup | RuntimeError: No candidate has a complete checked arm trajectory: {'finger_geometry': 1243}; last:  |
| Sugar box / 1 | navigation | RuntimeError: Navigation failed or the object was dropped: loaded pads [] |
| Soup can / 1 | carry | RuntimeError: Lost loaded pad contact or object slipped during motion: loaded pads [21] |
| Mustard bottle / 1 | carry | RuntimeError: Lost loaded pad contact or object slipped during motion: loaded pads [] |
| Potted meat can / 1 | carry | RuntimeError: Lost loaded pad contact or object slipped during motion: loaded pads [] |
| Strawberry / 1 | pickup_verified | RuntimeError: Pickup completed after the 3000-action budget |
| Apple / 1 | carry | RuntimeError: Lost loaded pad contact or object slipped during motion: loaded pads [] |
| Bleach cleanser / 1 | pickup | RuntimeError: No verified pickup in 5 attempts: Lost loaded pad contact or object slipped during motion: loaded pads [] |
| Bowl / 1 | pickup | RuntimeError: No candidate has a complete checked arm trajectory: {}; last:  |
| Mug / 1 | navigation | RuntimeError: Navigation failed or the object was dropped: loaded pads [] |
| Plate / 1 | pickup | RuntimeError: No candidate has a complete checked arm trajectory: {}; last:  |
| Hammer / 1 | navigation | RuntimeError: Navigation failed or the object was dropped: loaded pads [] |
| Foam brick / 1 | pickup_verified | RuntimeError: Pickup completed after the 3000-action budget |
| Master chef can / 3 | pickup | RuntimeError: No candidate has a complete checked arm trajectory: {'finger_geometry': 1239}; last:  |
| Sugar box / 3 | carry | RuntimeError: Lost loaded pad contact or object slipped during motion: loaded pads [] |
| Soup can / 3 | navigation | RuntimeError: Navigation failed or the object was dropped: loaded pads [22] |
| Mustard bottle / 3 | pickup_verified | RuntimeError: Pickup completed after the 3000-action budget |
| Potted meat can / 3 | carry | RuntimeError: Lost loaded pad contact or object slipped during motion: loaded pads [] |
| Apple / 3 | navigation | RuntimeError: Navigation failed or the object was dropped: loaded pads [] |
| Bleach cleanser / 3 | pickup | RuntimeError: No verified pickup in 5 attempts: Lift/hold validation failed: {'lift': 0.7349698391577139, 'lift_step': 12656, 'hold_drift': 0.001212965329108325, 'rotation_drift': 0.01296059853791446, 'bottom_clearance': 0.04984369551033441, 'held': False, 'hold_steps': 60, 'within_step_budget': False} |
| Bowl / 3 | pickup | RuntimeError: No candidate has a complete checked arm trajectory: {}; last:  |
| Plate / 3 | pickup | RuntimeError: No candidate has a complete checked arm trajectory: {}; last:  |
| Hammer / 3 | navigation | RuntimeError: Navigation failed or the object was dropped: loaded pads [22] |
| Foam brick / 3 | navigation | RuntimeError: Navigation failed or the object was dropped: loaded pads [] |

## Evidence and strategy

- [All trial metrics](validation/task_results.json)
- [Dependency/source manifest](validation/manifest.json)
- [Component test output](validation/component-tests.txt)
- [Navigation test output](validation/navigation-tests.txt)
- Per-trial raw logs and frozen executable source: outputs/final-evaluation/.
- [Detailed grasp, lift, carriage and navigation strategy](GRASPING_GUIDE.md)

## Additional goal-hold check

A separately instrumented full run of strawberry / seed 0 used the same frozen policy and then held the final base/arm commands for 60 more actions while normal grip-pressure feedback remained active. The extended check passed: True. Maximum extra-hold goal error: 0.197 mm.

Evidence: [extended hold metrics](validation/goal_hold.json), [lift](validation/strawberry_lift.png), [carriage](validation/strawberry_carry.png), and [goal after the extra hold](validation/strawberry_goal.png).

## Separate transport experiment

A gentler quintic base-translation profile was also tried on banana / seed 0, without changing the frozen benchmark policy. It completed navigation in 4,138 actions with no recorded obstacle contacts, but failed the final arm reach. It is not counted as a full success or included in the 45-case totals. Its metrics are in outputs/smooth-transport-experiment/result.json.
