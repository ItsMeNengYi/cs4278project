# Grasping revision: design, critique, and validation

This revision replaces the grasping pipeline around the supplied Fetch robot.
Navigation, mapping, and maze planning are unchanged. The replacement includes
settling, grasp proposals, full hand poses, arm IK, arm paths, insertion, closing,
lifting, hold verification, failure recovery, compact carry motion, and final arm
reach. It uses the provided point cloud and normals for grasp generation, as the
assignment requires. The known simulator geometry is used for collision checks.

**Validation status:** 32 tests pass and 4 Bullet integration tests are skipped
on the development host because PyBullet could not be installed (package index
access is denied). The passing tests cover numerical helpers, proposal mechanics,
and controller contracts with mocked physics. They do **not** establish physical
pickup success. No success rate or compliance with the 3000-step budget has been
measured. The executable evaluation harness is included to obtain that evidence.

## Running it

Use the recommended Python 3.10 environment and install the dependencies:

```bash
pip install -r requirements.txt
python run_task.py --headless --seed 3 --object 2 --grasp-only
python run_task.py --seed 3 --object 2 --grasp-only
python -m unittest discover -s tests -v
python evaluate_grasps.py --seeds 0 1 2 3 4 --objects 0 1 2 5 14 --output grasp_results.json
```

Without `--grasp-only`, `run_task.py` continues into the existing maze navigation
and uses the updated arm functions for carry and final reach. The official drop
from z=5 is the default. `--place-on-table` is an explicit upright-placement debug
setup and must be reported separately. The original runner silently used that
simpler setup. `python pick_nav_reach_env.py` now launches the working runner;
its old demonstration imported a missing `util_IK` module.

## Why this implementation fits this task

There is no universally best grasp algorithm. This environment already provides
1024 samples from the complete object mesh, outward mesh normals, object pose,
and collision geometry. It is much easier than inferring graspable surfaces from
an occluded, noisy camera image. A geometric candidate generator plus actual
hand, robot, and trajectory checks is a practical fit without model weights,
GPU inference, training data, or camera calibration.

Dex-Net 2.0 samples multiple antipodal grasps and ranks their robustness under
uncertainty. GPD generates complete hand poses from point clouds. Even the
official GraspNet baseline filters network proposals using approach-volume
collision detection. These support the architecture, but this implementation is
not a reimplementation of those learned systems. Its score is an engineering
heuristic, not a learned or calibrated probability of success.

## What was wrong with the previous method

The antipodal sign tests were correct. Opposite outward normals aligned with
the contact line are a sensible starting point. The weakness was treating a
promising pair as though it established an executable, stable grasp.

| Previous behavior | Why it fails | Replacement |
| --- | --- | --- |
| Keep one pair nearest the object-center line | It may be unreachable or put the palm through the object; nearby alternatives disappear | Up to 40 geometrically distinct ranked pairs |
| Distance to an infinite contact line | A line can pass through the center while both contacts lie off to one side | Distance to the finite segment plus midpoint centrality |
| Arbitrary 20-degree normal tests | The friction and normal-error assumptions are hidden | Explicit friction-cone angle and uncertainty margin |
| Trust isolated surface samples | A good normal on a tiny rim or mesh triangle may not support a real pad | Nearby, similarly oriented surface-patch support |
| Prefer horizontal normals | This describes a closing axis, not a collision-free top-down hand pose | Sample approach directions and construct the complete rigid gripper frame |
| Offset each open pad along its own surface normal | The targets need not fit a rigid parallel-jaw gripper | One common closing axis and symmetric opening |
| Endpoint collision checks | Two valid endpoints can be connected through a table, wall, or robot link | Check every sampled path edge against scene and self geometry |
| Unchecked above-table waypoint | Failed IK or a colliding waypoint can still be executed | Reject incomplete plans explicitly |
| Solve IK by moving the live robot | Numerical search becomes physical movement and can disturb the scene | Separate DIRECT planning client |
| Sum of axis cross products for orientation | At 180 degrees the cross product is zero; terms can also cancel | Full rotation-vector error with separate angular and positional tolerances |
| Silent IK stall | The next stage proceeds despite an unsolved target | Return failure and reject the candidate |
| Fixed joint increments | A radian and a metre have different motion limits | URDF velocity limits with conservative acceleration caps |
| Both fingers touched once | The object may remain table-supported or immediately slip | Loaded inner-pad contacts, persistent relative pose, actual lift and hold |
| Subtract 5 cm from finger setpoint to squeeze | This is a large position-control overclosure, not measured force control | Slow closure, sustained contact, small 2 mm compression target |
| Randomly discard 40% of points on retry | It loses useful surface information and may select the same bad geometry | Reobserve and skip distinct failed contact regions in the object's frame |

The old `sqrt(d^2+s^2) < 31.5 mm` test was not equivalent to a point being on a
pad. For example, a point 25 mm above the pad center is 13 mm beyond its 12 mm
half-height. Its old error is sqrt(25^2+13^2)=28.2 mm, which passes despite being
outside the pad. The new diagnostic measures distance to the actual pad rectangle
and requires the selected object-fixed contacts to lie within 6 mm after closure.

A heap is not what makes the old approach fail physically. Building a heap of all
accepted pairs and popping just one adds storage without solving feasibility.
The new generator vectorizes pair rows, retains bounded row winners, and applies
diversity suppression. Its pair search remains O(N^2), appropriate for 1024 points.

## The new pipeline

### 1. Wait for actual rest

`wait_until_rest` requires small linear velocity, small angular velocity, and
loaded contact with a support for 20 consecutive environment actions. Translation
alone cannot distinguish a stationary object from one rotating in place, and a
single small displacement can occur near the apex of a fall.

### 2. Generate multiple friction-compatible contact pairs

In `grasp_candidates.py`, for outward normals n1 and n2 and contact direction
u=(p2-p1)/||p2-p1||, the inward contact directions must satisfy:

    angle(-n1,u) <= atan(mu) - normal_uncertainty
    angle( n2,u) <= atan(mu) - normal_uncertainty

The task runner derives mu conservatively from the unchanged simulator contact
materials: 0.9 times the object and finger lateral-friction coefficients' product,
using Bullet's default friction combiner. It reserves 3 degrees for normal error.
The standalone generator defaults to mu=0.5 and a 5-degree margin, which are
configurable assumptions, not material measurements.

The score combines normalized friction margin (35%), local patch support (20%),
centrality (25%), horizontal closing direction (15%), and spare opening (5%).
These weights are explicit heuristics. They have not been tuned on hidden seeds.
The local patch count is a sampling-density proxy, not proof of a filled contact
patch or a wrench-space force-closure calculation. Centrality uses the supplied
object reference position; it does not estimate mass distribution from samples.

Candidates too wide for the jaws or too close to the support are filtered.
The runner uses a permissive 4 mm contact-height prefilter so thin objects are not
prematurely rejected. The full oriented gripper geometry is checked later.
Midpoint, axis, and width diversity prevent the candidate list being filled with
near-duplicates. `util_find_best_pair.py` remains as the original comparison
baseline and is no longer used by the task runner.

### 3. Turn a pair into a rigid hand pose

In `grasping._approaches`, the gripper's y axis follows the contact segment and
its x axis points into the object. z=x cross y completes an orthonormal frame.
Approach directions are sampled perpendicular to the closing line; above-table
and robot-facing approaches receive preference.

The palm must clear the object, and the palm and finger boxes must clear the
support. The pad midpoint can shift toward the approach direction while keeping
the selected contacts on the 63 mm finger faces. This is especially useful for
thin objects: the fingertips can contact them without putting the entire finger
center down at table height. Exact collision geometry then checks the candidate.

For this URDF, `gripper_link`'s origin is the symmetric pad midpoint. The maximum
inner-pad gap is 102.85 mm, not exactly 100 mm:

    gap = 2 * (finger_joint_q + 0.001425)

The code keeps opening clearance and solves one rigid TCP pose instead of two
independent, potentially incompatible pad-center objectives.

### 4. Solve IK without changing the live simulation

`grasp_motion.MotionPlanner` loads a separate DIRECT world containing the Fetch,
object, floor, table, and maze, at their observed poses. Every planning reset
addresses that client. There are no live joint resets or save/restore searches.
The planner resolves action columns and links by joint and link names.

Only the torso and seven arm joints are IK variables. Base poses are sampled as
fixed candidate stances; the existing navigation still drives to the chosen
stance. Position error is in metres. Orientation error is the rotation vector
of R_target R_current^T, so a half-turn has error pi rather than zero.

The damped least-squares update is:

    dq = J.T * inverse(J*J.T + lambda^2*I) * error

The implementation weights rotation by 0.4, clips increments and joint limits,
and uses backtracking plus a bounded number of deterministic restart seeds.
Success requires position <=3 mm and orientation <=0.06 rad independently.
Non-convergence returns None. It does not silently advance to the next stage.
Central differences of FK compute the Jacobian, avoiding an unverified assumption
about Bullet's COM-versus-link local Jacobian-point convention. This is slower
than a validated analytic Jacobian but keeps frame handling explicit.

### 5. Validate the complete arm motion

The planner checks robot/world collisions, object collisions, and nonadjacent
self collisions. Adjacent and rigidly connected geometry and the supplied SRDF
exclusions are handled explicitly so intended assembly overlap is not treated
as a collision. It rejects unknown collision bodies instead of ignoring them.

Transit first tries a checked straight joint path; bounded, seeded bidirectional
RRT-Connect is the fallback. Final insertion uses sequential Cartesian IK.
Every connecting edge is checked, and a partial path is failure. A prospective
closed-grasp lift is also checked with the object geometry carried along.

After navigation arrives, the arm transit is replanned from measured joints and
the actual base pose. Object translation and rotation are rechecked before
transit, before insertion, and before closure. Stale object targets trigger a
new attempt. The actual lift is replanned after closure using the measured
object-to-gripper transform.

Collision sampling is discrete (up to 45 mrad / 9 mm per general edge, finer for
Cartesian edges), not continuous collision detection. Checked shortcuts reduce
unnecessary stops. Cartesian shortcuts additionally keep sampled TCP motion
within 4 mm of the straight segment and orientation within 0.06 rad.

### 6. Execute the checked path and close deliberately

`execute_path` follows the checked joint segments using quintic time scaling.
Timing accounts for individual URDF joint velocity limits at 45% speed and
conservative acceleration caps. Exact collinear collision samples do not add
extra execution stops. This limits commanded setpoints, not a guarantee about
actual motor dynamics; tracking error is also checked at the end of the path.

`open_gripper` explicitly fails if the requested opening is not reached.
`close_gripper` reduces each finger target by 0.4 mm per action until both inner
pads have loaded contacts for eight consecutive actions. Contacts on finger
backs, tips outside the pad bounds, or near-zero-force contacts do not pass.
A small compression target is retained, then contact persistence is checked.
The measured object-relative pose becomes the reference for slip detection.

This remains position-based contact control because the supplied `env.step`
interface sets positions. It is not a force controller. No friction increase,
object teleport during pickup, collision disabling, or physics weld was added.
The environment's existing very large motor force limit is unchanged, so a
simulation result would not establish hardware force safety or realism.

### 7. Lift and prove the grasp holds

A collision-checked Cartesian lift is planned with the object included as
geometry attached to the shadow gripper. That attachment exists only in planning;
the live object is held by physical contacts. The normal lift is 15 cm. If a drop
landed on the floor, more lift is required to clear the actual table height.

Verification requires at least 50 actions (60 by default), loaded contact on both
pads throughout, object rise >=10 cm, object bottom >=10 cm above the table,
and relative translation/rotation drift within 10 mm / 0.15 rad during the hold.
The broader `holding` guard also checks deviation from the post-closure reference.
The report includes lift step, bottom clearance, relative drift, rotation drift,
hold length, and whether the lift occurred by step 3000.

Compact carry motion and final reaching also plan with the measured held-object
geometry and preserve the gripper's closure command. They fail explicitly when
no valid arm path exists. Base navigation and its path follower are unchanged.

### 8. Recover and try different evidence

A failed physical attempt opens the gripper, requests a checked retreat, waits
for rest, regenerates proposals from the current observation, and skips similar
failed contacts in the object's local frame. If a safe retreat or a complete new
plan cannot be found, execution stops with an explicit error. It does not keep
sending uncontrolled IK updates into a collision.

## Files and tests

| File | Role |
| --- | --- |
| `grasp_candidates.py` | Friction-aware ranked, diverse contact proposals; no simulator dependency |
| `grasp_motion.py` | Shadow scene, FK, IK, collision checks, RRT-Connect, Cartesian paths and checked shortcuts |
| `grasping.py` | Hand geometry, planning orchestration, execution, contacts, lift, carry, recovery |
| `run_task.py` | Ranked-candidate attempts and existing full-task integration; `--grasp-only` option |
| `evaluate_grasps.py` | Repeated seed/object trials with JSON results and failures |
| `requirements.txt` | Explicit runtime dependencies, including SciPy |
| `tests/test_grasp_candidates.py` | Friction direction, uncertainty, invalid data, support filtering and diversity |
| `tests/test_grasp_motion.py` | Rotation/Jacobian/shortcut logic; real Bullet checks when installed |
| `tests/test_grasp_execution.py` | Mocked contact, timing, stale targets, closure, slip, lift and failure contracts |

The four skipped integration tests cover actual Fetch FK/Jacobian agreement,
IK roundtrip with live-world isolation, table collision rejection, and a rotation
check in the live fixture. Real seed/object trials remain required to validate
URDF collision exclusions, contact dynamics, thin-object coverage, recovery,
and the step budget. Neural methods, continuous collision detection, measured
force feedback, full wrench-space robustness, and hardware deployment are outside
this implementation's demonstrated capabilities.

## Research references

1. Mahler et al., **Dex-Net 2.0**. Multiple antipodal candidates and grasp robustness under uncertainty:
   https://roboticsproceedings.org/rss13/p58.pdf
2. ten Pas et al., **Grasp Pose Detection in Point Clouds**, paper and official implementation:
   https://arxiv.org/abs/1706.09911 and https://github.com/atenpas/gpd
3. **GraspNet baseline**, explicit approach-volume collision filtering after predictions:
   https://github.com/graspnet/graspnet-baseline/blob/main/test.py
4. Lynch and Park, **Modern Robotics, Force Closure**. Two ideal spatial point contacts cannot resist torque about the line connecting them; finite soft contact patches change the contact model:
   https://modernrobotics.northwestern.edu/nu-gm-book-resource/12-2-3-force-closure/
5. **Modern Robotics, Numerical IK**. Transform-based error, separate linear/angular tolerances, compute before moving:
   https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-2-of-2/
6. **MoveIt, Kinematics**. Scene and self collision, allowed-collision exclusions:
   https://moveit.picknik.ai/main/doc/concepts/kinematics.html
7. **MoveIt, Trajectory Processing**. Paths need timing consistent with per-joint limits:
   https://moveit.picknik.ai/main/doc/concepts/trajectory_processing.html
8. **PyBullet Quickstart**. Multiple clients, contact data, FK and nearest-point collision queries:
   https://raw.githubusercontent.com/bulletphysics/bullet3/master/docs/pybullet_quickstartguide.pdf
