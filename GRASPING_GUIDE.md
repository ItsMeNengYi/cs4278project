# Grasping, lifting, navigation, and measured limits

The runner uses geometric antipodal proposals, numerical inverse kinematics, collision-checked arm paths, physical jaw contact, and A* navigation with the supplied holonomic base. It does not attach the live object to the robot. The exact scenario results and tested source snapshot are recorded in [TASK_VALIDATION.md](TASK_VALIDATION.md).

Passing the component tests does not establish reliable pickup or transport. Full physical trials remain the deciding evidence; the validation report includes unsuccessful objects and seeds.

## Objective and units

The course task is to generate a grasp from the provided point cloud and outward normals, pick up the object, carry it through the maze, and bring its reference position within 10 cm of the green goal at z = 0.9 m.

The runner checks these requirements:

| Check | Requirement used by the runner |
| --- | --- |
| Grasp position | Selected material contacts within 6 mm of the actual pad rectangles after closure |
| Pickup deadline | Lift completed by environment action 3,000 |
| Lift | Object reference rises at least 10 cm and its bottom clears the table by at least 10 cm |
| Initial stability | At least 50 actions; the default verification holds for 60 |
| Initial hold drift | At most 10 mm translation and 0.15 rad rotation relative to the hand during that hold |
| Goal | Measured 3D object-to-goal distance below 0.1 m |
| Full-task deadline | At most 15,000 environment actions in total |

Budgets use the repository's **env.step_count**. One environment action advances five calls to Bullet's stepSimulation at a fixed 1/240 s timestep, so an action represents 1/48 s of simulated time. These action counts must not be confused with counts of individual physics updates.

The normal commanded lift is 30 cm, giving room for the subsequent arm transfer. If the official drop lands an object on the floor, the lift planner also requires clearance above the actual table, rather than passing a floor pickup merely because it rose 10 cm.

The recorded lift action is the end of that full commanded lift, before the hold. This is a conservative deadline check: the runner does not record the earlier instant at which the object first crossed the rubric's 10 cm threshold. Consequently, a reported deadline failure can be stricter than that threshold alone.

## Running and reproducing the evaluation

From the existing checkout:

    cd /workspace/cs4278project
    source /workspace/venvs/pnr/bin/activate
    export MPLCONFIGDIR=/workspace/.cache/matplotlib
    export MPLBACKEND=Agg
    export OPENBLAS_NUM_THREADS=1
    python run_task.py --headless --seed 0 --object 6 --result outputs/strawberry-seed0.json
    python run_task.py --headless --seed 0 --object 6 --grasp-only
    python evaluate_tasks.py --seeds 0 1 3 --workers 4 --timeout 600 --output outputs/recheck
    python -m unittest discover -s tests -v
    python -m nav_uncertainty.tests.run_all

Object indices are the sorted directory order in assets/ycb_objects, from 0 through 14. The default setup is the environment's official drop from z = 5 m. The --place-on-table flag is a separate upright debug setup and is excluded from the reported official-drop trials.

Each batch freezes the Python source once, records its SHA-256 and dependency versions in manifest.json, and executes that snapshot in separate processes. Each trial has JSON metrics and a complete log. A timeout retains its last checkpoint and is reported separately from a completed physical failure. Assets are shared read-only by convention; every environment generates its own temporary maze URDF.

The task seed controls the sampled point cloud and maze entrance/exit randomness. The supplied environment retains its internal maze-generator seed of 5; this evaluation does not claim coverage of arbitrary maze-generation algorithms.

## 1. Observe a sufficiently settled object

The observation contains 1,024 complete surface samples, their outward normals, object pose, exact robot joint state, goal, and wall positions. This is a supplied-mesh simulation observation, rather than an occluded camera reconstruction.

Settling requires loaded support contact with a body other than the robot, low linear velocity, and a measured pose window with less than 2 mm translation and 0.02 rad rotation. This avoids accepting the apex of a fall or a spinning object. The pose window also tolerates alternating contact-solver velocity jitter when the actual orientation is stable.

## 2. Generate and rank diverse antipodal pairs

For contacts p1 and p2 with outward normals n1 and n2, let u be the unit vector from p1 to p2. The inward contact directions must fall inside both friction cones:

    angle(-n1, u) <= atan(mu) - epsilon
    angle( n2, u) <= atan(mu) - epsilon

The runner reads the unchanged object and finger friction coefficients and uses Bullet's product combiner for mu. It reserves one degree for normal uncertainty. With the supplied default 0.5 coefficients on both surfaces, mu is 0.25 and the allowed angular deviation is approximately 13 degrees.

The score combines friction margin (35%), nearby similarly oriented samples (20%), centrality (25%), horizontal closing direction (15%), and spare opening (5%). Centrality uses distance to the finite contact segment and its midpoint; it is a geometric heuristic, not a calibrated success probability or a full force-closure certificate.

Up to 160 proposals survive suppression by midpoint, unoriented axis, and width. The runner uses 4 mm spatial diversity. A thin-rim fallback permits a single local supporting sample and narrower visual contact pairs, but the actual hand geometry and physical lift must still pass.

The physical maximum jaw gap remains 102.85 mm. The finger joint command for a desired gap is:

    q_finger = clip(gap / 2 - 0.001425, 0, 0.05)

The minimum physical gap therefore remains 2.85 mm. Sampling a narrower visual pair does not extend the jaw travel or establish that it can be held.

## 3. Construct complete hand poses and base stances

The hand frame uses x as the insertion direction and y as the jaw direction. The y axis follows the average opposing surface normals. This matters when two sampled points are staggered: aiming the jaw axis through those points can tilt flat pads against flat faces and leave only edge contact.

Approach directions are perpendicular to the closing axis. The search tries side approaches across the candidate set before accepting an angled or top-down grasp. A side grasp can put a shorter vertical contact band near the object's centre and avoid a subsequent large wrist rotation. It still has to clear the table and the complete robot geometry.

The planner samples several insertion offsets, standoffs of 0.55–0.85 m, and stances on four sides of the contact pair. The supplied point cloud determines the grasp; the known simulator meshes determine collision feasibility. Exact shadow finger-box checks cheaply reject impossible openings before arm IK.

The parking configuration raises and straightens the complete arm. A visibility route goes around the expanded table to an entry point outside the selected edge. The robot turns there, then approaches the stance perpendicular to the edge. This avoids the old route that crossed a table leg while moving along a near-edge stance.

## 4. Solve IK in a separate planning world

The planner loads a separate DIRECT Bullet client containing the robot, object, floor, table, and that environment's maze. It reads live state, but all planning joint resets affect only the shadow robot.

IK varies the torso and seven arm joints while a candidate base stance remains fixed. Position error is in metres; orientation error is the full SO(3) rotation vector. A half-turn therefore produces an error of pi, rather than incorrectly producing zero as some cross-product residuals do.

The damped least-squares update has the form:

    dq = J.T @ inverse(J @ J.T + lambda**2 * I) @ error

Rotation is weighted separately, updates are bounded, and backtracking reduces the pose error. Success requires position error below 3 mm and orientation error below 0.06 rad. Failed IK returns failure.

The analytic Bullet Jacobian at the TCP link origin is checked against independent central forward-kinematics differences over multiple poses. Restart seeds explicitly cover both elbow branches; small noise around one branch was insufficient to find the other.

Measured joint-stop excursions of at most 1 mm/rad in the numerical check are projected onto the URDF limits. Larger violations fail. Commanded targets remain clipped by the environment's original joint limits.

## 5. Plan complete arm motion before execution

Transit first tries a checked joint-space edge. Seeded bidirectional RRT-Connect is the fallback. Insertion and lifting use sequential Cartesian IK; a partial path is rejected.

Checks include robot/world geometry, the object, and nonadjacent self-collisions. Assembly adjacency and the supplied SRDF exclusions are handled explicitly. The visual-only goal marker is excluded before collision-shape queries; querying its maximal-coordinate body caused the original crash.

The planner checks intended finger contact separately from unwanted collisions. A carried object may touch the palm within 1 mm of contact slop. This permits an actual settled grasp to start moving; it does not disable live palm collision response or permit penetration into the rest of the arm.

General edges are sampled at up to 45 mrad / 9 mm, with finer checks on Cartesian edges. Cartesian shortcuts also limit TCP deviation from the intended segment to 4 mm and orientation deviation to 0.06 rad. This is sampled validation, not continuous collision detection.

After the base arrives, the runner rechecks object translation and rotation and replans from measured joints, using the successful pregrasp configuration as its IK seed. It rechecks the object again before insertion and closure. Stale plans fail instead of being executed against an old pose.

## 6. Execute, close, and retain contact

Joint paths use quintic time scaling at 45% of the URDF velocity limits, with conservative acceleration caps. Continuous joints use the same shortest angular arc in execution as in planning. Otherwise, equivalent angles near plus/minus pi could cause an unplanned long rotation.

The jaws open to the real maximum gap. Closure advances each finger by 0.4 mm per action until both inner pads have loaded contacts for eight consecutive actions. The code checks the actual 63 mm by 24 mm pad rectangles, not just the finger link IDs.

A small 2 mm compression target is retained. During movement, a one-way pressure trim takes up a weakening contact in increments no larger than 0.2 mm per action, bounded to 2 mm additional closure. Its minimum-load estimate uses object weight and the unchanged friction coefficient. It does not reopen a loaded grasp.

This is position control with contact-force feedback, rather than a true force-controlled gripper. The environment retains its original high motor force setting. No new live attachment constraint, object teleport during pickup, or friction-coefficient increase is used.

## 7. Lift and verify stability physically

The actual object-to-hand transform is measured after closure and used for the shadow lift plan. Only the shadow planner treats the object as attached geometry; the live object remains held by contacts.

The runner checks rise, bottom clearance above the table, two loaded inner pads, relative translation drift, relative rotation drift, and the 60-action hold. The returned metrics include the lift action, clearance, drift, rotation, hold length, and budget compliance.

During movement, the contact monitor permits loaded rolling within the hand rather than demanding a rigid weld. It still requires both loaded inner pads and rejects separation beyond a capture radius derived from the observed object bounds and the physical pad dimensions. This allows an object held near its end to rotate without incorrectly declaring its centre of mass detached. The stricter stationary drift limits remain part of the initial lift-and-hold test.

A failed attempt opens the jaws, requests a checked retreat, returns to the parking configuration, reobserves, and avoids a similar failed contact region in object coordinates. There are at most five physical attempts. Recovery failures preserve the original cause and stop execution.

## 8. Transfer into a carriage that fits the corridor

The first choice is a Cartesian transfer that preserves the measured grasp orientation. This avoids spilling an object while turning a top-down pinch over. Candidate object positions vary forward/lateral reach and height. Collision checks include the held object.

For fixed-yaw holonomic travel, the relevant condition is the full world-axis footprint width, not distance from the base centre. A payload can protrude forward while the complete robot-plus-object box still fits a 1 m corridor. The planner permits at most a 0.48 m half-span, leaving 2 cm nominal clearance on each side.

The footprint centre can be offset from the base. Navigation plans for that centre and subtracts the measured offset to obtain base targets. It also checks a connection from the measured carriage to a free maze-grid cell, preventing an unchecked shortcut across the table.

If the orientation-preserving transfer is unavailable, the implementation also searches collision-checked folded postures and IK alternatives. A rigid shadow grasp does not prove that a real object will survive such a rotation. Live contact monitoring and the full trial results expose those failures.

## 9. Navigate with known-map A*

The task runner uses the wall map and exact pose supplied by the environment. It builds a 1 m grid, marks the table, and runs eight-connected A* with Euclidean costs and no diagonal corner cutting.

The supplied base has independent x/y translation joints. The runner uses that holonomic model and keeps yaw fixed while following the path, retaining the world orientation of the grasp. Translation speed ramps up and down along straight stretches, and the held arm configuration remains commanded.

Every action checks the grasp. The result records navigation actions, robot contact actions, and object contact actions. The separate nav_uncertainty package implements lidar mapping, D* Lite and localisation experiments; those algorithms are not the default run_task.py navigator.

## 10. Reach the goal with the whole robot

The goal constrains the object's position, not an arbitrary base standoff or hand orientation. The final controller first adjusts the torso and translates the holonomic base using collision-checked paths. It measures the remaining object error and uses arm motion for any residual correction.

The object remains held at the goal; there is no support surface at the green sphere. The final result uses the measured 3D object position. An unreachable final motion, dropped object, failed navigation, missed deadline, or missed goal produces a nonzero process exit and failed JSON result.

## Numerical and policy limitations

The environment uses four internal physics substeps and up to 200 solver iterations, preserving its fixed simulated time per action. This reduces finger-position integration error; it does not improve the material friction coefficients.

Antipodal geometry is necessary evidence, not a general guarantee of resistance to all object torques. Mesh normals, sparse patches, contact-manifold changes, and a rigid attached-object planning model can still disagree with the live grasp. Thin rims, nearly maximum-width objects, and objects that need repositioning before pickup remain difficult.

The evaluation report distinguishes verified initial pickups from complete task successes. Component tests, a planned trajectory, a process exit alone, or one successful object must not be read as an all-object reliability claim.
