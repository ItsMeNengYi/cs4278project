import io
import math
import heapq
import time
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pick_nav_reach_env as E
from pick_nav_reach_env import PickNavReachEnv

E.time = time           # env calls time.sleep but only imports the function `time`; give it the module
FREE, WALL = 0, -1

class GridMap:
    def __init__(self, cube_xy, goal_pos):
        xs = np.unique(np.round(cube_xy[:, 0], 3))
        ys = np.unique(np.round(cube_xy[:, 1], 3))

        self.cell = float(min(np.diff(xs).min(), np.diff(ys).min()))   # block spacing
        self.origin_x = xs.min() - self.cell / 2                       # left edge
        self.top_y = ys.max() + self.cell / 2                          # top edge
        
        # maze size
        self.n_cols = int(round((xs.max() + self.cell / 2 - self.origin_x) / self.cell))
        self.n_rows = int(round((self.top_y - (ys.min() - self.cell / 2)) / self.cell))

        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=np.int8)
        self.build_map(cube_xy)
        self.goal_cell = self.world_to_cell(goal_pos[0], goal_pos[1])

    def build_map(self, cube_xy):
        self.grid[:] = FREE
        for x, y in cube_xy:
            r, c = self.world_to_cell(x,y)
            if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                self.grid[r,c] = WALL

    def world_to_cell(self, x, y):
        c = int(np.floor((x - self.origin_x) / self.cell))
        r = int(np.floor((self.top_y - y) / self.cell))
        return r, c

    def cell_to_world(self, r, c):
        x = self.origin_x + (c + 0.5) * self.cell
        y = self.top_y - (r + 0.5) * self.cell
        return x, y

    def is_free(self, r, c):
        return (0 <= r < self.n_rows and 0 <= c < self.n_cols and self.grid[r, c] == FREE)

    def astar(self, start, goal=None):
        """A* on self.grid. start/goal are (row, col). Returns list of (row, col) or None."""
        goal = self.goal_cell if goal is None else goal
        if not (self.is_free(*start) and self.is_free(*goal)):
            return None

        sq2 = math.sqrt(2)
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                    (-1, -1, sq2), (-1, 1, sq2), (1, -1, sq2), (1, 1, sq2)]

        def h(n):                                   # Euclidean distance
            return math.hypot(n[0] - goal[0], n[1] - goal[1])

        g = {start: 0.0}                            # cheapest known cost to each cell
        came_from = {}                              # parent of each cell on its best path
        open_heap = [(h(start), 0.0, start)]        # entries are (f, g, cell)
        closed = set()

        while open_heap:
            _, g_cur, cur = heapq.heappop(open_heap)
            if cur in closed:                       # stale heap entry, skip
                continue
            if cur == goal:
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                return path[::-1]
            closed.add(cur)

            r, c = cur
            for dr, dc, w in moves:
                nxt = (r + dr, c + dc)
                if nxt in closed or not self.is_free(*nxt):
                    continue
                # no corner cutting: a diagonal needs both side cells free
                if dr and dc and not (self.is_free(r + dr, c) and self.is_free(r, c + dc)):
                    continue
                g_new = g_cur + w
                if g_new < g.get(nxt, math.inf):
                    g[nxt] = g_new
                    came_from[nxt] = cur
                    heapq.heappush(open_heap, (g_new + h(nxt), g_new, nxt))
        return None   

    def find_entrance(self):
        walls_per_col = (self.grid == WALL).sum(axis=0)
        self.maze_col = next(c for c in range(1, self.n_cols) if walls_per_col[c] > 2)
        rows = np.where(self.grid[:, self.maze_col] == FREE)[0]
        return (int(rows[0]), self.maze_col)

BASE_X0 = -2.0                                  # world x of the base when its x joint is 0 (_load_agent)
TUCK = np.zeros(15)
TUCK[6:13] = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]   # shoulder pan/lift, upperarm roll, elbow, forearm roll, wrist flex/roll


def show_plan(m, path, start):
    maze = m.grid[:, m.maze_col:]                  # crop away the room
    plt.imshow(maze, cmap="gray_r")
    rs, cs = zip(*path)
    plt.plot(np.array(cs) - m.maze_col, rs, "r-")  # shift columns to match the crop
    plt.scatter(0, start[0], c="green", s=80)      # entrance
    plt.scatter(m.goal_cell[1] - m.maze_col, m.goal_cell[0], c="red", s=80, marker="*")  # goal
    plt.title("Planned path")
    plt.show(block=False)                          # don't block: the robot starts right after
    plt.pause(0.5)


def point_at(pts, cum, s):
    return np.array([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])])


def follow_path(env, waypoints, hold_q, max_step=0.01, max_lag=0.15, tol=0.05, max_iters=15000):
    pts = np.array(waypoints)                                    # (N, 2) world xy
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total, s = cum[-1], 0.0
    if total < 1e-10:
        return True, 0, 0
    collisions = 0
    for i in range(max_iters):
        q = env.agent_qpos
        base = np.array([q[0] + BASE_X0, q[1]])
        if np.linalg.norm(point_at(pts, cum, s) - base) < max_lag:   # only advance while the base keeps up
            s = min(s + max_step, total)
        carrot = point_at(pts, cum, s)
        action = hold_q.copy()
        action[0], action[1] = carrot[0] - BASE_X0, carrot[1]        # everything else, incl. yaw, stays held
        with contextlib.redirect_stdout(io.StringIO()):              # silence env.step's per-step prints
            env.step(action)
        if any(c[2] not in (0, env.object_id) for c in p.getContactPoints(bodyA=env.robot_id)):
            collisions += 1                                          # touching anything but ground/object
        if i % 50 == 0 and plt.get_fignums():
            plt.pause(0.001)                                         # keep the plot window responsive
        if i % 200 == 0:
            print(f"step {i}: base=({base[0]:.2f}, {base[1]:.2f}), progress {s / total:.0%}")
        if s >= total and np.linalg.norm(pts[-1] - base) < tol:
            return True, i, collisions
    return False, max_iters, collisions


def corners(waypoints, eps=1e-6):
    """Drop waypoints in the middle of straight runs, keeping only the turning points."""
    pts = [np.asarray(waypoints[0], dtype=float)]
    for k in range(1, len(waypoints) - 1):
        a, b, c = pts[-1], np.asarray(waypoints[k], dtype=float), np.asarray(waypoints[k + 1], dtype=float)
        u, v = b - a, c - b
        if np.linalg.norm(u) < eps:
            continue
        if abs(u[0] * v[1] - u[1] * v[0]) > eps * np.linalg.norm(u) * np.linalg.norm(v) or u @ v < 0:
            pts.append(b)
    pts.append(np.asarray(waypoints[-1], dtype=float))
    return pts


def follow_path_facing(env, waypoints, hold_q, v_max=0.01, accel=0.0004, w_max=0.01, w_accel=0.0003,
                       max_lag=0.05, tol=0.02, max_iters=15000, still_ok=None, face_path=True):
    """Drive the path always facing the direction of travel: at each corner, stop and turn in place,
    then drive the straight stretch forwards. Speed and turn rate ramp up and down (trapezoidal
    profiles), so a held object is never jerked. Units: m and rad per env step.

    still_ok: optional callable(env) -> bool, checked every step; returning False aborts (e.g. dropped).
    Returns (reached, steps, robot_wall_steps, object_hit_steps)."""
    action = hold_q.copy()
    pts = corners(waypoints)
    steps, robot_hits, object_hits = 0, 0, 0

    def base():
        q = env.agent_qpos
        return np.array([q[0] + BASE_X0, q[1]])

    def tick():
        nonlocal steps, robot_hits, object_hits
        with contextlib.redirect_stdout(io.StringIO()):
            env.step(action)
        steps += 1
        if any(c[2] not in (0, env.object_id) for c in p.getContactPoints(bodyA=env.robot_id)):
            robot_hits += 1
        if any(c[2] not in (0, env.robot_id) for c in p.getContactPoints(bodyA=env.object_id)):
            object_hits += 1
        if steps % 200 == 0:
            b = base()
            print(f"step {steps}: base=({b[0]:.2f}, {b[1]:.2f}), yaw {env.agent_qpos[2]:.2f}")
        if steps >= max_iters or (still_ok is not None and not still_ok(env)):
            raise StopIteration

    try:
        for a, b in zip(pts[:-1], pts[1:]):
            a = base()                                              # start each stretch where the base is
            d = b - a
            length = float(np.linalg.norm(d))
            if length < tol:
                continue
            h = d / length
            # turn in place to face the stretch
            if face_path:
                yaw0 = float(env.agent_qpos[2])
                turn = math.atan2(math.sin(math.atan2(h[1], h[0]) - yaw0), math.cos(math.atan2(h[1], h[0]) - yaw0))
                done, w = 0.0, 0.0
                while abs(turn) - done > 1e-4:
                    left = abs(turn) - done
                    w = min(w + w_accel, w_max, math.sqrt(2 * w_accel * left), left)
                    done += w
                    action[2] = yaw0 + math.copysign(done, turn)
                    tick()
                for _ in range(30):                                 # let the turn settle
                    tick()
            # drive forwards along the stretch
            s, v = 0.0, 0.0
            while s < length - 1e-4:
                if np.linalg.norm(a + s * h - base()) < max_lag:   # only advance while the base keeps up
                    left = length - s
                    v = min(v + accel, v_max, math.sqrt(2 * accel * left), left)
                    s += v
                cmd = a + s * h
                action[0], action[1] = cmd[0] - BASE_X0, cmd[1]
                tick()
        while np.linalg.norm(pts[-1] - base()) > tol:               # settle on the last point
            tick()
        return True, steps, robot_hits, object_hits
    except StopIteration:
        return False, steps, robot_hits, object_hits


if __name__ == "__main__":
    env = PickNavReachEnv(seed=0)
    obs = env.reset()
    m = GridMap(obs["cube_positions"], obs["goal_pos"])

    start = m.find_entrance()
    path = m.astar(start)
    assert path is not None, "no path found"
    waypoints = [m.cell_to_world(r, c) for r, c in path]
    print(f"path: {len(path)} cells, entrance {start}, goal {m.goal_cell}")

    # output 1: the planned path (plot window, non-blocking)
    show_plan(m, path, start)

    # output 2: teleport to the entrance with the arm tucked, then drive the path
    q0 = TUCK.copy()
    q0[0], q0[1] = waypoints[0][0] - BASE_X0, waypoints[0][1]       # yaw stays 0
    env._set_qpos(q0)

    ok, steps, collisions = follow_path(env, waypoints, hold_q=q0)
    print(f"reached goal: {ok}, steps: {steps}, steps in contact: {collisions}")
    input("Press Enter to close...")
