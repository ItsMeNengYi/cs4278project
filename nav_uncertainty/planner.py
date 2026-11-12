"""D* Lite (Koenig & Likhachev, 2002, optimised version) and an A* baseline on the same 8-connected grid.

Both use the same edge costs, so their path costs must agree; the tests check this, including after
cost changes. Cells are flat indices i = ix * ny + iy, and costs live in Python lists for fast
scalar access.
"""
import heapq
import math
import time

import numpy as np

INF = math.inf
SQ2 = math.sqrt(2)
KEY_DIGITS = 9  # see DStarLite._key
MOVES = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


class GridGraph:
    def __init__(self, cost):
        self.nx, self.ny = cost.shape
        self.cost = cost.ravel().tolist()

    def flat(self, cell):
        return int(cell[0]) * self.ny + int(cell[1])

    def cell(self, i):
        return divmod(i, self.ny)

    def neighbours(self, i):
        """Yields (j, edge cost) for the 8 neighbours of i; the cost is inf when the move is blocked."""
        ny, cost = self.ny, self.cost
        ix, iy = divmod(i, ny)
        ci = cost[i]
        for dx, dy, length in MOVES:
            jx, jy = ix + dx, iy + dy
            if not (0 <= jx < self.nx and 0 <= jy < ny):
                continue
            j = jx * ny + jy
            cj = cost[j]
            if ci == INF or cj == INF:
                yield j, INF
            elif dx and dy and (cost[i + dx * ny] == INF or cost[i + dy] == INF):
                yield j, INF        # no squeezing diagonally between two walls
            else:
                yield j, length * (ci + cj) * 0.5

    def h(self, i, j):
        """Octile distance; admissible because every cell costs at least 1 per unit length."""
        ax, ay = divmod(i, self.ny)
        bx, by = divmod(j, self.ny)
        dx, dy = abs(ax - bx), abs(ay - by)
        return dx + dy + (SQ2 - 2) * min(dx, dy)


def astar(graph, start, goal):
    """Plain A* from scratch. Returns (path as flat indices or None, cost, expansions)."""
    g = {start: 0.0}
    came = {}
    heap = [(graph.h(start, goal), 0.0, start)]
    closed = set()
    expansions = 0
    while heap:
        _, gu, u = heapq.heappop(heap)
        if u in closed:
            continue
        if u == goal:
            path = [u]
            while u in came:
                u = came[u]
                path.append(u)
            return path[::-1], gu, expansions
        closed.add(u)
        expansions += 1
        for v, c in graph.neighbours(u):
            if c == INF or v in closed:
                continue
            gv = gu + c
            if gv < g.get(v, INF):
                g[v] = gv
                came[v] = u
                heapq.heappush(heap, (gv + graph.h(v, goal), gv, v))
    return None, INF, expansions


class DStarLite:
    def __init__(self, graph, start, goal):
        self.G = graph
        n = graph.nx * graph.ny
        self.g = [INF] * n
        self.rhs = [INF] * n
        self.U = {}                 # node -> its current key; heap entries not matching it are stale
        self.heap = []
        self.km = 0.0
        self.start = self.last = start
        self.goal = goal
        self.expansions = 0
        self.rhs[goal] = 0.0
        self._push(goal)

    def _key(self, u):
        # On uniform-cost cells the heuristic is exact, so many first components tie with the start's
        # in exact arithmetic. Summed in floats (km grows as the robot moves) they differ by ~1e-14 and
        # the heap stops ordering them by the second component, so the search could halt with a tied,
        # stale node still queued and report a cost to goal that is too low. Rounding makes ties exact.
        m = min(self.g[u], self.rhs[u])
        return (round(m + self.G.h(self.start, u) + self.km, KEY_DIGITS), m)

    def _push(self, u):
        k = self._key(u)
        self.U[u] = k
        heapq.heappush(self.heap, (k[0], k[1], u))

    def _top(self):
        while self.heap:
            k1, k2, u = self.heap[0]
            if self.U.get(u) == (k1, k2):
                return (k1, k2), u
            heapq.heappop(self.heap)
        return None, None

    def _update_vertex(self, u):
        if u != self.goal:
            best = INF
            g = self.g
            for v, c in self.G.neighbours(u):
                if c != INF and c + g[v] < best:
                    best = c + g[v]
            self.rhs[u] = best
        if self.g[u] != self.rhs[u]:
            self._push(u)
        else:
            self.U.pop(u, None)

    def compute(self):
        g, rhs = self.g, self.rhs
        while True:
            k_old, u = self._top()
            if u is None:
                break
            if not (k_old < self._key(self.start) or rhs[self.start] != g[self.start]):
                break
            heapq.heappop(self.heap)
            del self.U[u]
            self.expansions += 1
            k_new = self._key(u)
            if k_old < k_new:
                self._push(u)
            elif g[u] > rhs[u]:
                g[u] = rhs[u]
                for v, _ in self.G.neighbours(u):
                    self._update_vertex(v)
            else:
                g[u] = INF
                self._update_vertex(u)
                for v, _ in self.G.neighbours(u):
                    self._update_vertex(v)

    def move_to(self, start):
        if start != self.start:
            self.start = start
            self.km += self.G.h(self.last, start)
            self.last = start

    def update_costs(self, changes):
        """changes: iterable of (flat index, new cost)."""
        affected = set()
        for i, c in changes:
            self.G.cost[i] = c
            affected.add(i)
            affected.update(v for v, _ in self.G.neighbours(i))
        for u in affected:
            self._update_vertex(u)

    def cost_to_goal(self):
        return self.g[self.start]

    def path(self, max_len=100000):
        if self.g[self.start] == INF:
            return None
        u, out = self.start, [self.start]
        while u != self.goal and len(out) < max_len:
            best, nxt = INF, None
            for v, c in self.G.neighbours(u):
                if c != INF and c + self.g[v] < best:
                    best, nxt = c + self.g[v], v
            if nxt is None:
                return None
            u = nxt
            out.append(u)
        return out if u == self.goal else None


class Planner:
    """Common interface for the agent: 'dstar' repairs its search, 'astar' replans from scratch."""

    def __init__(self, kind, cost, start_cell, goal_cell):
        self.kind = kind
        self.graph = GridGraph(cost)
        self.goal = self.graph.flat(goal_cell)
        self.start = self.graph.flat(start_cell)
        self.plan_time = 0.0
        self.replans = 0
        self._astar_expansions = 0
        self._path = None
        if kind == "dstar":
            self.ds = DStarLite(self.graph, self.start, self.goal)
        self._replan()

    @property
    def expansions(self):
        return self.ds.expansions if self.kind == "dstar" else self._astar_expansions

    def _replan(self):
        t0 = time.perf_counter()
        if self.kind == "dstar":
            self.ds.move_to(self.start)
            self.ds.compute()
            self._path = self.ds.path()
            self.cost = self.ds.cost_to_goal()
        else:
            self._path, self.cost, e = astar(self.graph, self.start, self.goal)
            self._astar_expansions += e
        self.plan_time += time.perf_counter() - t0

    def update(self, start_cell, new_cost=None):
        """Move the start; if new_cost (full array) differs from the current costs, repair/replan.
        Returns True if a replan happened."""
        self.start = self.graph.flat(start_cell)
        changed = []
        if new_cost is not None:
            flat = new_cost.ravel()
            diff = np.flatnonzero(flat != np.asarray(self.graph.cost)).tolist()
            changed = [(i, float(flat[i])) for i in diff]
        if changed:
            self.replans += 1
            if self.kind == "dstar":
                self.ds.move_to(self.start)
                self.ds.update_costs(changed)
            else:
                for i, c in changed:
                    self.graph.cost[i] = c
            self._replan()
        elif self.kind == "dstar":
            t0 = time.perf_counter()
            self.ds.move_to(self.start)
            self.ds.compute()
            self._path = self.ds.path()
            self.cost = self.ds.cost_to_goal()
            self.plan_time += time.perf_counter() - t0
        return bool(changed)

    def path_cells(self):
        if self._path is None:
            return None
        return [self.graph.cell(i) for i in self._path]
