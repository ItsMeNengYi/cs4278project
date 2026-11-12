"""D* Lite must give the same optimal cost as A* from scratch, before and after cost changes."""
import numpy as np

from nav_uncertainty.planner import INF, DStarLite, GridGraph, astar


def random_cost(rng, nx=25, ny=25, wall_frac=0.25):
    cost = 1.0 + rng.random((nx, ny)) * 3.0
    cost[rng.random((nx, ny)) < wall_frac] = np.inf
    cost[0, 0] = cost[-1, -1] = 1.0
    return cost


def ref_cost(cost, start, goal):
    return astar(GridGraph(cost), start, goal)[1]


def path_cost(graph, path):
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += dict(graph.neighbours(a))[b]
    return total


def test_static_matches_astar():
    rng = np.random.default_rng(0)
    for _ in range(30):
        cost = random_cost(rng)
        g = GridGraph(cost)
        start, goal = g.flat((0, 0)), g.flat((24, 24))
        ds = DStarLite(GridGraph(cost), start, goal)
        ds.compute()
        ref = ref_cost(cost, start, goal)
        assert np.isclose(ds.cost_to_goal(), ref) or (ref == INF and ds.cost_to_goal() == INF)
        if ref < INF:
            assert np.isclose(path_cost(ds.G, ds.path()), ref)


def test_repair_matches_astar():
    """Walk along the path while cells are randomly blocked and unblocked; compare after every repair."""
    _repair_trials(uniform=False)


def test_repair_uniform_costs():
    """Uniform costs make the heuristic exact, so many queue keys tie with the start's key."""
    _repair_trials(uniform=True)


def test_repair_open_corridor_many_moves():
    """Like a robot mapping a maze: wide open uniform space, the start advances one cell per update
    (so km collects sums of sqrt(2)) and walls appear ahead of it. The float-tie bug fixed in
    DStarLite._key is reproduced by test_integration.test_mapping_dstar_matches_astar instead."""
    rng = np.random.default_rng(3)
    for trial in range(6):
        nx, ny = 160, 60
        cost = np.ones((nx, ny))
        graph = GridGraph(cost.copy())
        goal = graph.flat((nx - 2, ny - 5))
        ds = DStarLite(graph, graph.flat((2, 3)), goal)
        ds.compute()
        for step in range(120):
            path = ds.path()
            ds.move_to(path[1])
            if step % 3 == 0:
                cx, cy = graph.cell(path[min(len(path) - 1, int(rng.integers(5, 40)))])
                w = cost.copy()
                w[max(cx - 1, 0):cx + 2, max(cy - int(rng.integers(2, 12)), 0):cy + int(rng.integers(2, 12))] = np.inf
                w[graph.cell(ds.start)] = cost[graph.cell(ds.start)]
                w[graph.cell(goal)] = 1.0
                diff = np.flatnonzero(w.ravel() != cost.ravel())
                cost = w
                ds.update_costs([(int(i), float(cost.ravel()[i])) for i in diff])
            ds.compute()
            ref = ref_cost(cost, ds.start, goal)
            got = ds.cost_to_goal()
            assert (ref == INF and got == INF) or np.isclose(got, ref), (trial, step, got, ref)
            if ref == INF:
                break


def _repair_trials(uniform):
    rng = np.random.default_rng(2 if uniform else 1)
    for trial in range(15):
        cost = random_cost(rng, wall_frac=0.15)
        if uniform:
            cost[np.isfinite(cost)] = 1.0
        graph = GridGraph(cost.copy())
        start, goal = graph.flat((0, 0)), graph.flat((24, 24))
        ds = DStarLite(graph, start, goal)
        ds.compute()
        for _ in range(60 if uniform else 25):
            path = ds.path()
            if path is not None and len(path) > 2:
                ds.move_to(path[2])
                if uniform:                 # extra moves without changes, like the robot between map updates
                    ds.compute()
                    path = ds.path()
                    if path is not None and len(path) > 2:
                        ds.move_to(path[1])
            changes = []
            for _ in range(rng.integers(1, 30)):
                i = int(rng.integers(0, cost.size))
                if i in (ds.start, goal):
                    continue
                new = np.inf if rng.random() < 0.6 else (1.0 if uniform else 1.0 + rng.random() * 3.0)
                cost.ravel()[i] = new
                changes.append((i, new))
            ds.update_costs(changes)
            ds.compute()
            ref = ref_cost(cost, ds.start, goal)
            got = ds.cost_to_goal()
            assert (ref == INF and got == INF) or np.isclose(got, ref), (trial, got, ref)
            if ref < INF:
                assert np.isclose(path_cost(ds.G, ds.path()), ref)
