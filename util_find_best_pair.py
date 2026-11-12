import heapq

import numpy as np


def find_best_pair(
    points,
    normals,
    object_center,
    max_width,
    *,
    min_width=1e-6,
    max_opposition_deg=20.0,
    max_alignment_deg=20.0,
    up=(0.0, 0.0, 1.0),
    decimals=3,
):
    """Return the best contact pair, or None if no pair passes the filters.

    All positions must use the same world frame and length unit (usually m).
    Normals must point OUTWARD; their lengths need not already equal one.
    max_width is the usable contact separation, after reserving opening clearance.

    A pair passes if its separation is in [min_width, max_width], its normals
    are nearly opposite, and each inward normal points toward the other contact.
    Priority (smaller is better):
      1. Distance from object_center to the infinite contact line, rounded.
      2. Mean absolute normal dot up, rounded (horizontal normals preferred).
      3. Worst normal-alignment error, rounded, then original point indices.

    Only ranking scores are rounded; positions, normals, and filters keep full
    precision. With metres and decimals=3, the distance ranking uses 1 mm bins.
    This is a geometric heuristic: it does not check finger length, collisions,
    robot reachability, friction, or whether the center lies between contacts.
    Cost: O(N**2) pair calculations and O(K) storage for K accepted pairs.
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    center = np.asarray(object_center, dtype=float)
    up = np.asarray(up, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("points and normals must both have shape (N, 3).")
    if center.shape != (3,) or up.shape != (3,):
        raise ValueError("object_center and up must have shape (3,).")
    if not all(np.all(np.isfinite(v)) for v in (points, normals, center, up)):
        raise ValueError("Positions, normals, and up must be finite.")
    if not (np.isfinite(min_width) and np.isfinite(max_width)
            and 0 < min_width <= max_width):
        raise ValueError("Require 0 < min_width <= max_width, both finite.")
    if not all(np.isfinite(a) and 0 <= a < 90
               for a in (max_opposition_deg, max_alignment_deg)):
        raise ValueError("Angular tolerances must be finite and in [0, 90) degrees.")
    if isinstance(decimals, (bool, np.bool_)) or not isinstance(decimals, (int, np.integer)) or decimals < 0:
        raise ValueError("decimals must be a nonnegative integer.")
    lengths = np.linalg.norm(normals, axis=1)
    up_length = np.linalg.norm(up)
    if np.any(lengths < 1e-12) or up_length < 1e-12:
        raise ValueError("Normals and up must have nonzero length.")
    normals = normals / lengths[:, None]
    up = up / up_length
    opposition_cos = np.cos(np.deg2rad(max_opposition_deg))
    alignment_cos = np.cos(np.deg2rad(max_alignment_deg))
    heap = []

    # Process one row at a time; do not allocate an N x N x 3 array.
    for i in range(len(points) - 1):
        delta = points[i + 1:] - points[i]
        widths = np.linalg.norm(delta, axis=1)
        selected = np.flatnonzero((widths >= min_width) & (widths <= max_width))
        if not len(selected):
            continue
        js = selected + i + 1
        direction = delta[selected] / widths[selected, None]
        opposition = -(normals[js] @ normals[i])
        alignment_1 = direction @ (-normals[i])
        alignment_2 = np.einsum("ij,ij->i", direction, normals[js])
        accepted = ((opposition >= opposition_cos - 1e-12)
                    & (alignment_1 >= alignment_cos - 1e-12)
                    & (alignment_2 >= alignment_cos - 1e-12))
        distances = np.linalg.norm(np.cross(direction, center - points[i]), axis=1)
        verticalities = (abs(normals[i] @ up) + np.abs(normals[js] @ up)) / 2
        errors = 1 - np.minimum.reduce((opposition, alignment_1, alignment_2))
        for k in np.flatnonzero(accepted):
            j = int(js[k])
            priority = (
                round(float(distances[k]), decimals),
                round(float(verticalities[k]), decimals),
                round(float(errors[k]), decimals),
                i, j,
            )
            heap.append(priority)

    if not heap:
        return None
    heapq.heapify(heap)
    priority = heapq.heappop(heap)
    i, j = priority[-2:]
    delta = points[j] - points[i]
    width = float(np.linalg.norm(delta))
    direction = delta / width
    return {
        "p1": points[i].copy(),
        "p2": points[j].copy(),
        "n1": normals[i].copy(),
        "n2": normals[j].copy(),
        "indices": (i, j),
        "width": width,
        "center_distance": float(np.linalg.norm(np.cross(center - points[i], direction))),
        "normal_verticality": float((abs(normals[i] @ up) + abs(normals[j] @ up)) / 2),
        "priority": priority,
    }

    