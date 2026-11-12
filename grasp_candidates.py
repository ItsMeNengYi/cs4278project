"""Deterministic, diverse antipodal proposals from an object's surface samples.

This module has no simulator dependency. Positions and OUTWARD normals must be
in the same world frame, with lengths in metres. A proposal is a contact pair,
not a promise that the hand or arm can reach it. ``grasping.plan_grasp`` performs
the subsequent hand-geometry, IK, and trajectory checks.
"""

import numpy as np


def _positive(value, name, *, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _patch_support(points, normals, radius, depth, normal_cos):
    """Count nearby, similarly oriented samples in a small surface patch.

    Distance along the contact normal is limited separately from tangent-plane
    distance. Thus a nearby opposite wall does not masquerade as pad support.
    The sample itself counts as one. These counts estimate local support; they
    do not certify a filled surface between samples.
    """
    counts = np.zeros(len(points), dtype=int)
    for i, (point, normal) in enumerate(zip(points, normals)):
        delta = points - point
        along = delta @ normal
        tangent_sq = np.maximum(0.0, np.einsum("ij,ij->i", delta, delta) - along**2)
        supported = ((tangent_sq <= radius**2) & (np.abs(along) <= depth)
                     & (normals @ normal >= normal_cos))
        counts[i] = np.count_nonzero(supported)
    return counts


def generate_grasp_candidates(
    points, normals, object_center, max_width, *, support_top=None,
    friction_mu=0.5, normal_uncertainty_deg=5.0, max_candidates=40,
    min_width=0.008, pad_half_height=0.012, table_clearance=0.004,
    patch_radius=0.018, patch_depth=0.004, patch_normal_deg=30.0,
    min_patch_points=2, diversity_distance=0.012, diversity_angle_deg=15.0,
    up=(0.0, 0.0, 1.0),
):
    """Return ranked, distinct candidate dictionaries (best first).

    The closing axis is ``axis = unit(p2 - p1)``. The contact line must lie
    inside both Coulomb friction cones, with an angular margin reserved for
    uncertain normals: ``angle <= atan(mu) - normal_uncertainty_deg``. The
    coefficient is a conservative planning assumption, not a measurement or a
    change to the simulator's friction.

    Ranking combines friction margin, local patch support, proximity to the
    supplied center *proxy*, horizontal closing direction, and spare opening.
    ``object_center`` is NOT assumed to be the physical center of mass; an
    object's mesh origin and its mass center can differ. ``center_distance``
    measures distance to the finite contact segment, unlike an infinite-line
    metric that can rate two contacts on one side of the center as perfect.

    If a support height is supplied, both contacts need pad-height clearance.
    This is an early coarse filter: orientation-dependent palm/finger/table
    clearance must still be checked later. Diverse proposals are retained by
    midpoint and unoriented closing-axis suppression; retrying need not throw
    away random point-cloud samples.

    ``score`` is a dimensionless heuristic (larger is better), NOT a success
    probability or a full 3D force-closure certificate. Two ideal point
    contacts cannot resist twisting about their connecting line; real pad
    contact and a physical lift/hold check remain necessary.

    Runtime is O(N**2), appropriate for the environment's 1024 samples. Pair
    rows are vectorized and only bounded row winners are retained, rather
    than constructing an N x N x 3 tensor or storing every accepted pair.
    """
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    center, up = np.asarray(object_center, float), np.asarray(up, float)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("points and normals must both have shape (N, 3)")
    if center.shape != (3,) or up.shape != (3,):
        raise ValueError("object_center and up must have shape (3,)")
    if not all(np.all(np.isfinite(a)) for a in (points, normals, center, up)):
        raise ValueError("positions, normals, center and up must be finite")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-12) or np.linalg.norm(up) < 1e-12:
        raise ValueError("normals and up must have nonzero length")
    normals = normals / lengths[:, None]
    up = up / np.linalg.norm(up)
    max_width = _positive(max_width, "max_width")
    min_width = _positive(min_width, "min_width")
    if min_width > max_width:
        raise ValueError("min_width must not exceed max_width")
    friction_mu = _positive(friction_mu, "friction_mu")
    normal_uncertainty_deg = _positive(normal_uncertainty_deg, "normal_uncertainty_deg", allow_zero=True)
    allowed_angle = np.arctan(friction_mu) - np.deg2rad(normal_uncertainty_deg)
    if allowed_angle <= 0:
        raise ValueError("normal uncertainty must be smaller than atan(friction_mu)")
    for name, value in (("patch_radius", patch_radius), ("patch_depth", patch_depth),
                        ("diversity_distance", diversity_distance)):
        _positive(value, name)
    for name, value in (("pad_half_height", pad_half_height), ("table_clearance", table_clearance)):
        _positive(value, name, allow_zero=True)
    for name, value in (("patch_normal_deg", patch_normal_deg), ("diversity_angle_deg", diversity_angle_deg)):
        if not np.isfinite(value) or not 0 <= value < 90:
            raise ValueError(f"{name} must be in [0, 90) degrees")
    for name, value in (("max_candidates", max_candidates), ("min_patch_points", min_patch_points)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if support_top is not None and not np.isfinite(support_top):
        raise ValueError("support_top must be finite or None")
    if len(points) < 2:
        return []

    patch_counts = _patch_support(points, normals, patch_radius, patch_depth,
                                  np.cos(np.deg2rad(patch_normal_deg)))
    eligible = patch_counts >= min_patch_points
    if support_top is not None:
        # The current task uses a horizontal support plane. `up` controls the
        # ranking preference only; `support_top` is a world-z height.
        eligible &= points[:, 2] >= support_top + pad_half_height + table_clearance

    cone_cos = np.cos(allowed_angle)
    extent_scale = max(0.025, float(np.linalg.norm(np.ptp(points, axis=0))) * 0.35)
    row_keep = max(8, min(32, max_candidates))
    winners = []
    for i in range(len(points) - 1):
        if not eligible[i]:
            continue
        js = np.flatnonzero(eligible[i + 1:]) + i + 1
        if not len(js):
            continue
        delta = points[js] - points[i]
        widths = np.linalg.norm(delta, axis=1)
        ok = (widths >= min_width) & (widths <= max_width)
        js, delta, widths = js[ok], delta[ok], widths[ok]
        if not len(js):
            continue
        axes = delta / widths[:, None]
        alignment = np.minimum(axes @ (-normals[i]), np.einsum("ij,ij->i", axes, normals[js]))
        ok = alignment >= cone_cos
        js, axes, widths, alignment = js[ok], axes[ok], widths[ok], alignment[ok]
        if not len(js):
            continue

        angles = np.arccos(np.clip(alignment, -1.0, 1.0))
        margin = allowed_angle - angles
        midpoints = (points[i] + points[js]) / 2
        projection = np.clip((center - points[i]) @ axes.T, 0.0, widths)
        nearest = points[i] + projection[:, None] * axes
        distances = np.linalg.norm(nearest - center, axis=1)
        midpoint_distances = np.linalg.norm(midpoints - center, axis=1)
        centrality = np.exp(-(distances + 0.35 * midpoint_distances) / extent_scale)
        support = np.minimum(patch_counts[i], patch_counts[js])
        patch_quality = np.minimum(1.0, support / 6.0)
        horizontal = 1 - np.abs(axes @ up)
        spare_opening = np.minimum(1.0, (max_width - widths) / 0.025)
        scores = (0.35 * margin / allowed_angle + 0.20 * patch_quality
                  + 0.25 * centrality + 0.15 * horizontal + 0.05 * spare_opening)
        chosen = np.arange(len(js)) if len(js) <= row_keep else np.argpartition(-scores, row_keep - 1)[:row_keep]
        for k in chosen:
            winners.append((float(scores[k]), i, int(js[k]), float(widths[k]),
                            float(margin[k]), float(angles[k]), int(support[k]),
                            float(distances[k]), float(midpoint_distances[k])))

    if not winners:
        return []
    winners.sort(key=lambda c: (-c[0], -c[4], c[7], c[1], c[2]))
    selected, midpoints, axes = [], [], []
    same_axis_cos = np.cos(np.deg2rad(diversity_angle_deg))
    for score, i, j, width, margin, angle, support, distance, mid_distance in winners:
        midpoint = (points[i] + points[j]) / 2
        axis = (points[j] - points[i]) / width
        if any(np.linalg.norm(midpoint - old_mid) < diversity_distance
               and abs(axis @ old_axis) >= same_axis_cos
               and abs(width - old["width"]) < diversity_distance
               for old_mid, old_axis, old in zip(midpoints, axes, selected)):
            continue
        selected.append({
            "p1": points[i].copy(), "p2": points[j].copy(),
            "n1": normals[i].copy(), "n2": normals[j].copy(),
            "indices": (i, j), "width": width, "axis": axis,
            "midpoint": midpoint, "score": score,
            "friction_margin_deg": float(np.rad2deg(margin)),
            "normal_alignment_deg": float(np.rad2deg(angle)),
            "required_friction_mu": float(np.tan(angle)),
            "patch_counts": (int(patch_counts[i]), int(patch_counts[j])),
            "patch_support": support, "center_distance": distance,
            "midpoint_distance": mid_distance,
            "normal_verticality": float((abs(normals[i] @ up) + abs(normals[j] @ up)) / 2),
            "opening_margin": max_width - width,
            "priority": (-score, i, j),
        })
        midpoints.append(midpoint)
        axes.append(axis)
        if len(selected) >= max_candidates:
            break
    return selected
