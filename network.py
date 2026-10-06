import numpy as np


def response_network(distance_cm, angular_area_rank, intercept,
                     distance_coefficient, rank_coefficient):
    """Return response probabilities; row i receives from column j."""
    visible = np.isfinite(angular_area_rank)
    adjacency = np.zeros(distance_cm.shape)
    log_odds = (
        intercept + distance_coefficient * np.log10(distance_cm[visible])
        + rank_coefficient * angular_area_rank[visible]
    )
    adjacency[visible] = 1 / (1 + np.exp(-log_odds))
    return adjacency


def all_to_all(n_fish):
    """Return a network without self-connections; row i receives from column j."""
    adjacency = np.ones((n_fish, n_fish))
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def is_connected(adjacency):
    """Check that every fish in an undirected network is reachable from fish zero."""
    reached = {0}
    to_visit = [0]
    while to_visit:
        fish_id = to_visit.pop()
        for neighbour in np.flatnonzero(adjacency[fish_id]):
            if neighbour not in reached:
                reached.add(neighbour)
                to_visit.append(neighbour)
    return len(reached) == len(adjacency)


def random_network(n_fish, connection_probability, random_seed=None):
    """Draw an undirected random network, retrying until it is connected."""
    if n_fish < 1 or not 0 < connection_probability <= 1:
        raise ValueError("Use n_fish >= 1 and 0 < connection_probability <= 1.")
    rng = np.random.default_rng(random_seed)
    while True:
        connections = rng.random((n_fish, n_fish)) < connection_probability
        adjacency = np.triu(connections, k=1).astype(float)
        adjacency = adjacency + adjacency.T
        if is_connected(adjacency):
            return adjacency


def synthetic_spatial_network(n_fish, intercept, distance_coefficient,
                              rank_coefficient, arena_size_cm=(100.0, 100.0),
                              min_separation_cm=3.0, body_width_cm=2.0,
                              field_of_view_deg=360.0, max_visual_distance_cm=None,
                              occlusion=True, positions_cm=None, spatial_scale=1.0, 
                              random_seed=None, return_geometry=False):
    """Build a spatial sensory network; row i receives from column j."""
    rng = np.random.default_rng(random_seed)

    # Sample positions with a minimum separation between fish.
    if positions_cm is None:
        width, height = arena_size_cm
        positions_cm = []
        max_attempts = 1000
        attempts = 0
        while len(positions_cm) < n_fish:
            candidate = np.array([rng.uniform(0, width), rng.uniform(0, height)])
            if len(positions_cm) == 0:
                positions_cm.append(candidate)
            else:
                existing = np.asarray(positions_cm)
                distances = np.linalg.norm(existing - candidate, axis=1)
                if np.all(distances >= min_separation_cm):
                    positions_cm.append(candidate)
            attempts += 1
            if attempts >= max_attempts and len(positions_cm) < n_fish:
                raise RuntimeError("Could not place all fish with min_separation_cm.")
        positions_cm = np.asarray(positions_cm)
    else:
        positions_cm = np.array(positions_cm, dtype=float, copy=True)

    # Scale each fish's displacement from the group centroid.
    centroid = positions_cm.mean(axis=0)
    positions_cm = centroid + spatial_scale * (positions_cm - centroid)
    orientations_rad = rng.uniform(-np.pi/2, np.pi/2, n_fish)

    # Displacements point from each receiver to each sender.
    delta = positions_cm[None, :, :] - positions_cm[:, None, :]
    distance_cm = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance_cm, np.inf)
    bearing_rad = np.arctan2(delta[:, :, 1], delta[:, :, 0])

    # Wrap sender bearings relative to each receiver's heading.
    relative_angle = bearing_rad - orientations_rad[:, None]
    relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
    half_fov = np.deg2rad(field_of_view_deg) / 2
    visible = np.abs(relative_angle) <= half_fov
    np.fill_diagonal(visible, False)
    if max_visual_distance_cm is not None:
        visible &= distance_cm <= max_visual_distance_cm

    angular_size_rad = 2 * np.arctan((body_width_cm / 2) / distance_cm)
    np.fill_diagonal(angular_size_rad, 0.0)

    # Closer visible fish block senders within their angular half-width.
    if occlusion:
        for receiver in range(n_fish):
            candidates = np.flatnonzero(visible[receiver])
            order = candidates[np.argsort(distance_cm[receiver, candidates])]
            accepted = []
            for sender in order:
                sender_bearing = bearing_rad[receiver, sender]
                blocked = False
                for blocker in accepted:
                    blocker_bearing = bearing_rad[receiver, blocker]
                    angular_difference = sender_bearing - blocker_bearing
                    angular_difference = np.arctan2(
                        np.sin(angular_difference), np.cos(angular_difference),
                    )
                    blocker_half_angle = angular_size_rad[receiver, blocker] / 2
                    if abs(angular_difference) <= blocker_half_angle:
                        blocked = True
                        break
                if blocked:
                    visible[receiver, sender] = False
                else:
                    accepted.append(sender)

    # Rank visible neighbours from largest to smallest apparent angular size.
    angular_area_rank = np.full((n_fish, n_fish), np.nan)
    for receiver in range(n_fish):
        neighbours = np.flatnonzero(visible[receiver])
        if len(neighbours) == 0:
            continue
        order = neighbours[np.argsort(-angular_size_rad[receiver, neighbours])]
        angular_area_rank[receiver, order] = np.arange(1, len(order) + 1)

    adjacency = response_network(distance_cm, angular_area_rank, intercept, distance_coefficient, rank_coefficient)
    np.fill_diagonal(adjacency, 0.0)
    if not return_geometry:
        return adjacency

    geometry = {
        "positions_cm": positions_cm,
        "orientations_rad": orientations_rad,
        "distance_cm": distance_cm,
        "bearing_rad": bearing_rad,
        "visible": visible,
        "angular_size_rad": angular_size_rad,
        "angular_area_rank": angular_area_rank,
        "centroid_cm": positions_cm.mean(axis=0),
    }
    return adjacency, geometry
