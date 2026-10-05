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


def random_directed_network(n_fish, mean_in_degree, random_seed=None):
    """Match the mean in-degree with a cycle and random extra directed links."""
    if n_fish < 2 or not 1 <= mean_in_degree <= n_fish - 1:
        raise ValueError("Use n_fish >= 2 and 1 <= mean_in_degree <= n_fish - 1.")
    rng = np.random.default_rng(random_seed)
    adjacency = np.zeros((n_fish, n_fish))

    # A shuffled cycle makes every fish reachable from every other fish.
    fish_order = rng.permutation(n_fish)
    adjacency[np.roll(fish_order, -1), fish_order] = 1.0

    # Extra links fill the remaining edge budget without self-connections.
    available = (adjacency == 0) & ~np.eye(n_fish, dtype=bool)
    receivers, senders = np.where(available)
    n_links = round(n_fish * mean_in_degree)
    extra_links = rng.choice(len(receivers), n_links - n_fish, replace=False)
    adjacency[receivers[extra_links], senders[extra_links]] = 1.0
    return adjacency


def ring(n_fish, neighbours_each_side=1):
    """Connect each fish to neighbours on both sides of a circular ordering."""
    adjacency = np.zeros((n_fish, n_fish))
    for fish_id in range(n_fish):
        for offset in range(1, neighbours_each_side + 1):
            adjacency[fish_id, (fish_id + offset) % n_fish] = 1.0
            adjacency[fish_id, (fish_id - offset) % n_fish] = 1.0
    np.fill_diagonal(adjacency, 0.0)
    return adjacency
