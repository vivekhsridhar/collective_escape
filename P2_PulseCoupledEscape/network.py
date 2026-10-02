import numpy as np


def all_to_all(n_fish):
    """Return a network without self-connections; row i receives from column j."""
    adjacency = np.ones((n_fish, n_fish))
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def random_network(n_fish, connection_probability, random_seed=None):
    """Connect each unordered pair independently with the given probability."""
    rng = np.random.default_rng(random_seed)
    connections = rng.random((n_fish, n_fish)) < connection_probability
    adjacency = np.triu(connections, k=1).astype(float)
    return adjacency + adjacency.T


def ring(n_fish, neighbours_each_side=1):
    """Connect each fish to neighbours on both sides of a circular ordering."""
    adjacency = np.zeros((n_fish, n_fish))
    for fish_id in range(n_fish):
        for offset in range(1, neighbours_each_side + 1):
            adjacency[fish_id, (fish_id + offset) % n_fish] = 1.0
            adjacency[fish_id, (fish_id - offset) % n_fish] = 1.0
    np.fill_diagonal(adjacency, 0.0)
    return adjacency
