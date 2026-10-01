import numpy as np


def all_to_all(n_fish):
    """Return a network without self-connections; row i receives from column j."""
    adjacency = np.ones((n_fish, n_fish))
    np.fill_diagonal(adjacency, 0.0)
    return adjacency
