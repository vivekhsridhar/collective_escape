import numpy as np
from scipy.stats import rankdata


def pairwise_distances(x, y):
    """Return Euclidean distances in the same units as the coordinates."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.hypot(dx, dy)


def rank_angular_areas(angular_area):
    """Rank positive areas per viewer, averaging ties and excluding self."""
    visible = angular_area > 0
    np.fill_diagonal(visible, False)
    ranks = np.full(angular_area.shape, np.nan)
    for fish_id in range(len(angular_area)):
        neighbours = visible[fish_id]
        ranks[fish_id, neighbours] = rankdata(
            -angular_area[fish_id, neighbours], method="average"
        )
    return ranks
