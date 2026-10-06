import numpy as np


def draw_doses(weights, active, dt, dose_rate, dose_size, rng):
    """Draw incoming doses; rows receive from columns, with probabilities at most one."""
    probability = dose_rate * weights * dt
    arrivals = (rng.random(weights.shape) < probability) & active
    neighbour_count = (weights > 0).sum(axis=1)
    received_dose = dose_size * arrivals.sum(axis=1)
    return np.divide(
        received_dose, neighbour_count,
        out=np.zeros(len(active)), where=neighbour_count > 0,
    )


def accumulate_doses(dose_history, step, memory_steps):
    """Sum the received doses in the memory window ending at this update."""
    window_start = max(0, step - memory_steps + 1)
    return dose_history[:, window_start:step + 1].sum(axis=1)
