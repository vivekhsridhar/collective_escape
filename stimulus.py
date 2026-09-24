import numpy as np


def square_wave(time, amplitude, duration):
    """Return a square-wave stimulus."""
    return np.where((time >= 0) & (time < duration), amplitude, 0)
