import numpy as np


def square_wave(time, amplitude, duration, start=0.0):
    """Return a square pulse, including its start and excluding its end."""
    active = (time >= start) & (time < start + duration)
    return np.where(active, amplitude, 0.0)


def loom(time, amplitude, object_radius, start_distance, end_distance,
         approach_speed, start=0.0):
    """Return angular-size input for a disk approaching at constant speed.

    The input approaches amplitude at end_distance, then disappears.
    """
    duration = (start_distance - end_distance) / approach_speed
    end = start + duration
    active = (time >= start) & (time < end)
    elapsed = time[active] - start
    distance = start_distance - approach_speed * elapsed
    angle = 2 * np.arctan(object_radius / distance)
    end_angle = 2 * np.arctan(object_radius / end_distance)

    stimulus = np.zeros_like(time, dtype=float)
    stimulus[active] = amplitude * angle / end_angle
    return stimulus
