import numpy as np

from agent import Agent
from dose import draw_doses, accumulate_doses
from stimulus import spontaneous_startle


def cascade_size(adjacency, thresholds, starter_id, dt, dose_rate, dose_size,
                 active_duration, memory_duration, rng):
    """Run one cascade and count all escaped fish, including the starter."""
    agents = [Agent(threshold) for threshold in thresholds]
    active_steps = round(active_duration / dt)
    memory_steps = round(memory_duration / dt)
    max_steps = len(agents) * active_steps + 1
    dose_history = np.zeros((len(agents), max_steps))
    spontaneous_startle(agents, active_steps, rng, starter_id)

    # Doses use the states before any fish updates in the current timestep.
    for step in range(1, max_steps):
        active = np.array([fish.state == 1 for fish in agents])
        susceptible = np.array([fish.state == 0 for fish in agents])
        dose_history[:, step] = susceptible * draw_doses(
            adjacency, active, dt, dose_rate, dose_size, rng,
        )
        social_evidence = accumulate_doses(dose_history, step, memory_steps)
        for fish_id, fish in enumerate(agents):
            if fish.state == 0:
                fish.evidence = social_evidence[fish_id]
                if fish.evidence > fish.threshold:
                    fish.state = 1
                    fish.active_timer = active_steps
            elif fish.state == 1:
                fish.active_timer -= 1
                if fish.active_timer == 0:
                    fish.state = 2
        if not any(fish.state == 1 for fish in agents):
            break
    return sum(fish.state != 0 for fish in agents)
