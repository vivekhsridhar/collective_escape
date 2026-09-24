import numpy as np


class Agent:
    """One fish with a binary state and a remaining recovery timer."""

    def __init__(self):
        self.spin = np.random.choice([0, 1])
        self.recovery_timer = 0
