from dataclasses import dataclass

import numpy as np


@dataclass
class Agent:
    """One fish with a random state: 0 for baseline, 1 for escape."""
    
    def __post_init__(self):
        self.spin = np.random.choice([0, 1])
