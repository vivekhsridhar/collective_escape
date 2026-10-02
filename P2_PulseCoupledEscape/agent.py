class Agent:
    """One fish with a position, accumulated evidence, and a shelter state."""

    def __init__(self, position):
        self.position = position
        self.evidence = 0.0
        self.state = 0
