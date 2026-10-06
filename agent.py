class Agent:
    """One fish with a fixed threshold, evidence, state, and an active timer."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.evidence = 0.0
        self.state = 0
        self.active_timer = 0
