# P2: Pulse-coupled escape, step by step

## Independent fish: external input, escape, and recovery

Start with [agent.py](agent.py). The `Agent` class stores two values:

- `evidence`: accumulated evidence of a threat, initially `0.0`.
- `state`: `0` means unsheltered; `1` will mean sheltered.

Open [simulation.ipynb](simulation.ipynb) and run its short cells in order.
Use `P2_PulseCoupledEscape/` as the notebook working directory, as in P1,
so `from agent import Agent` imports the file beside the notebook.

The notebook creates `n_fish` separate agents and applies the square stimulus
pulse only to the first `n_exposed` fish. The others receive zero input.
They respond independently, without social coupling.
Each history array has fish on rows and timesteps on columns.

Exposed fish share the same parameters and deterministic update, so their
evidence curves overlap. Unexposed fish remain unsheltered with zero evidence.
One fish's response cannot affect another's.

Reaching `threshold` while unsheltered triggers escape. While sheltered, any
positive external input prevents evidence from decreasing; stronger input can
still raise it. Once the input is absent, evidence decays with timescale
`tau_evidence`. The fish returns to the unsheltered state only when evidence falls below
`recovery_threshold`, which is lower than `threshold`.

The fish remains sheltered while evidence stays at or above `recovery_threshold`.
Another escape spike requires recovery first. The notebook plots the
input to exposed fish, evidence curves, escape spikes by fish, and a shelter-state
heatmap. The update loop is visible in the notebook.

Edit the parameter cells and run from the top to reset the simulation.
Input weaker than `threshold` produces a subthreshold response: evidence rises
and then decays, with no escape spike.
Time is in seconds. This step is deterministic; social interactions and noise
will come later.

The previous full implementation is preserved as a
[reference snapshot](../archive/README.md). P1 is unchanged.
