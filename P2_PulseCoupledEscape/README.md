# P2: Pulse-coupled escape, step by step

## Independent fish: noisy evidence, escape, and recovery

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

Every fish receives a Gaussian noise increment at each timestep, independent
across fish and time and scaled by `noise_strength * np.sqrt(dt)`. The
square-root scaling makes the noise variance proportional to the timestep
duration. Evidence retains memory through the leaky update even though the
noise draws are independent. `random_seed` makes runs reproducible.
Set `noise_strength` to zero for the deterministic response.
Evidence can fluctuate below zero, and noise can trigger escape even in
unexposed fish. One fish's response cannot affect another's.

Sheltered and unsheltered fish use the same evidence update. The deterministic
part pulls evidence toward the perceived input strength with timescale
`tau_evidence`, or toward zero when input is absent. Evidence can rise or fall
in either state, with noise added at every timestep.

State only determines which behavioral threshold applies. Reaching `threshold`
while unsheltered triggers escape. A sheltered fish recovers when evidence
falls below `recovery_threshold`, including when noise causes the crossing.
This threshold is lower than `threshold`.

The fish remains sheltered while evidence stays at or above `recovery_threshold`.
Another escape spike requires recovery first. The notebook plots the
input to exposed fish, evidence curves, escape spikes by fish, and a shelter-state
heatmap. The update loop is visible in the notebook.

Edit the parameter cells and run from the top to reset the simulation.
Without noise, input weaker than `threshold` produces a subthreshold response
with no escape spike. With noise, threshold crossings can occur even under
weak input. Time is in seconds. Social interactions will come later.

The previous full implementation is preserved as a
[reference snapshot](../archive/README.md). P1 is unchanged.
