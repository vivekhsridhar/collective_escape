# P2: Pulse-coupled escape, step by step

## Pulse-coupled fish: looming input, social escapes, and recovery

Start with [agent.py](agent.py). The `Agent` class stores two values:

- `evidence`: accumulated evidence of a threat, initially `0.0`.
- `state`: `0` means unsheltered; `1` will mean sheltered.

Open [simulation.ipynb](simulation.ipynb) and run its short cells in order.
Use `P2_PulseCoupledEscape/` as the notebook working directory, as in P1,
so `from agent import Agent` imports the file beside the notebook.

The notebook creates `n_fish` separate agents and applies a looming stimulus
only to the first `n_exposed` fish. The others receive no direct loom input,
but all fish can receive evidence from neighbour escapes.
Each history array has fish on rows and timesteps on columns.

The network is fully connected without self-connections. `adjacency[i, j]`
indicates whether fish `i` receives an escape response from fish `j`.
Each escape adds one pulse of size `social_strength` to each neighbour's
evidence on the next timestep. Pulses add together without averaging by group
size. The pulse is an evidence increment, so it is not multiplied by `dt`.
Its contribution then decays through the existing `tau_evidence` term.

All social inputs use the previous timestep's escape spikes, computed before
any fish updates. Newly triggered escapes therefore affect neighbours on the
next timestep, regardless of loop order. Remaining sheltered emits no further
pulses. Sheltered fish still receive social input. Set `social_strength` to
zero to recover the independent-fish model.

The loom represents a disk approaching at constant `approach_speed`.
Its angular size is `2 * np.arctan(object_radius / distance)`. Lengths are in
metres and speed is in metres per second; the geometry values are illustrative.
The stimulus appears at `start_distance` and disappears at `end_distance`, so
`stimulus_duration = (start_distance - end_distance) / approach_speed`.

Input is proportional to angular size, scaled toward `stimulus_strength` at
the endpoint. It starts above zero, increases throughout the approach, and
drops to zero at `stimulus_end`. The last sample before the exclusive end is
slightly below the endpoint strength. With fixed geometry, speed changes
duration but not endpoint strength. This version does not give slow approaches
a lower amplitude or impose a preference for intermediate speeds.

Every fish receives a Gaussian noise increment at each timestep, independent
across fish and time and scaled by `noise_strength * np.sqrt(dt)`. The
square-root scaling makes the noise variance proportional to the timestep
duration. Evidence retains memory through the leaky update even though the
noise draws are independent. `random_seed` makes runs reproducible.
Set `noise_strength` to zero for the deterministic response.
Evidence can fluctuate below zero, and noise can trigger escape even in
unexposed fish. Those escapes can also supply social evidence to neighbours.

Sheltered and unsheltered fish use the same evidence update. Between social
pulses, the deterministic part pulls evidence toward the loom input with
timescale `tau_evidence`, or toward zero when the loom is absent. Social pulses
add to that same evidence, and noise is added at every timestep.

State only determines which behavioral threshold applies. Reaching `threshold`
while unsheltered triggers escape. A sheltered fish recovers when evidence
falls below `recovery_threshold`, including when noise causes the crossing.
This threshold is lower than `threshold`.

The fish remains sheltered while evidence stays at or above `recovery_threshold`.
Another escape spike requires recovery first. The notebook plots the
input to exposed fish, evidence curves, escape spikes by fish, and a shelter-state
heatmap. The update loop is visible in the notebook.

Edit the parameter cells and run from the top to reset the simulation.
Without noise or social input, a loom peak weaker than `threshold` produces a
subthreshold response with no escape spike. Noise and social pulses can cause
threshold crossings even under weak or absent loom input. Time is in seconds.

The previous full implementation is preserved as a
[reference snapshot](../archive/README.md). P1 is unchanged.
