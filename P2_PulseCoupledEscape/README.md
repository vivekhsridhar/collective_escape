# P2: Pulse-coupled escape, step by step

## Pulse-coupled fish: looming input, social escapes, and recovery

Start with [agent.py](agent.py). The `Agent` class stores two values:

- `evidence`: accumulated evidence of a threat, initially `0.0`.
- `state`: `0` means unsheltered; `1` will mean sheltered.

The other reusable components are:

- [stimulus.py](stimulus.py): `loom` and `square_wave`, each returning an input array matching `time`.
- [network.py](network.py): `all_to_all`, returning an editable adjacency matrix.

Open [simulation.ipynb](simulation.ipynb) and run its short cells in order.
Use `P2_PulseCoupledEscape/` as the notebook working directory, as in P1,
so the imports resolve to the files beside the notebook. Parameters, the
stimulus choice, network setup, simulation loop, and plots remain visible there.

The notebook uses `loom(...)` by default. Replacing that call with a square
pulse only changes the input array:

```python
external_input = square_wave(
    time, stimulus_strength, duration=0.3, start=stimulus_start,
)
```

Both functions include the start and exclude the end of the stimulus.
Other stimulus functions can return another array with one value per timestep.

The notebook creates `n_fish` separate agents and applies a looming stimulus
only to the first `n_exposed` fish. The others receive no direct loom input,
but all fish can receive evidence from neighbour escapes.
Each history array has fish on rows and timesteps on columns.

`adjacency = all_to_all(n_fish)` creates a fully connected network without
self-connections. `adjacency[i, j]` is the weight from fish `j` to fish `i`.
Edit this matrix before running the simulation to remove or reweight connections.
For example, removing the connection between fish 0 and fish 1 in both directions is:

```python
adjacency[0, 1] = 0.0
adjacency[1, 0] = 0.0
```

A weight of `0.5` halves that connection's contribution. A replacement network
must have shape `(n_fish, n_fish)`; keeping its diagonal zero avoids self-input.

Each escape adds one pulse of size `social_strength` to each neighbour's
evidence, multiplied by the connection weight, on the next timestep.
Pulses add together without averaging by group
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
`duration = (start_distance - end_distance) / approach_speed`.

Input is proportional to angular size, scaled toward `stimulus_strength` at
the endpoint. It starts above zero, increases throughout the approach, and
drops to zero when the approach ends. The last sample before the exclusive end is
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
