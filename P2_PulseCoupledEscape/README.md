# P2: Pulse-coupled escape, step by step

## Pulse-coupled fish: square-wave input, social escapes, and recovery

Start with [agent.py](agent.py). The `Agent` class stores two values:

- `evidence`: accumulated evidence of a threat, initially `0.0`.
- `state`: `0` means unsheltered; `1` will mean sheltered.

The other reusable components are:

- [stimulus.py](stimulus.py): `loom` and `square_wave`, each returning an input array matching `time`.
- [network.py](network.py): `all_to_all`, `ring`, and `random_network`, each returning an editable adjacency matrix.

Open [simulation.ipynb](simulation.ipynb) and run its short cells in order.
Use `P2_PulseCoupledEscape/` as the notebook working directory, as in P1,
so the imports resolve to the files beside the notebook. Parameters, the
stimulus choice, network setup, simulation loop, and plots remain visible there.

The notebook uses `square_wave(...)` by default. Its amplitude, onset, and
duration are visible in the parameter cell:

```python
external_input = square_wave(
    time, stimulus_strength, duration=stimulus_duration, start=stimulus_start,
)
```

Both functions include the start and exclude the end of the stimulus.
Other stimulus functions can return another array with one value per timestep.

The notebook creates `n_fish` separate agents and applies the square pulse
only to the first `n_exposed` fish. The others receive no direct stimulus input,
but all fish can receive evidence from neighbour escapes.
Each history array has fish on rows and timesteps on columns.

The notebook uses `adjacency = random_network(n_fish, connection_probability, network_seed)`.
Each pair connects independently with the chosen probability.
`adjacency[i, j]` is the weight from fish `j` to fish `i`.
To choose another network, replace the assignment with one of these calls:

- `adjacency = ring(n_fish, neighbours_each_side)` connects fish to neighbours on either side in a circular ordering.
- `adjacency = all_to_all(n_fish)` connects every pair.

All three networks have connections in both directions and no self-connections.
Their parameters are visible in the notebook. `network_seed` makes the random
network reproducible independently of the evidence noise's `random_seed`.

Edit this matrix before running the simulation to remove or reweight connections.
For example, removing the connection between fish 0 and fish 1 in both directions is:

```python
adjacency[0, 1] = 0.0
adjacency[1, 0] = 0.0
```

Positive weights set relative contributions to a weighted average. Scaling every
weight in a row by the same positive factor leaves that fish's social input
unchanged. A replacement network must have shape `(n_fish, n_fish)`; keeping its
diagonal zero avoids self-input.

Social input is the weighted mean of neighbouring escape cues:

```math
S_i = w\frac{\sum_j A_{ij}c_j}{\sum_j A_{ij}}.
```

For binary connections, the denominator is the number of neighbours of fish
`i`. Fish without neighbours receive zero social input. With binary cues, this
would be the fraction of neighbours escaping. The current cues fade over time
and can overlap after repeated escapes, so their mean need not stay below one.
It does not measure the fraction currently sheltered.

For an all-to-all network, the mean divides the summed input by `n_fish - 1`.
`social_strength` sets the response to the mean cue, independently of neighbour
count, so the same value gives weaker input than in the summed model.

Each escape adds one to the escaping fish's `social_cue`. The cue reaches
neighbours on the next timestep, scaled by `social_strength` and the connection
weight divided by the recipient's total incoming weight. Between timesteps it
decays by `np.exp(-dt / tau_social)`. An isolated cue
retains about 37% of its initial value after `tau_social` seconds, with no abrupt
cutoff. The default timescale is illustrative, following the archived example.

`tau_social` describes how long an escape remains an incoming signal.
`tau_evidence` describes how quickly accumulated evidence responds to the total
input and how it decays after input disappears. Increasing `tau_social` at fixed
amplitude also increases the total social input delivered by an escape.

The averaged social input and direct input enter the drift together:

```python
total_input = direct_input + social_input[fish_id]
drift = (total_input - fish.evidence) / tau_evidence
```

The evidence change is `drift * dt` plus Gaussian noise. The decaying social
signal feeds this drift over multiple timesteps. Its lifetime is measured in
seconds rather than timesteps, so reducing `dt` refines the approximation to
the same cue. Accumulated evidence can continue rising while a cue is fading
if total input remains above the evidence level.

All social inputs use cues computed before any fish updates. At the end of each
timestep, old cues decay and new escape spikes are added. Newly triggered
escapes therefore affect neighbours on the next timestep, regardless of loop
order. Remaining sheltered emits no new cues, but the existing cue continues to
fade. Repeated escapes add new cues. Sheltered fish still receive social input.
Set `social_strength` to zero to recover the independent-fish model.

The optional `loom` function represents a disk approaching at constant `approach_speed`.
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

Sheltered and unsheltered fish use the same evidence update. The drift pulls
evidence toward the sum of direct and social input with timescale `tau_evidence`,
or toward zero when both are absent. Gaussian noise is added at every timestep.

State only determines which behavioral threshold applies. Reaching `threshold`
while unsheltered triggers escape. A sheltered fish recovers when evidence
falls below `recovery_threshold`, including when noise causes the crossing.
This threshold is lower than `threshold`.

The fish remains sheltered while evidence stays at or above `recovery_threshold`.
Another escape spike requires recovery first. The notebook plots the
input to exposed fish, evidence curves, escape spikes by fish, and a shelter-state
heatmap. A further plot compares incoming social input with accumulated evidence
for `focal_fish`, which selects an unexposed fish in the default setup. The update
loop is visible in the notebook.

Edit the parameter cells and run from the top to reset the simulation.
Without noise or social input, a stimulus peak weaker than `threshold` produces a
subthreshold response with no escape spike. Noise and social pulses can cause
threshold crossings even under weak or absent direct input. Time is in seconds.

The previous full implementation is preserved as a
[reference snapshot](../archive/README.md). P1 is unchanged.
