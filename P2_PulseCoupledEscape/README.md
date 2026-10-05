# P2: One startle and a social escape cascade

The notebook follows one initially startled fish and the escapes it can trigger
through social evidence. Fish progress from unsheltered to actively escaping
to sheltered. Shelter is absorbing: each fish can escape at most once per run.

## Files and setup

[agent.py](agent.py) defines `Agent(threshold)`, which stores four values:

- `threshold`: the fish's fixed activation threshold, supplied at initialization.
- `evidence`: remembered social evidence, initially `0.0`.
- `state`: `0` for unsheltered, `1` for actively escaping, and `2` for sheltered.
- `active_timer`: the number of active-phase updates remaining, initially `0`.

The reusable functions are:

- [stimulus.py](stimulus.py): `spontaneous_startle` activates the initial fish.
  The file also retains `loom` and `square_wave` for external-stimulus experiments.
- [network.py](network.py): `random_network` supplies the notebook's adjacency
  matrix. `all_to_all` and `ring` are also available in the module.
- [dose.py](dose.py): `draw_doses` samples packets from active neighbours;
  `accumulate_doses` sums the packets in a finite memory window.

Open [simulation.ipynb](simulation.ipynb) and run its short cells in order, using
`P2_PulseCoupledEscape/` as the working directory. Parameters, network setup,
initialization, the update loop, and three plots remain visible in the notebook.
[dose_memory.ipynb](dose_memory.ipynb) explains the dose and memory functions
separately, using one sender and one receiver without further escapes.

## Start one fish

`spontaneous_startle(agents, active_steps, rng, fish_id=None)` sets one fish's
state to active, gives it the full `active_steps` timer, and returns its index.
It bypasses the evidence threshold and is called once during initialization.
All other fish start unsheltered with zero evidence.

The notebook's `initial_fish = None` selects the starter uniformly at random.
Setting `initial_fish` to a fish index selects that fish instead. Fish indices
start at zero. The returned starter's escape onset is recorded in column zero.

`startle_rng = np.random.default_rng([random_seed, 0])` controls starter choice.
`dose_rng = np.random.default_rng([random_seed, 1])` controls dose arrivals.
These separate generators make each part reproducible. `network_seed` controls
network construction independently.

There is no external input or personal-evidence noise. Every later escape is
triggered by social evidence exceeding that fish's `threshold`.

## Individual thresholds

The notebook draws `thresholds` uniformly between zero and `2 * mean_threshold`
and passes one value to each agent. Each threshold stays fixed during the run.
`threshold_rng = np.random.default_rng([random_seed, 2])` makes these draws
independent of starter selection and dose arrivals. The same seed reproduces
the same thresholds when the notebook is rerun from the top.

`mean_threshold` is the distribution's expected mean, not the mean received
dose. A finite sample of fish need not have exactly this average threshold.
`mean_threshold = 0.034` adopts Sosna's baseline fit from SI section 6.3,
page 9: Context 1, before the first Schreckstoff exposure, in the Fig. 4D
comparison. Its reported 95% credible interval is `[0.030, 0.035]`.
The paper fitted this value using experimental networks and cascade sizes.
Here it is a reference value, not a recalibration to the illustrative random
network.

## Receive and remember social doses

The notebook uses `random_network(n_fish, connection_probability, network_seed)`.
Each pair connects independently, in both directions, with no self-connections.
`adjacency[i, j]` is the weight from sender `j` to receiver `i`. The network
is fixed during a run and does not use fish positions.

At each update, an active neighbour sends a packet with probability
`dose_rate * adjacency[i, j] * dt`. This probability must be at most one.
Each arriving packet contributes `dose_size / neighbour_count[i]`, where the
count includes all positive incoming connections, including inactive neighbours.
An isolated fish receives no doses.

The default binary connections give certain packet arrivals while a neighbour
is active. Fractional weights between zero and one make arrivals stochastic.
The weight changes arrival probability, not packet size.

`dose_history[:, step]` stores the sum of normalized packets at that update.
`accumulate_doses` sums the most recent `memory_steps` columns, including the
current one. Evidence is this remembered sum, compared directly with the
escape threshold without an additional multiplier.
The sum is neither added to previous evidence nor multiplied by `dt` again.

A packet received at update `k` keeps its full contribution through
`k + memory_steps - 1` and leaves the sum at `k + memory_steps`.
`memory_steps = max(1, round(memory_duration / dt))` converts the memory
window from seconds to whole updates. This memory has no exponential leak.

## Active phases and timing

`n_steps` time samples run from index `0` through `n_steps - 1`, with
`time = np.arange(n_steps) * dt`. Column zero stores the initial state, with
one active fish and zero evidence. The loop performs `n_steps - 1` updates,
starting at index one. Each later column records the result of that update.

Dose arrivals use the active-state snapshot from before the update. A fish
newly activated at update `k` first sends on update `k + 1`. An already-active
fish sends before its timer decreases, so fish update order cannot create
several links of propagation within one timestep.

`active_steps = max(1, round(active_duration / dt))` sets the active-phase
length. With `m = active_steps`, a fish activated at index `k` is recorded
active at `k` through `k + m - 1`. It sends during updates `k + 1` through
`k + m`, then enters shelter at index `k + m`. This also applies to the
initial starter with `k = 0`.

Sheltered fish remain sheltered and send no doses. Evidence is still recorded
in every state, but only unsheltered fish can respond to it. Entering shelter
does not clear doses already held in neighbours' memories.

The run ends at its fixed observation limit. Late escapes may still be active
in the last column; increasing `total_time` reveals their shelter entry.
The plots show evidence, escape onsets, and the three behavioral states.
There is no common threshold line; `thresholds` stores the fish-specific values.
Each fish has at most one onset mark, including the initial starter at time zero.

## Reproducing Sosna et al. (2019)

The model uses the paper's dose scale and rate, neighbour normalization,
finite social memory, active duration, and absorbing terminal state. The
notebook's single forced initial startle also follows its cascade setup.
The following differences remain:

| Component | Current notebook | Sosna model |
| --- | --- | --- |
| Starter | Uniform random fish or a chosen index | Observed initial startler for each experimental cascade |
| Network | Illustrative binary random network | Empirical directed weights from positions and visual features |
| Mean threshold | Baseline reference `0.034` from SI section 6.3, applied to the random network | Mean threshold fitted to experimental cascade sizes, with the same uniform distribution |
| Dose reception | Evidence recorded in every state | Doses received by susceptible fish |
| Run duration | Fixed observation limit | Stop when no active fish remain |
| Comparison | One realization and its time courses | Repeated simulations fitted to experimental cascade sizes |

This notebook explores the cascade mechanism. Reproducing published cascade-size
distributions additionally requires the experimental configurations, starter
identities, and threshold fitting procedure. See the
paper's [Behavioral Contagion Model](https://pmc.ncbi.nlm.nih.gov/articles/PMC6789631/)
and SI section 6.

The previous full implementation is preserved as a
[reference snapshot](../archive/README.md). P1 is unchanged.
