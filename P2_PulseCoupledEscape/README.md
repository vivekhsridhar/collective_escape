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
- [network.py](network.py): `random_network` generates a connected, undirected
  binary network; `is_connected` checks whether all fish are reachable.
  `random_directed_network` provides a directed network with a specified mean
  in-degree and a directed path between every pair of fish.
- [dose.py](dose.py): `draw_doses` samples packets from active neighbours;
  `accumulate_doses` sums the packets in a finite memory window.

Run [simulation.ipynb](simulation.ipynb) from the top, using
`P2_PulseCoupledEscape/` as the working directory and the `pymc_env` kernel.
Group size, mean in-degree, seeds, initialization, the update loop,
and plots remain visible in the notebook. No input data file is required.
[dose_memory.ipynb](dose_memory.ipynb) explains dose arrivals and finite memory
using one sender and one receiver.

[network_frame.ipynb](network_frame.ipynb) retains the separate recorded-frame
exploration. The current simulation constructs its connected random network directly.

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
the shuffled cycle and extra directed links independently.

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
`mean_threshold = 0.10` is an exploratory setting for the connected binary
random network, chosen to delay responses while allowing cascades. Thresholds
are drawn from zero to `0.20`; the distribution still permits values near zero.
This setting is not fitted to experimental cascade sizes. Sosna's baseline
reference was `0.034` (SI section 6.3, page 9), fitted to empirical networks
with probability weights.

## Directed networks with a specified mean in-degree

`random_directed_network(n_fish, mean_in_degree, random_seed=None)` is an
the generator used by the simulation, defined in [network.py](network.py). Rows receive from columns,
so `adjacency.sum(axis=1)` gives each fish's in-degree. Connections are binary;
self-connections and duplicate links are excluded.

The generator shuffles the fish into a directed cycle, giving every fish an
incoming and an outgoing link and making the whole network strongly connected.
It then selects extra directed links uniformly from the remaining possibilities.
This construction can produce different in-degrees across fish while ensuring
that no fish has zero in-degree.

The total number of links is `round(n_fish * mean_in_degree)`, so the realized
mean is that number divided by `n_fish`. Rounding is necessary when the requested
average would require a fractional link. Use `n_fish >= 2` and
`1 <= mean_in_degree <= n_fish - 1`. At the lower limit the graph is a directed
cycle; at the upper limit every fish receives from all other fish. The same seed
reproduces the same network. This cycle-based construction is not a uniform
sample of all strongly connected graphs with the given number of links.

The notebook uses `mean_in_degree = 3`; individual in-degrees can vary.
The undirected `random_network` generator remains available in `network.py`.

## Receive and remember social doses

The notebook uses
`random_directed_network(n_fish, mean_in_degree, network_seed)`.
Every fish has at least one incoming neighbour, and all fish are mutually
reachable along directed paths. Thresholds still determine whether a cascade
propagates along those paths.

`adjacency[i, j]` is the weight from sender `j` to receiver `i`.
The network stays fixed during the run. Rows are not normalized.

Only susceptible fish receive doses. The receiver mask and active sender mask
use the states from the preceding update. At each update, an active neighbour
sends a packet with probability
`dose_rate * adjacency[i, j] * dt`. This probability must be at most one.
Each arriving packet contributes `dose_size / neighbour_count[i]`, where the
count includes all positive incoming connections, including inactive neighbours.
An isolated fish receives no doses.

With the current binary weights and dose settings, each active connection
supplies a packet every update. Fractional weights would make arrivals stochastic.
The weight changes arrival probability, not packet size.

`dose_history[:, step]` stores the sum of normalized packets at that update.
`accumulate_doses` sums the most recent `memory_steps` columns, including the
current one. Evidence is this remembered sum, compared directly with the
escape threshold without an additional multiplier.
The sum is neither added to previous evidence nor multiplied by `dt` again.

A packet received at update `k` keeps its full contribution through
`k + memory_steps - 1` and leaves the sum at `k + memory_steps`.
`memory_steps = round(memory_duration / dt)` converts the memory
window from seconds to whole updates. This memory has no exponential leak.

## Active phases and timing

Each fish can be active only once, so `n_fish * active_steps` bounds the
number of updates before the cascade ends. Arrays initially reserve
`max_steps = n_fish * active_steps + 1` columns, including the initial state.
This capacity does not impose a shorter observation window.

After every complete update, the simulation stops if no fish remains active.
The check includes fish that activated during that update. Histories are then
trimmed to `n_steps = step + 1`, retaining the final inactive state.
`time = np.arange(n_steps) * dt` runs from sample zero through `n_steps - 1`.
Column zero stores the initial state; each later column stores one update.

Dose arrivals use the active-state snapshot from before the update. A fish
newly activated at update `k` first sends on update `k + 1`. An already-active
fish sends before its timer decreases, so fish update order cannot create
several links of propagation within one timestep.

`active_steps = round(active_duration / dt)` sets the active-phase
length. With `m = active_steps`, a fish activated at index `k` is recorded
active at `k` through `k + m - 1`. It sends during updates `k + 1` through
`k + m`, then enters shelter at index `k + m`. This also applies to the
initial starter with `k = 0`.

Only susceptible fish update their evidence. The dose that triggers activation
is included, and the evidence value at activation is retained for plotting
through the active and sheltered states. The initial starter retains zero
evidence because its activation is forced. Active fish send doses but receive
none; sheltered fish neither send nor receive doses. Entering shelter does not
clear doses already held in susceptible neighbours' memories.

The final column has no active fish: each fish is either still unsheltered
or has escaped and entered shelter. The plots end at this stopping time.
The plots show evidence, escape onsets, and the three behavioral states.
There is no common threshold line; `thresholds` stores the fish-specific values.
Each fish has at most one onset mark, including the initial starter at time zero.

## Reproducing Sosna et al. (2019)

The model uses the paper's dose scale and rate, neighbour normalization,
finite social memory, active duration, absorbing terminal state, and stopping
condition of no remaining active fish. The
notebook's single forced initial startle also follows its cascade setup.
The following differences remain:

| Component | Current notebook | Sosna model |
| --- | --- | --- |
| Starter | Uniform random fish or a chosen index | Observed initial startler for each experimental cascade |
| Network | Strongly connected directed binary random network with specified mean in-degree | Empirical directed response weights from experimental configurations |
| Mean threshold | Exploratory mean `0.10` for the connected binary random network | Mean threshold fitted to experimental cascade sizes, with the same uniform distribution |
| Comparison | One realization and its time courses | Repeated simulations fitted to experimental cascade sizes |

This notebook explores the cascade mechanism. Reproducing published cascade-size
distributions additionally requires the experimental configurations, starter
identities, and threshold fitting procedure. See the
paper's [Behavioral Contagion Model](https://pmc.ncbi.nlm.nih.gov/articles/PMC6789631/)
and SI section 6.

The previous full implementation is preserved as a
[reference snapshot](../archive/README.md). P1 is unchanged.
