# Phase 1 (complete) — Model 1: Implementation specification for collective escape

**Status: Phase 1 simulations are complete.** Retain this specification and the existing model as the baseline.

Implement a school of fish with binary behavioral states, stochastic escape initiation, positive social feedback, fixed escape durations, and a refractory period after recovery. Build `simulation.ipynb` using the mathematical rules, data structures, and execution order specified below.

## 1. Organize the implementation

Use Python, NumPy, and Matplotlib. Place three files in the same directory:

- `agent.py`: define an ordinary `Agent` class with an `__init__` constructor. Store each agent's behavioral state in `state`, its remaining escape time in `recovery_timer`, and its remaining refractory time in `refractory_timer`.
- `stimulus.py`: define `square_wave(time, amplitude, duration)`, returning the amplitude where time is greater than or equal to zero and strictly less than the duration, and zero elsewhere.
- `simulation.ipynb`: import these components, define parameters, initialize the school, allocate histories, run the simulation, and plot the states.

In the agent constructor, randomly assign `state` to 0 or 1 using NumPy's global random generator and initialize both timers to zero. In the notebook, initialize all agents to the baseline state (0). Store and update both timers on the agents themselves.

## 2. Define states and parameters

For fish $i=1,\ldots,N$, let

```math
x_i\in\{0,1\},
```

where 0 means baseline and 1 means escaping. Store $x_i$ in `Agent.state`. Let $r_i$ be the nonnegative integer stored in `Agent.recovery_timer` and $q_i$ the nonnegative integer stored in `Agent.refractory_timer`. Distinguish three conditions while retaining a binary behavioral state:

- Escaping: $x_i=1$, $r_i>0$, and $q_i=0$.
- Refractory: $x_i=0$, $r_i=0$, and $q_i>0$.
- Eligible baseline: $x_i=0$, $r_i=0$, and $q_i=0$.

Maintain these invariants throughout the initialized simulation. Refractory fish cannot initiate an escape, even when directly exposed, and contribute zero social input.

Define the following parameters in a notebook cell, using these default values:

| Variable | Symbol | Value | Meaning |
| --- | --- | --- | --- |
| `n_fish` | $N$ | 20 | Number of fish |
| `total_timesteps` | $L$ | 200 | Number of updates and recorded columns |
| `random_seed` | — | 2026 | Simulation random seed |
| `J` | $J$ | 1.0 | Social coupling strength |
| `T` | $T$ | 0.1 | Noise temperature |
| `threshold` | $\theta$ | 0.5 | Common activation threshold |
| `escape_duration` | $d_E$ | 5 | Number of steps spent escaping |
| `refractory_duration` | $d_R$ | 10 | Full steps blocked after recovery |
| `stimulus_start` | $t_{\mathrm{start}}$ | 10 | First stimulus step |
| `stimulus_strength` | $s_0$ | 1 | Pulse amplitude |
| `n_exposed` | $n$ | 5 | Number of fish directly exposed to the pulse |
| `stimulus_duration` | $d_s$ | 5 | Pulse duration |

Treat all parameters as dimensionless and express durations in integer update steps. Use positive integers for population size, simulation length, and escape duration. Use a nonnegative integer for `refractory_duration`; zero disables the additional refractory period. Use a positive temperature $T$ and reserve $L$ for the simulation length.

## 3. Initialize the simulation

Create one NumPy random generator using `default_rng` with `random_seed`. Use this generator for every update permutation and activation draw. Construct an array of $N$ agents, then set every agent's `state` to zero. Initialize their recovery and refractory timers to zero through the constructor, so every fish starts eligible for activation.

Choose an integer `n_exposed` between 0 and $N$, inclusive. Using the simulation generator, select exactly `n_exposed` distinct fish uniformly without replacement. Store this selection in a Boolean array named `exposed` and keep it fixed throughout the run. Let $e_i=1$ for selected fish and $e_i=0$ for all others. Use this same generator for subsequent update permutations and activation draws. A value of 0 gives a no-direct-exposure control; a value of $N$ exposes the whole school.

Create the time array containing $t=0,\ldots,L-1$. Evaluate the square-wave function on time shifted by `stimulus_start`, giving

```math
s_t=
\begin{cases}
 s_0,&t_{\mathrm{start}}\leq t\lt t_{\mathrm{start}}+d_s,\\
 0,&\text{otherwise}.
\end{cases}
```

With the supplied values, the stimulus is 1 at steps 10 through 14 and zero elsewhere. Apply the pulse only to selected fish: fish $i$ receives direct input $e_i s_t$. Exposure changes activation probability; it does not force a fish to escape. Unexposed fish receive social input and retain baseline stochastic activation.

Allocate `state_history` as an $N\times L$ array of 8-bit integers. Rows represent fish and columns represent update steps. Reserve each column for the outcome of its indexed update, starting with update 0. Use the post-update recording convention described below.

## 4. Compute activity, field, and activation probability

The activity is the fraction of fish escaping:

```math
a=\frac{1}{N}\sum_{i=1}^{N}x_i.
```

For an eligible baseline fish at its update turn, compute the effective field

```math
h_i=e_i s_t+Ja_i-\theta,
```

where $a_i$ is the activity immediately before that fish's turn. The fish share the same activation parameters, but direct exposure differs between fish. Their fields also change within a step as earlier fish activate.

Use 0/1 activity for the social term. Each escaping fish contributes $J/N$ to the field, while each baseline fish contributes zero. Thus, even one escaping fish provides positive social input when $J>0$; a majority is not required.

Activate an eligible baseline fish with probability

```math
p_i=P(x_i:0\rightarrow1\mid h_i)
=\frac{1}{1+\exp(-h_i/T)}.
```

Draw one independent uniform number $u_i\in[0,1)$ from the simulation generator. If $u_i<p_i$, set the fish's state to 1 and its recovery timer to $d_E$. Otherwise leave it at baseline with recovery timer zero. Its refractory timer remains zero in either case. This is a probability per visit, with no extra time-step multiplier. Do not make activation draws for escaping or refractory fish.

For positive $T$, positive fields favor activation and negative fields suppress it. At zero field, the probability is $1/2$. Smaller $T$ makes the response sharper; as $T\rightarrow0^+$, it approaches a threshold at $h_i=0$. As $T\rightarrow\infty$, activation approaches probability $1/2$ regardless of the field. In inverse-temperature notation, $\beta=1/T$; the default $T=0.1$ corresponds to $\beta=10$. Use $T$ directly in the implementation.

Finite noise allows spontaneous escapes. With no stimulus and no escaping neighbors, the supplied parameters give

```math
p_0=\frac{1}{1+\exp(\theta/T)}
=\frac{1}{1+\exp(5)}\approx0.00669
```

per eligible baseline fish per step, or approximately 0.669%. Evaluate the exponential directly in the activation formula.

### Relation to Glauber dynamics

Glauber dynamics uses stochastic single-site updates. In its heat-bath form, a site's state is selected according to its local conditional probability. For a binary state with local energy difference

```math
\Delta E_i=E_i(1)-E_i(0)=-h_i,
```

the probability of selecting the active state is

```math
P(x_i=1\mid\text{other states})
=\frac{1}{1+\exp(\Delta E_i/T)}
=\frac{1}{1+\exp(-h_i/T)}.
```

Use this logistic form for baseline-to-escape activation. Use the binary-state convention above, with exponent $-h_i/T$ and no factor of two.

Implement **Glauber-like activation with fixed escape and refractory durations**. Standard two-way heat-bath updates can select either state when a site is visited. For this model, apply stochastic activation only to eligible baseline fish, recover escaping fish when their escape timers expire, and block further activation until their refractory timers expire. Use the local energy interpretation to understand the activation formula; a global energy calculation is not required. The timer-based process should not be interpreted as equilibrium dynamics satisfying detailed balance.

## 5. Execute each update step in the specified order

At the start of step $t$, save a Boolean mask named `already_escaping` that identifies agents whose state is 1, and a second mask named `already_refractory` that identifies agents with positive refractory timers. Keep both masks unchanged throughout the step. Exclude both groups when determining eligibility, even if a timer will expire during this step.

1. Count the fish in the saved `already_escaping` mask to initialize `active_count`.
2. Obtain the indices of eligible baseline fish in ascending order, excluding both saved masks, then use the simulation generator's random permutation to determine their visitation order. Visit each exactly once.
3. Before each visit, divide the current `active_count` by $N$. Multiply the current stimulus by the visited fish's exposure indicator, then use that direct input and activity to compute the field and logistic probability.
4. Draw one uniform number. If the fish activates, change its `state` to 1, assign $d_E$ to its `recovery_timer`, and immediately increment `active_count`. Later fish see this additional social input.
5. After all activation attempts, visit the fish identified by the saved `already_refractory` mask and decrease each one's `refractory_timer` by one. A fish whose timer reaches zero becomes eligible on the next step.
6. Separately visit the fish identified by the saved `already_escaping` mask. Decrease each one's `recovery_timer` by one. If it reaches zero, set that fish's `state` to 0 and assign $d_R$ to its `refractory_timer`.
7. Copy all agent states into history column $t$.

Repeat for every step from 0 through $L-1$. Generate one permutation per step and one uniform draw per eligible fish, in visitation order, so that random-number consumption is reproducible.

Perform activation as a random sequential sweep within each discrete time step, then advance the saved groups' timers. Include fish due to recover in the social field throughout the activation sweep. Leave newly escaping fish's escape timers unchanged until the following step. Likewise, do not decrement a newly assigned refractory timer in the step of recovery. Use the saved masks to enforce both rules.

### Mathematical form of the sweep

Let $x_i^{(t)}$, $r_i^{(t)}$, and $q_i^{(t)}$ be the state, escape timer, and refractory timer before update $t$. Define the escaping, refractory, and eligible baseline sets:

```math
E_t=\{i:x_i^{(t)}=1\},\qquad
R_t=\{i:q_i^{(t)}\gt0\},\qquad
B_t=\{i:x_i^{(t)}=0,\ q_i^{(t)}=0\}.
```

These sets partition the school under the state and timer invariants above. For a random permutation $\pi_1,\ldots,\pi_{|B_t|}$ of the eligible baseline fish, start with $A_0=|E_t|$. At visit $k$, compute

```math
a_k=\frac{A_{k-1}}{N},\qquad
h_k=e_{\pi_k}s_t+Ja_k-\theta,\qquad
p_k=\frac{1}{1+\exp(-h_k/T)},
```

```math
z_k=\mathbf{1}[u_k\lt p_k],\qquad A_k=A_{k-1}+z_k.
```

Here $\mathbf{1}[C]$ is 1 when condition $C$ is true and 0 otherwise. The complete state and timer updates are

```math
x_i^{(t+1)}=
\begin{cases}
 z_k,&i=\pi_k\in B_t,\\
 \mathbf{1}[r_i^{(t)}\gt 1],&i\in E_t,\\
 0,&i\in R_t.
\end{cases}
```

```math
r_i^{(t+1)}=
\begin{cases}
 d_Ez_k,&i=\pi_k\in B_t,\\
 r_i^{(t)}-1,&i\in E_t,\\
 0,&i\in R_t.
\end{cases}
```

```math
q_i^{(t+1)}=
\begin{cases}
 q_i^{(t)}-1,&i\in R_t,\\
 d_R,&i\in E_t\text{ and }r_i^{(t)}=1,\\
 0,&\text{otherwise}.
\end{cases}
```

### Escape and refractory timing example

With the default $d_E=5$ and $d_R=10$, a fish activated during step $t$ appears as escaping in recorded columns $t$ through $t+4$. It recovers at the end of step $t+5$, after contributing to that step's social field, and receives a refractory timer of 10. Block activation during the following 10 full steps, $t+6$ through $t+15$. Its timer reaches zero at the end of step $t+15$, so its next eligible activation step is $t+16$.

In general, the next eligible step is $t+d_E+d_R+1$. Setting $d_R=0$ makes a recovered fish eligible on the step immediately after recovery, restoring the behavior without an additional refractory period. Use deterministic timer expiration for both phases. Further stimulus exposure must not extend either timer.

## 6. Record and visualize the result

Let $H_{i,t}$ denote the entry at row $i$, column $t$ of `state_history`. After each complete update, including recovery, record

```math
H_{i,t}=x_i^{(t+1)}.
```

Store the outcome of update 0 in column 0 and retain exactly $L$ post-update columns. Retain the time, stimulus, and exposure arrays and leave the final states and both timers on the agents after the run.

Create a two-panel Matplotlib figure with a shared time axis and constrained layout. In the upper panel, draw `state_history` using `pcolormesh`, with the time array on the horizontal axis and fish indices on the vertical axis. Use a two-color map: blue (`#2166ac`) for baseline and red (`#b2182b`) for escape. Label every fish index and append an asterisk to directly exposed fish. Label the vertical axis “Fish (* = directly exposed),” and include the population size and exposed count in the title. Add a colorbar labeled “State,” with ticks at 0 and 1 labeled “Baseline (0)” and “Escape (1).” Remember that state 0 includes both eligible and refractory fish; this binary raster does not distinguish them.

For interpretation, define the escape-fraction trajectory as

```math
f_{\mathrm{escape}}[t]
=\frac{1}{N}\sum_iH_{i,t}.
```

This equals activity at the end of each step. Plot it as a step curve in the lower panel, with the vertical axis covering 0 to 1 and labeled “Fraction escaping.” Align each raster column and curve segment with the interval from its update index to the next. Mark stimulus onset with a dashed black vertical line and shade the stimulus interval in both panels and label the shared horizontal axis “Time (simulation steps).” Display the figure.

## 7. Run the notebook

Execute the cells in order to reset the random generator, agents, timers, and history before each fresh trial. Repeating the full run with the same seed in the same software environment should reproduce the trajectory. Rerunning only the simulation cell continues from the existing states and random generator while overwriting the history.

---

# Phase 2: Pulse-coupled evidence integration

**Status: specification and planning only; implementation has not started.** Phase 1 simulations are complete. The instructions below define a new, alternative model; they do not supersede the Phase 1 baseline. Implementation and the listed deliverables are future work.

## Objective

Extend the existing minimal Ising-like collective-escape model by adding an alternative, event-driven model in which fish integrate brief social escape cues over time.

The biological question is:

> Does a fish escape in response to one sufficiently strong neighbour cue, or does it temporally integrate multiple escape cues before crossing an escape threshold?

Do not replace or alter the existing Ising model. Preserve it as a baseline and make the new model selectable through the existing configuration or command-line interface.

## Biological interpretation

Each fish occupies a fixed home coral. Escape direction does not need to be modelled: once a fish responds, it enters its coral and remains in an absorbing sheltered state for the rest of that trial.

Fish can receive two kinds of evidence:

1. **Direct evidence:** visual exposure to the looming stimulus.
2. **Social evidence:** brief visual events generated when visible neighbours escape into their corals.

A neighbour should generate a transient escape cue at its escape time. It must not exert a persistent social field merely because it remains in the escaped state.

## Model

For every unescaped fish $i$, maintain a continuous decision variable $x_i(t)$:

```math
\tau\frac{dx_i}{dt}
=
-x_i
+L_i(t)
+\alpha S_i(t)
+\eta_i(t).
```

Fish $i$ escapes when

```math
x_i(t)\geq\theta_i.
```

After escaping:

- record its escape time;
- change its state to sheltered;
- generate one social escape event for its neighbours;
- stop updating its decision variable.

### Direct evidence

Represent direct loom evidence as $L_i(t)$.

The implementation must support different loom exposure histories for different fish:

- no direct exposure;
- weak or subthreshold exposure;
- strong exposure;
- different exposure onset times.

If the existing model already calculates loom visibility or an external field, reuse that implementation where possible. Otherwise, define a clean interface that accepts either a function or an agent-by-time array for $L_i(t)$.

Do not introduce escape direction.

### Social evidence

When neighbour $j$ escapes at time $t_j$, fish $i$ receives a delayed transient cue:

```math
C_{ij}(t)
=
V_{ij}\,K(d_{ij})\,q(t-t_j-\delta).
```

Here:

- $V_{ij}$ is the visibility or adjacency between fish $i$ and $j$;
- $K(d_{ij})$ is the distance-dependent interaction strength;
- $\delta$ is the social sensory–motor delay;
- $q(\cdot)$ is a brief temporal pulse.

Use an exponential pulse by default:

```math
q(u)=
\begin{cases}
0, & u<0,\\
\exp(-u/\tau_{\mathrm{cue}}), & u\geq0.
\end{cases}
```

Make the pulse kernel modular so that a square pulse or another kernel could be substituted later.

## Social-integration modes

Implement the following three modes behind a common interface.

### 1. Single-neighbour mode

Only the strongest currently available neighbour cue contributes:

```math
S_i(t)=\max_j C_{ij}(t).
```

This represents a system in which one sufficiently strong neighbour can trigger an escape and simultaneous cues are not summed.

### 2. Accumulation mode

All currently available neighbour cues are summed:

```math
S_i(t)=\sum_j C_{ij}(t).
```

Because $x_i(t)$ leaks with timescale $\tau$, cues arriving close together can jointly cause threshold crossing, whereas widely separated cues may not.

This should be the default new model.

### 3. Loom-gated accumulation mode

Maintain a separate direct-evidence variable:

```math
\tau_L\frac{dx_i^{L}}{dt}
=
-x_i^{L}+L_i(t).
```

Use this to modulate the effectiveness of social evidence:

```math
S_i^{\mathrm{effective}}(t)
=
G(x_i^{L})\sum_j C_{ij}(t).
```

Support either:

- a hard gate, $G=1$ when $x_i^{L}\geq\phi$ and $0$ otherwise; or
- a sigmoid gate centred on $\phi$.

This mode tests whether social cues primarily recruit fish already primed by weak direct evidence.

## Numerical implementation

Use simultaneous updates for all agents to avoid order-dependent cascades.

For each timestep:

1. Calculate direct input for every unescaped fish.
2. Calculate social input from escape events that have completed their delay.
3. Update all decision variables using Euler–Maruyama:

   ```math
   x_i(t+\Delta t)
   =
   x_i(t)
   +
   \Delta t
   \left[
   -\frac{x_i(t)}{\tau}
   +L_i(t)
   +\alpha S_i(t)
   \right]
   +
   \sigma\sqrt{\Delta t}\,\epsilon_i,
   ```

   where $\epsilon_i\sim\mathcal N(0,1)$.

4. Identify all threshold crossings simultaneously.
5. Record those escapes and add their social events to the event queue.
6. Newly escaped fish may affect others only on subsequent timesteps and after the specified delay.

All randomness must use the project’s seeded random-number generator so simulations are reproducible.

## Parameters

Expose the following parameters through the existing configuration system:

- `dt`
- `tau_evidence`
- `tau_cue`
- `social_delay`
- `social_strength`
- `threshold`
- `threshold_variation`
- `noise_sigma`
- `distance_kernel`
- `distance_scale`
- `integration_mode`
- `gate_type`
- `gate_threshold`
- `gate_steepness`

Allow thresholds to be identical initially, with optional agent-level heterogeneity.

Do not fit these parameters to data in this implementation. They should be available for systematic simulation sweeps.

## Interaction network

Reuse the existing spatial arrangement and interaction network.

If positions are available, calculate

```math
W_{ij}=V_{ij}K(d_{ij}).
```

Keep the distance kernel modular. Include at least:

- constant coupling within a cutoff;
- exponential decay;
- Gaussian decay.

The interaction network can remain fixed within a trial. Do not add movement, collision avoidance or directional dynamics.

## Outputs

Retain all existing outputs and add an event-level table containing:

- trial ID;
- agent ID;
- position or coral ID;
- direct-exposure category;
- direct loom input at escape;
- accumulated direct evidence at escape;
- social input at escape;
- number of neighbours that escaped previously;
- number of social cues received within configurable windows, such as 25, 50, 100 and 200 ms;
- escape time;
- escape rank;
- threshold;
- integration mode.

Also calculate trial-level summaries:

- total cascade size;
- fraction of the group escaping;
- number of directly exposed fish;
- number of socially recruited fish;
- time of first escape;
- cascade duration;
- inter-escape intervals;
- temporal dispersion of the initial escapes;
- peak number of escapes within a moving time window.

Do not assign a definitive causal label such as “socially caused” merely because social input was present. Record the direct and social contributions at threshold crossing so they can be analysed later.

## Required simulation experiments

Add a script or notebook that runs the following controlled comparisons.

### Experiment 1: Temporal summation

Hold group geometry and the total number of initially escaping fish constant. Vary only the temporal dispersion of their escape times.

Test whether tightly synchronized initial escapes generate larger cascades than the same number of temporally dispersed escapes.

### Experiment 2: Number of early responders

Vary the number of directly exposed fish while holding total group size and geometry constant.

Compare cascade probability and cascade size among the three integration modes.

### Experiment 3: Weak direct priming

Compare otherwise identical focal fish receiving:

- no loom evidence;
- weak subthreshold loom evidence;
- strong loom evidence.

Test how each responds to the same sequence of social escape cues, particularly under loom-gated integration.

### Experiment 4: Distance and cue number

Compare:

- one nearby escaping neighbour;
- one distant escaping neighbour;
- several individually subthreshold neighbours escaping close together in time.

This should demonstrate whether multiple weak social cues can jointly trigger escape.

## Visualizations

Produce at least:

1. an escape raster showing agent escape times;
2. decision-variable trajectories $x_i(t)$ for representative fish;
3. cascade size versus temporal dispersion of initial escapes;
4. escape probability versus number of recent neighbour escapes;
5. cascade size across the three integration modes;
6. an example spatial network coloured by escape time.

Use the existing plotting style where one exists.

## Tests and acceptance criteria

Add automated tests confirming that:

- the original Ising model produces unchanged output under a fixed seed;
- setting `social_strength = 0` eliminates social recruitment;
- no social cue arrives before `social_delay`;
- each fish emits exactly one social escape event;
- escaped fish cannot escape again;
- updates are independent of agent iteration order;
- two subthreshold pulses arriving within the integration window can jointly cross threshold in accumulation mode;
- the same pulses separated by several integration times do not cross threshold;
- single-neighbour mode uses the maximum cue rather than their sum;
- a hard loom gate blocks social recruitment in an unprimed fish;
- all stochastic simulations are reproducible from a seed.

## Scope restrictions

Do not add:

- explicit neural populations;
- Mauthner-cell circuitry;
- directional escape decisions;
- swimming or collision dynamics;
- biological-versus-linear-motion classification;
- parameter fitting or Bayesian inference.

The purpose of this extension is specifically to test how the timing, strength and integration of transient social escape cues determine whether a local response becomes a collective cascade.

## Deliverables

After implementation:

1. summarize the existing architecture you found;
2. list every file changed or added;
3. explain any assumptions required;
4. show how to run the original and extended models;
5. run the automated tests;
6. run a small demonstration of all three integration modes;
7. report whether synchronized initial escapes generate different cascade outcomes from temporally dispersed escapes.

## Pre-implementation notes

These notes identify decisions to resolve during Phase 2 design. They do not change the supplied requirements above.

- **Existing architecture:** Parameters and simulation loops currently live in `simulation.ipynb` and `phase_diagram.ipynb`, with `agent.py` and `stimulus.py` providing small shared components. There is no existing command-line interface or standalone configuration system. Plan model selection through notebook configuration, retaining the current Ising baseline and its random-number consumption.
- **Spatial inputs:** The baseline uses global mean-field coupling and has no stored positions, coral layout, visibility matrix, or explicit interaction network. Phase 2 therefore needs a fixed-position/network input interface for its distance experiments. Document any example geometry and coupling normalization as new assumptions.
- **Equation convention:** The continuous equation places the entire direct and social input under the evidence timescale after division by `tau_evidence`; the supplied Euler–Maruyama update divides only the leak term by that timescale. Resolve which convention defines the new model, including evidence and noise units, before coding or choosing numerical parameters.
- **Additional configuration:** The gated model requires a direct-evidence timescale (the specified tau_L), which is absent from the parameter list. Also define time units, cue-count windows, the trial horizon, and a controlled schedule of initial escape events for the timing experiments. Record accumulated direct evidence in all modes so their outputs are comparable.
- **Timing and outcomes:** Define cue arrival at non-grid-aligned delays, simultaneous escape ranks, the initial-responder cohort, cascade criteria, and no-escape summary values. Count received cues by their delayed arrival times. Define a cautious operational recruitment measure and use matched no-social controls when assessing social effects; social input at threshold crossing alone is not a causal label.
