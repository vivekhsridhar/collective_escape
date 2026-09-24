# Model 1: Two-State Kinetic Ising Model for Collective Escape

## Objective

Implement the simplest stochastic model of collective escape in fish. Each fish is represented by a binary behavioral state, and positive social coupling allows an initially small threat response to spread through the group.

The model should capture three mechanisms:

1. External threat detection can activate escape in individual fish.
2. Escaping fish increase the probability that other fish escape.
3. Escaping fish eventually recover to the baseline state.

This is a population-level susceptibility and cascade model.

## Deliverable

Create the following files:
- An python file named `agent.py` that is an ising spin. Using an agent-based formulation will allow us to add complexity later.
- A python file named `stimulus.py` that is a simulated threat response. It is a square wave of specified amplitude and duration.
- A Jupyter notebook named `model-1.ipynb` with our model execution.

The notebook should be organized so that the model can be run incrementally and the experiments visualized as they are executed.

Use standard scientific Python libraries where appropriate, especially:

- `numpy` for simulation and numerical operations
- `matplotlib` for visualization
- `dataclasses` or equivalent lightweight structures for model parameters

Keep the model implementation separate from analysis and plotting functions within the notebook. Do not introduce a compiled extension or a separate programming language.

## Model Definition

### Fish states

For each fish $i=1,\ldots,N$, define a binary state:

$$
\sigma_i(t)\in\{-1,+1\}
$$

with:

- $\sigma_i=-1$: baseline state
- $\sigma_i=+1$: escape state

The population activity is:

$$
m(t)=\frac{1}{N}\sum_{i=1}^{N}\sigma_i(t).
$$

Also record the escape fraction:

$$
f_{\mathrm{escape}}(t)
=
\frac{1}{N}\sum_i\mathbf{1}\{\sigma_i(t)=+1\}.
$$

### Effective field

The effective field experienced by fish $i$ is:

$$
h_i(t)=s(t)+Jm(t)-\theta_i.
$$

Parameters:

- $s(t)$: external threat stimulus
- $J\geq0$: social coupling strength
- $\theta_i$: individual escape threshold
- $m(t)$: current population activity

For Model 1, use identical thresholds by default:

$$
\theta_i=\theta.
$$

The social term is mean-field: every fish responds to the current aggregate state of the group. This is intentional. Spatial and distance-dependent interactions will be added in a later model.

### Escape probability

For a fish currently in the baseline state, use a logistic transition probability:

$$
P_i(-1\rightarrow+1)
=
\frac{1}{1+\exp[-\beta h_i(t)]}.
$$

Here, $\beta$ is the inverse noise level:

- large $\beta$: sharp, nearly deterministic response
- small $\beta$: noisy response

Numerically clip the logistic argument if needed to avoid overflow.

### Escape duration and recovery

When a fish enters the escape state, assign it an escape timer. It remains in the escape state for a fixed duration $\tau_E$ measured in simulation time steps, then returns to baseline.

Use a fixed duration in Model 1 rather than a probabilistic recovery rule. This makes the pulse-like cascade dynamics easy to interpret and leaves refractory dynamics for a later extension.

Fish already in the escape state should not be re-sampled by the escape probability. They only decrement their timer and recover when the timer reaches zero.

## Stimulus and Initial Conditions

Support both of the following experiment modes.

### External stimulus pulse

Define a finite threat pulse:

$$
s(t)=
\begin{cases}
 s_0, & 0\leq t\leq\tau_s,\\
 0, & t>\tau_s.
\end{cases}
$$

Parameters:

- $s_0$: stimulus strength
- $\tau_s$: stimulus duration

### Initially escaping fish

Allow the simulation to begin with a specified number $n_0$ of fish already in the escape state. This separates social amplification from sensory detection.

If both an external stimulus and initially escaping fish are provided, support both effects simultaneously.

## Update Scheme

Use asynchronous updates to avoid imposing artificial synchrony on the group.

At each simulation time step:

1. Evaluate the external stimulus $s(t)$.
2. Compute the current population activity $m(t)$ from the state before new transitions.
3. Identify fish currently in the baseline state.
4. Compute each eligible fish's effective field and escape probability.
5. Sample baseline-to-escape transitions using the model random generator.
6. Set the escape timer for newly escaping fish.
7. Decrement timers for fish already escaping.
8. Return fish whose timers have elapsed to the baseline state.
9. Record the full state and population observables.

For the first implementation, an asynchronous step may update one randomly selected eligible fish at a time, with the number of update attempts per unit simulation time controlled explicitly. Alternatively, use a randomized sequential sweep in which each fish is visited once per time step. Choose one scheme, document it in the notebook, and use it consistently across experiments.

Use an explicit random-number generator with a user-controlled seed so that simulations are reproducible.

## Parameters

Expose the following parameters through a single model-configuration object:

- `population_size`: $N$
- `social_coupling`: $J$
- `inverse_noise`: $\beta$
- `threshold`: $\theta$
- `escape_duration`: $\tau_E$
- `time_step`
- `simulation_duration`
- `stimulus_strength`: $s_0$
- `stimulus_duration`: $\tau_s$
- `initial_escape_count`: $n_0$
- `random_seed`

Use dimensionless parameters initially. Avoid introducing biological units until the qualitative dynamics have been validated.

## Recorded Outputs

Each simulation should return a structured result containing at least:

- time points
- state of every fish at every recorded time point
- population activity $m(t)$
- escape fraction $f_{\mathrm{escape}}(t)$
- number of newly escaping fish per time step
- number of recovered fish per time step
- model parameters and random seed

The implementation should make it straightforward to run one trial or many repeated trials.

## Analysis Functions

Implement analysis functions for:

- maximum escape fraction
- time of maximum escape fraction
- escape latency
- cascade duration
- final escape fraction
- cascade probability across repeated trials
- trial-to-trial variability in latency and peak escape fraction
- false-alarm probability when no stimulus is present

Define a cascade initially as:

$$
\max_t f_{\mathrm{escape}}(t)\geq f_{\mathrm{cascade}},
$$

with a documented default such as $f_{\mathrm{cascade}}=0.5$.

Do not hard-code the cascade threshold inside the simulator; it belongs in the analysis layer.

## Notebook Visualizations

The notebook should include clear plots for individual trials:

1. A raster plot showing each fish's state over time.
2. The escape fraction $f_{\mathrm{escape}}(t)$.
3. The population activity $m(t)$.
4. The external stimulus $s(t)$.
5. Newly escaping and recovering fish per time step.

It should also include repeated-trial summaries:

- cascade probability versus $J$
- mean peak escape fraction versus $J$
- escape latency distributions
- stimulus-response curves versus $s_0$
- response curves versus initial escape count $n_0$
- false-alarm probability versus $J$ when $s(t)=0$

Use fixed random seeds in demonstration cells, but allow seed variation in parameter sweeps.

## Required Validation Experiments

### 1. No-social-interaction control

Set $J=0$.

Expected result:

- fish respond independently;
- there is no collective amplification beyond the imposed stimulus;
- increasing $N$ should not create a coordinated endogenous cascade.

### 2. Coupling sweep

Vary $J$ while holding all other parameters fixed.

Expected result:

- weak coupling produces mostly local or short-lived responses;
- intermediate coupling produces a sharp increase in cascade probability;
- strong coupling produces rapid, nearly population-wide responses and may generate false alarms when noise is present.

### 3. Noise sweep

Vary $\beta$.

Expected result:

- lower $\beta$ produces noisier and less reliable responses;
- higher $\beta$ produces more threshold-like responses;
- sufficiently high coupling combined with noise can produce spontaneous escapes.

### 4. Initial perturbation sweep

Vary $n_0$ with no external stimulus.

Expected result:

- below the amplification regime, small perturbations die out;
- near the transition, a few initially escaping fish can trigger a group cascade;
- increasing $n_0$ should reduce the latency of successful cascades.

### 5. Stimulus-strength sweep

Vary $s_0$.

Expected result:

- cascade probability increases with stimulus strength;
- the response is steepest near the transition between non-cascading and cascading behavior.

### 6. Population-size sweep

Vary $N$.

Expected result:

- larger groups should provide a smoother estimate of the mean-field activity;
- the transition may become sharper with increasing group size;
- results should be interpreted cautiously because this finite-size model is not expected to show a mathematical phase transition at small $N$.

## Acceptance Criteria

The implementation is complete when:

- the notebook runs from beginning to end in a clean kernel;
- fish have binary baseline and escape states;
- escape and recovery dynamics are stochastic and reproducible;
- social coupling is mediated by the current population activity;
- asynchronous updating is implemented and documented;
- setting $J=0$ removes social amplification;
- increasing $J$ increases collective amplification in repeated trials;
- single-trial and repeated-trial visualizations are generated;
- the required summary metrics are calculated;
- the notebook contains a short interpretation of the observed parameter sweeps;

The purpose of Model 1 is to establish whether a minimal kinetic Ising system can reproduce the basic threshold, amplification, and cascade phenomenology of collective escape before biological detail is added.
