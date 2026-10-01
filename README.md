This code is intended to simulate collective escape of predatory threat by damselfish. These are obligate coral-dwelling fish and are prey to most species. Their primary escape strategy is to use spaces within the coral branching as refuge. They respond to conspecifics that share their coral head and this code is meant to model the escape cascades that are commonly exhibited by these species. Individuals respond to both personal information about threat but also to social information obtained from the escape of coral-mates.

# Phase 1 (complete) — Model 1: Implementation of a simple collective escape model

Implement a school of fish with binary behavioral states, stochastic escape initiation, positive social feedback, fixed escape durations, and a refractory period after recovery.

## 1. Organize the implementation

The Phase 1 code lives in [`P1_BasicIsingEscape/`](P1_BasicIsingEscape/):

- [`agent.py`](P1_BasicIsingEscape/agent.py): define an ordinary `Agent` class with an `__init__` constructor. Store each agent's behavioral state in `state`, its remaining escape time in `escape_timer`, and its remaining refractory time in `refractory_timer`.
- [`stimulus.py`](P1_BasicIsingEscape/stimulus.py): define a `square_wave` function that returns a stimulus of specified amplitude for a fixed duration, and zero amplitude outside this timeframe.
- [`simulation.ipynb`](P1_BasicIsingEscape/simulation.ipynb): import these components, define parameters, initialize the school, allocate histories, run the simulation, and plot the states.
- [`phase_diagram.ipynb`](P1_BasicIsingEscape/phase_diagram.ipynb): run parameter sweeps and plot phase diagrams.

Open either notebook and run its cells from top to bottom, using `P1_BasicIsingEscape/` as the notebook working directory so that its local imports resolve.

## 2. Define states and parameters

For fish $i=1,\ldots,N$, let

```math
x_i\in\{0,1\},
```

where 0 means baseline and 1 means escaping. Store $x_i$ in `Agent.state`. Let $r_i$ be the nonnegative integer stored in `Agent.escape_timer` and $q_i$ the nonnegative integer stored in `Agent.refractory_timer`. The agent can then exhibit one of these behaviors:

- Baseline: $x_i=0$, $r_i=0$, and $q_i=0$.
- Escaping: $x_i=1$, $r_i>0$, and $q_i=0$.
- Refractory: $x_i=0$, $r_i=0$, and $q_i>0$.

Refractory fish are ones inside the coral head and cannot initiate an escape. They contribute zero social input.

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

## 3. Initialize the simulation

Construct an array of $N$ agents, then set every agent's `state` to zero. Initialize their recovery and escape timers to zero so every fish is in the baseline state, outside the coral refuge and eligible for an escape.

Randomly select `n_exposed` distinct fish that experience the external predatory stimulus. Let $e_i=1$ for selected fish and $e_i=0$ for all others. An `n_exposed` value of 0 gives a no-direct-exposure control; a value of $N$ exposes the whole school.

Evaluate the square-wave stimulus on a time array containing $t=0,\ldots,L-1$

```math
s_t=
\begin{cases}
 s_0,&t_{\mathrm{start}}\leq t\lt t_{\mathrm{start}}+d_s,\\
 0,&\text{otherwise}.
\end{cases}
```

With the tabulated values, the stimulus is 1 at steps 10 through 14 and zero elsewhere. Apply the pulse only to the selected fish: fish $i$ receives direct input $e_i s_t$. Exposure changes activation probability; it does not force a fish to escape. Unexposed fish receive social input and retain baseline stochastic activation.

Allocate `state_history` as an $N\times L$ array of 8-bit integers. Rows represent fish and columns represent update steps.

## 4. Compute activity, field, and activation probability

At each timestep, update all fish in a randomly determined order. The activity is the fraction of fish escaping:

```math
a=\frac{1}{N}\sum_{i=1}^{N}x_i.
```

For a baseline fish at its update, compute the effective field

```math
h_i=e_i s_t+Ja_i-\theta,
```

where $a_i$ is the activity immediately before that fish's update. The fish share the same activation parameters, but direct exposure differs between fish. Their fields also change within a step as earlier fish activate.

Use 0/1 activity for the social term. Each escaping fish contributes $J/N$ to the field, while each baseline fish contributes zero. Thus, even one escaping fish provides positive social input when $J>0$; a majority is not required.

Activate an eligible baseline fish with probability

```math
p_i=P(x_i:0\rightarrow1\mid h_i)
=\frac{1}{1+\exp(-h_i/T)}.
```

If activated, set the fish's state to 1 and its recovery timer to $d_E$. Otherwise leave it at baseline with escape timer zero. Its refractory timer remains zero in either case. Do not make activation draws for escaping or refractory fish.

For positive $T$, positive fields favor activation and negative fields suppress it. At zero field, the probability is $1/2$. Smaller $T$ makes the response sharper; as $T\rightarrow0^+$, it approaches a threshold at $h_i=0$. As $T\rightarrow\infty$, activation approaches probability $1/2$ regardless of the field.

Finite noise allows spontaneous escapes. With no stimulus and no escaping neighbors, the supplied parameters give

```math
p_0=\frac{1}{1+\exp(\theta/T)}
=\frac{1}{1+\exp(5)}\approx0.00669
```

per eligible baseline fish per step, or approximately 0.669%. Evaluate the exponential directly in the activation formula.

## 5. Execute each update step in the specified order

At the start of step $t$, save a Boolean mask named `already_escaping` that identifies agents whose state is 1, and a second mask named `already_refractory` that identifies agents with positive refractory timers. Exclude both groups when determining eligibility for escape.

1. Count the fish in the saved `already_escaping` mask to initialize `active_count`.
2. Obtain the indices of eligible baseline fish in ascending order, excluding both saved masks, then use random permutation to determine their visitation order.
3. Before each visit, divide the current `active_count` by $N$. Multiply the current stimulus by the visited fish's exposure indicator, then use that direct input and activity to compute the field and logistic probability.
4. Draw one uniform number. If the fish activates, change its `state` to 1, assign $d_E$ to its `escape_timer`, and immediately increment `active_count`. Later fish see this additional social input.
5. After all activation attempts, visit the fish identified by the saved `already_refractory` mask and decrease each one's `refractory_timer` by one. A fish whose timer reaches zero becomes eligible on the next step.
6. Separately visit the fish identified by the saved `already_escaping` mask. Decrease each one's `escape_timer` by one. If it reaches zero, set that fish's `state` to 0 and assign $d_R$ to its `refractory_timer`.
7. Copy all agent states into history column $t$.

Repeat for every step from 0 through $L-1$.

Perform activation as a random sequential sweep within each discrete time step, then advance the saved groups' timers.

### Mathematical form of the sweep

Let $x_i^{(t)}$, $r_i^{(t)}$, and $q_i^{(t)}$ be the state, escape timer, and refractory timer before update $t$. Define the baseline, escaping, refractory sets:

```math
B_t=\{i:x_i^{(t)}=0,\ q_i^{(t)}=0\},\qquad
E_t=\{i:x_i^{(t)}=1\},\qquad
R_t=\{i:q_i^{(t)}\gt0\}

```

These sets partition the school under the state and timer above. For a random permutation $\pi_1,\ldots,\pi_{|B_t|}$ of the eligible baseline fish, start with $A_0=|E_t|$. At visit $k$, compute

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

---

# Phase 2: Pulse-coupled evidence integration

P2 is being rebuilt incrementally, with short, readable notebook cells.

**Current step: independent fish with partial exposure to an external stimulus.**

- [`agent.py`](P2_PulseCoupledEscape/agent.py) stores the fish's evidence and state.
- [`simulation.ipynb`](P2_PulseCoupledEscape/simulation.ipynb) applies a stimulus pulse to the first `n_exposed` fish, updates each fish independently, and plots evidence, escape spikes by fish, and shelter states.

See the [P2 guide](P2_PulseCoupledEscape/README.md) for the current step.
Social interactions will be added after reviewing the independent group response.

The [previous P2 implementation and full specification](archive/README.md)
are preserved as a reference snapshot. Phase 1 is unchanged.
