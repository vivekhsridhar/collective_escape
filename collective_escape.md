# Model 1: Implementation specification for collective escape

Implement a school of fish with binary behavioral states, stochastic escape initiation, positive social feedback, and fixed-duration recovery. Build `simulation.ipynb` using the mathematical rules, data structures, and execution order specified below.

## 1. Organize the implementation

Use Python, NumPy, and Matplotlib. Place three files in the same directory:

- `agent.py`: define an ordinary `Agent` class with an `__init__` constructor. Store each agent's behavioral state in `spin` and its remaining escape time in `recovery_timer`.
- `stimulus.py`: define `square_wave(time, amplitude, duration)`, returning the amplitude where time is greater than or equal to zero and strictly less than the duration, and zero elsewhere.
- `simulation.ipynb`: import these components, define parameters, initialize the school, allocate histories, run the simulation, and plot the states. Keep those stages in separate cells so they can be executed in order.

In the agent constructor, randomly assign `spin` to 0 or 1 using NumPy's global random generator and initialize `recovery_timer` to zero. During notebook initialization, override these random states by resetting all agents to baseline before running any updates. Store and update recovery timers on the agents themselves.

## 2. Define states and parameters

For fish $i=1,\ldots,N$, let

$$
x_i\in\{0,1\},
$$

where 0 means baseline and 1 means escaping. Store $x_i$ in `Agent.spin`. Let $r_i$ be the nonnegative integer stored in `Agent.recovery_timer`. Maintain the following invariant throughout the initialized simulation: baseline fish have zero timers and escaping fish have positive timers.

Define the following parameters in a notebook cell, using these default values:

| Variable | Symbol | Value | Meaning |
| --- | --- | --- | --- |
| `n_fish` | $N$ | 20 | Number of fish |
| `total_timesteps` | $T$ | 100 | Number of updates and recorded columns |
| `random_seed` | — | 2026 | Simulation random seed |
| `J` | $J$ | 0.5 | Social coupling strength |
| `beta` | $\beta$ | 8.0 | Inverse noise parameter |
| `threshold` | $\theta$ | 0.5 | Common activation threshold |
| `escape_duration` | $d_E$ | 6 | Number of steps spent escaping |
| `stimulus_start` | $t_{\mathrm{start}}$ | 10 | First stimulus step |
| `stimulus_strength` | $s_0$ | 1 | Pulse amplitude |
| `stimulus_duration` | $d_s$ | 2 | Pulse duration |

Treat all parameters as dimensionless and express durations in integer update steps. Use positive integers for population size, simulation length, and escape duration.

## 3. Initialize the simulation

Create one NumPy random generator using `default_rng` with `random_seed`. Use this generator for every update permutation and activation draw. Construct an array of $N$ agents, then set every agent's `spin` to zero. Initialize their recovery timers to zero through the constructor.

Create the time array containing $t=0,\ldots,T-1$. Evaluate the square-wave function on time shifted by `stimulus_start`, giving

$$
s_t=
\begin{cases}
 s_0,&t_{\mathrm{start}}\leq t<t_{\mathrm{start}}+d_s,\\
 0,&\text{otherwise}.
\end{cases}
$$

With the supplied values, the stimulus is 1 at steps 10 and 11 and zero elsewhere. Apply the same stimulus to all fish.

Allocate `state_history` as an $N\times T$ array of 8-bit integers. Rows represent fish and columns represent update steps. Reserve each column for the outcome of its indexed update, starting with update 0. Use the post-update recording convention described below.

## 4. Compute activity, field, and activation probability

The activity is the fraction of fish escaping:

$$
a=\frac{1}{N}\sum_{i=1}^{N}x_i.
$$

For a baseline fish at its update turn, compute the effective field

$$
h_i=s_t+Ja_i-\theta,
$$

where $a_i$ is the activity immediately before that fish's turn. The fish share the same parameters; their fields can differ within a step because earlier fish may have activated.

Use 0/1 activity for the social term. Each escaping fish contributes $J/N$ to the field, while each baseline fish contributes zero. Thus, even one escaping fish provides positive social input when $J>0$; a majority is not required.

For comparison, if signed spins are defined as $\sigma_i=2x_i-1$, their mean is $m=2a-1$. The equivalent field is $s_t+J(m+1)/2-\theta$. Substituting $m$ directly for $a$ would change the model.

Activate a baseline fish with probability

$$
p_i=P(x_i:0\rightarrow1\mid h_i)
=\frac{1}{1+\exp(-\beta h_i)}.
$$

Draw one independent uniform number $u_i\in[0,1)$ from the simulation generator. If $u_i<p_i$, set the fish's state to 1 and its recovery timer to $d_E$. Otherwise leave it at baseline with timer zero. This is a probability per visit, with no extra time-step multiplier.

For positive $\beta$, positive fields favor activation and negative fields suppress it. At zero field, the probability is $1/2$. Larger $\beta$ makes the response sharper; in the limit $\beta\rightarrow\infty$, it approaches a threshold at $h_i=0$. At $\beta=0$, activation has probability $1/2$ regardless of the field.

Finite noise allows spontaneous escapes. With no stimulus and no escaping neighbors, the supplied parameters give

$$
p_0=\frac{1}{1+\exp(\beta\theta)}
=\frac{1}{1+\exp(4)}\approx0.018
$$

per baseline fish per step. Evaluate the exponential directly in the activation formula.

### Relation to Glauber dynamics

Glauber dynamics uses stochastic single-site updates of a spin system. In its heat-bath form, a site's state is selected according to its local conditional probability. For a binary state with local energy difference

$$
\Delta E_i=E_i(1)-E_i(0)=-h_i,
$$

the probability of selecting the active state is

$$
P(x_i=1\mid\text{other states})
=\frac{1}{1+\exp(\beta\Delta E_i)}
=\frac{1}{1+\exp(-\beta h_i)}.
$$

Use this logistic form for baseline-to-escape activation. A conventional signed spin with local energy $-H_i\sigma_i$ instead has an energy difference of $-2H_i$, producing an exponent $-2\beta H_i$. Use the binary-state convention above, with exponent $-\beta h_i$ and no factor of two.

Implement **Glauber-like activation with fixed-duration recovery**. Standard two-way heat-bath updates can select either state when a site is visited. For this model, apply stochastic activation only to baseline fish and recover escaping fish when their timers expire. Use the local energy interpretation to understand the activation formula; a global energy calculation is not required. The timer-based process should not be interpreted as equilibrium dynamics satisfying detailed balance.

## 5. Execute each update step in the specified order

At the start of step $t$, save a Boolean mask named `already_escaping` that identifies agents whose state is 1. Keep this mask unchanged throughout the step. It determines both which fish are eligible for activation and which timers must advance.

1. Count the fish in the saved mask to initialize `active_count`.
2. Obtain the indices of baseline fish in ascending order, then use the simulation generator's random permutation to determine their visitation order. Visit each exactly once.
3. Before each visit, divide the current `active_count` by $N$. Use that activity and the current stimulus to compute the field and logistic probability.
4. Draw one uniform number. If the fish activates, change its `spin` to 1, assign $d_E$ to its `recovery_timer`, and immediately increment `active_count`. Later fish see this additional social input.
5. After all activation attempts, visit the fish identified by the saved `already_escaping` mask. Decrease each one's `recovery_timer` by one. If it reaches zero, set that fish's `spin` to 0.
6. Copy all agent states into history column $t$.

Repeat for every step from 0 through $T-1$. Generate one permutation per step and one uniform draw per eligible fish, in visitation order, so that random-number consumption is reproducible.

Perform activation as a random sequential sweep within each discrete time step, then process recovery. Include fish due to recover in the social field throughout the activation sweep. Leave newly escaping fish's timers unchanged until the following step by using the saved mask to select recovery updates.

### Mathematical form of the sweep

Let $x_i^{(t)}$ and $r_i^{(t)}$ be the state and timer before update $t$. Define

$$
E_t=\{i:x_i^{(t)}=1\},\qquad
B_t=\{i:x_i^{(t)}=0\}.
$$

For a random permutation $\pi_1,\ldots,\pi_{|B_t|}$ of the baseline fish, start with $A_0=|E_t|$. At visit $k$, compute

$$
a_k=\frac{A_{k-1}}{N},\qquad
h_k=s_t+Ja_k-\theta,\qquad
p_k=\frac{1}{1+\exp(-\beta h_k)},
$$

$$
z_k=\mathbf{1}\{u_k<p_k\},\qquad A_k=A_{k-1}+z_k.
$$

The complete state and timer updates are

$$
x_i^{(t+1)}=
\begin{cases}
 z_k,&i=\pi_k\in B_t,\\
 \mathbf{1}\{r_i^{(t)}>1\},&i\in E_t,
\end{cases}
$$

$$
r_i^{(t+1)}=
\begin{cases}
 d_Ez_k,&i=\pi_k\in B_t,\\
 r_i^{(t)}-1,&i\in E_t.
\end{cases}
$$

### Recovery timing example

With $d_E=6$, a fish activated during step $t$ appears as escaping in recorded columns $t$ through $t+5$. It recovers at the end of step $t+6$, after contributing to that step's social field. It becomes eligible for activation again at step $t+7$.

Make recovered fish eligible again on the following step, without an additional refractory period. Use deterministic timer expiration for recovery and leave an already escaping fish's timer unaffected by further stimulus exposure.

## 6. Record and visualize the result

After each complete update, including recovery, record

$$
\texttt{state\_history}[i,t]=x_i^{(t+1)}.
$$

Store the outcome of update 0 in column 0 and retain exactly $T$ post-update columns. Retain the time and stimulus arrays and leave the final states and recovery timers on the agents after the run.

Create a Matplotlib figure of size 12 by 5 inches with constrained layout. Draw `state_history` using `pcolormesh`, with the time array on the horizontal axis and fish indices on the vertical axis. Use a two-color map: blue (`#2166ac`) for baseline and red (`#b2182b`) for escape. Label every fish index, label the axes “Time (simulation steps)” and “Fish,” and title the demonstration “Escape states of a school of 20 fish.” Add a dashed black vertical line of width 2 at stimulus onset. Add a colorbar labeled “State,” with ticks at 0 and 1 labeled “Baseline (0)” and “Escape (1).” Display the figure.

For interpretation, define the escape-fraction trajectory as

$$
f_{\mathrm{escape}}[t]
=\frac{1}{N}\sum_i\texttt{state\_history}[i,t].
$$

This equals activity at the end of each step. Use the state raster as the required visualization; a separate calculation or plot of population summaries is not required.

## 7. Check behavior and reproducibility

Arrange the cells so that executing them in order resets the generator, agents, and arrays before a fresh trial. To start another trial, rerun initialization and history allocation before the simulation loop.

Check that the history has shape $(20,100)$ and contains only 0 and 1, and that the pulse is nonzero only at steps 10 and 11. Confirm that new escapes affect later visits immediately, that only previously escaping agents have their timers decremented, and that a recovering fish cannot reactivate in the same step. Repeating a complete run with the same seed in the same software environment should reproduce the trajectory.

Deliver a notebook that runs one trial from an all-baseline school, applies the pulse to every fish, and produces one state raster. Keep the scope to this population-level model; spatial interactions, seeded escaping minorities, repeated-trial statistics, and parameter sweeps are not required.
