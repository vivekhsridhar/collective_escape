# Collective escape

A research model of escape cascades in a synthetic fish school, inspired by
Sosna et al. (2019). One fish starts escaping; other fish respond when evidence
from their active neighbours exceeds an individual threshold. The current
experiment tests how compressing the same school changes cascade propagation.

## Running the notebooks

Use a Python environment with NumPy, Matplotlib, and Jupyter. Open the notebooks
from the project root and run their cells in order. Parameters, experiment
setup, and plotting code stay visible in the notebooks; reusable functions live
in the Python modules beside them.

| Notebook | Purpose |
| --- | --- |
| [dose_memory.ipynb](dose_memory.ipynb) | Follow dose arrivals and finite memory with one sender and one receiver. |
| [simulation.ipynb](simulation.ipynb) | Inspect one complete cascade, including evidence, escape times, and states. |
| [baseline_calibration.ipynb](baseline_calibration.ipynb) | Calibrate the mean threshold to baseline recruitment on the synthetic school. |
| [compression_experiment.ipynb](compression_experiment.ipynb) | Compare baseline and compressed schools over paired trials. |

The calibrated threshold is entered explicitly in the simulation and compression
notebooks. Recalibrating does not automatically change those parameter cells.
The compression experiment creates `output/` and saves its figures and results there.

## Project files

| File or folder | Contents |
| --- | --- |
| [network.py](network.py) | Synthetic spatial geometry, visibility, occlusion, and response weights. |
| [agent.py](agent.py) | Each fish's threshold, evidence, state, and active timer. |
| [dose.py](dose.py) | Stochastic dose arrivals and evidence accumulated over a finite memory window. |
| [stimulus.py](stimulus.py) | Initial startle and stimulus helpers. |
| [cascade.py](cascade.py) | Reusable cascade simulation returning the number of escaped fish. |
| [output/](output/) | Generated PNG figures and NPZ simulation results. |

The notebooks and Python modules in the project root are the active implementation.

## How the cascade works

`adjacency[i, j]` is the directed sensory weight from sender `j` to receiver `i`.
Positions and visibility determine which connections exist. Distance and apparent
angular-size rank determine their weights through a logistic response function.
The network stays fixed during each cascade.

At timestep zero, one fish is activated directly. Each other fish has a threshold
drawn uniformly between zero and `2 * mean_threshold`, fixed throughout the run.
The three states are unsheltered (`0`), actively escaping (`1`), and sheltered (`2`).

On each update, an active sender delivers a dose with probability
`dose_rate * adjacency[i, j] * dt`. Each received packet contributes `dose_size`
divided by the receiver's total number of positive incoming connections,
including neighbours that are currently inactive. Evidence is the sum of doses
within `memory_duration`.

An unsheltered fish escapes when its evidence exceeds its threshold. It begins
sending doses on the next update, remains active for `active_duration`, then
enters absorbing shelter. Each fish escapes at most once. The run ends when no
active fish remain, and cascade size includes the starter. External stimuli and
fish movement are not part of the current cascade experiment.

## Baseline and compression

The current notebooks use these settings:

| Parameter | Value |
| --- | --- |
| `n_fish` | 40 |
| `mean_threshold` | 0.005596, calibrated for this synthetic baseline |
| `dt` | 0.001 s |
| `active_duration` / `memory_duration` | 0.5 s / 2 s |
| `dose_rate` / `dose_size` | 1000 per second / 0.001 |
| `intercept` / `distance_coefficient` / `rank_coefficient` | 0.06449 / -3.20552 / -0.08016 |
| `arena_size_cm` | (66.6, 66.6) |
| `min_separation_cm` / `body_width_cm` | 3 / 2 |
| `field_of_view_deg` / `occlusion` | 360 / True |
| `network_seed` | 2026 |
| `spatial_scales` | Baseline 1.0; compressed 0.7 |

`spatial_scale` multiplies each fish's displacement from the centroid. Keep
`network_seed` and the other geometry settings fixed to contract the same school.
The baseline median nearest-neighbour distance is approximately 6 cm; a scale of
0.7 reduces it to approximately 4.2 cm. Visibility and weights are recalculated
after scaling. Minimum separation is enforced during initial placement, before
contraction.

The calibrated threshold targets roughly 40% of baseline events producing at
least one secondary escape, using a visual estimate from Sosna et al.'s SI
Fig. S5A. Keep this threshold fixed when comparing spatial scales. A single
simulation can still produce only one escape; rerunning the same seeds repeats
the same event.

The saved experiment contains 1,000 paired trials on one fixed school. Each pair
shares thresholds, starter identity, and dose random seed across conditions.

| Outcome | Baseline | Compressed |
| --- | ---: | ---: |
| Mean cascade size, including starter | 1.897 | 5.226 |
| Events with secondary escapes | 36.8% | 60.2% |
| Median nearest-neighbour distance | 6.00 cm | 4.20 cm |
| Mean visible-neighbour count | 23.98 | 19.90 |

Saved outputs are the [school geometry](output/compression_geometry.png),
[cascade distributions](output/compression_cascades.png), and
[trial results](output/compression_results.npz). The NPZ stores cascade sizes
(rows are trials, columns follow `spatial_scales`), trial IDs, spatial scales,
mean threshold, and network seed. The notebook contains the remaining parameters
and paired bootstrap intervals.

## Reproducing Sosna et al. 2019

The current comparison demonstrates the qualitative mechanism: contraction
increases propagation at fixed responsiveness, despite reducing the number of
visible neighbours. It is not yet a quantitative reproduction of the empirical
cascade distributions.

The geometry uses angular width and a simple occlusion approximation. Response
coefficients come from the pooled first-exposure fit, while the implementation
uses `log10(distance_cm)`; the paper's logarithm convention remains unverified.
The threshold fits one recruitment statistic on one synthetic school.

Further comparisons should examine multiple school configurations and the full
baseline and compressed cascade distributions, alongside the paper's visual
network construction and response transformation.

Reference: [Sosna et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6789631/)
and its supporting information.
