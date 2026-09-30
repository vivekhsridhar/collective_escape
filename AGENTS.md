# Project instructions

These instructions apply throughout this repository. Add or edit rules here to
guide future coding work. More specific instructions can go in an `AGENTS.md`
inside the relevant subdirectory.

## Simulation workflow

- Use Jupyter notebooks for final simulations so the implementation and figures can be inspected together.
- Keep simulation parameters, experiment setup, and plotting code visible in the notebooks.
- Put reusable functions and classes in separate `.py` files and import them into the notebooks.
- A simulation with `n_steps` timesteps runs exactly `n_steps` updates using `range(n_steps)`, from `0` through `n_steps - 1`. Time and history arrays contain `n_steps` entries, with each result recorded at its update index. Do not add an extra timestep for the initial state.

## Coding style

- Name variables using snake_case.
- Always use double hashes (##) for commenting. Switch to single hashes inside loops or when code is indented.
- Write comments so they describe what's happening in the code block below. Do not write comments that highlight change from previous code, or comments that read as instructions. Comments should always reflect content of the code and what is being implemented.
- Do not use parameter values in comments. These change dynamically. If comments require clarification, use variable names. Never a numeric value.
- Do not overload code with checks and raising errors. This is not production level code. This is meant for anyone to follow the code and understand what is being implemented. It is research code. Always keep things simple and readable.
- At the same time, do not write trivial lines of code and comments. Very obvious lines also make reading code harder because they increase the number of lines.

## Additional instructions

- I'm new to python coding. I want to go through code and follow every step. So never write a wall of code in one go. Always write code incrementally, a few lines at a time. I will let you know once I follow existing code and we can build on things iteratively.
