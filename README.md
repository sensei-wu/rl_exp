# RL Experiments

This repository contains a small collection of reinforcement learning experiments and notebooks for learning and exploration.

## Overview

The project is organized around:
- gymnasium_lab/: environment setup and Gymnasium-based RL examples
- gymnasium_lab/cartpole/: a CartPole notebook walkthrough and related documentation
- gymnasium_lab/core/: supporting notebook work and environment experiments
- gymnasium_lab/taxi/: a Taxi environment implementation and example code

## Project Structure

```text
.
├── README.md
├── .gitignore
├── gymnasium_lab/
│   ├── environment.yml
│   ├── core/
│   │   └── env.ipynb
│   ├── cartpole/
│   │   ├── cartpole.ipynb
│   │   └── README.md
│   └── taxi/
│       ├── README.md
│       └── taxi.py
```

## Prerequisites

- Python 3.11
- Conda or Mamba

## Setup

Create and activate the environment:

```bash
conda env create -f gymnasium_lab/environment.yml
conda activate rl
```

## Running the Notebooks

Start Jupyter from the repository root:

```bash
jupyter lab
```

Then open the notebooks in:
- core/env.ipynb
- gymnasium_lab/cartpole/cartpole.ipynb

## Notes

The environment file installs common RL dependencies such as PyTorch, Gymnasium, NumPy, Matplotlib, and Plotly.

