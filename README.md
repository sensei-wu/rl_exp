# RL Experiments

This repository contains a small collection of reinforcement learning experiments and notebooks for learning and exploration.

## Overview

The project is organized around:
- core/: general experiment notes and exploratory work
- gymnasium_lab/: environment setup and Gymnasium-based RL examples
- gymnasium_lab/cartpole/: a CartPole notebook walkthrough and related documentation

## Project Structure

```text
.
├── README.md
├── core/
│   └── env.ipynb
├── gymnasium_lab/
│   ├── environment.yml
│   └── cartpole/
│       ├── cartpole.ipynb
│       └── README.md
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

