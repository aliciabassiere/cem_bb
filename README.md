**"Capacity Expansion Model (CEM-BB)"**  
*by Alicia Bassière and David Benatia.*

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [License](#license)

## Project Overview

The framework uses the **generation expansion model introduced in the paper "Moving Forward Blindly: Uncertainty, Reliability, and the Energy Transition"** to evaluate and optimize energy portfolios under deterministic, stochastic, and perfect foresight planning scenarios. It optimizes on short-term capacity investments and long-term demand rationing, with **uncertainties** in demand, supply, and policy pathways.


**Planners included:**
- **Stochastic Planner**: Optimizes under uncertainty using scenario generation.
- **Deterministic Planner**: Uses a reference scenario for optimization.
- **Perfect Foresight Planner**: Assumes perfect knowledge of future scenarios.

## Features

- **Scenario Generation**: Quasi-random Sobol sequences to model uncertain demand.
- **Cost Models**: Detailed computation of investment and operational costs.
- **Constraints**: Carbon targets, wind and solar capacity limits, gas constraints.
- **Gradient Boosting**: Implements optimization using advanced machine learning techniques.
- **Result Visualization**: Generates CSV files and visual outputs for analysis.

## Installation

### Prerequisites
- Python 3.8+
- Required libraries (see `requirements.txt`).

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/DynamicInvestmentProgramming.git
   cd DynamicInvestmentProgramming

2. Set up a virtual environment and install dependencies:
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt

### Usage

To run a simulation with the corresponding planner:

1. Modify key parameters in `simulation_parameters.py`.
2. Launch the corresponding notebook.

```bash
# Example command to launch Jupyter and open the stochastic planner notebook
jupyter notebook stochastic_planner.ipynb

### Outputs
Results are saved in the outputs directory, organized by batch simulation.

### Project Structure

DynamicInvestmentProgramming/
├── inputs/                      # Input data for simulations
├── outputs/                     # Simulation results
├── stochastic_planner.ipynb     # Stochastic planner notebook
├── deterministic_planner.ipynb  # Deterministic planner
├── perfect_foresight_planner.ipynb # Perfect foresight planner
├── simulation_parameters.py     # Global simulation configuration
├── generate_scenario.py         # Scenario generation
├── cost_functions.py            # Cost computations
├── constraints.py               # Carbon and capacity constraints
├── results_writing.py           # Result saving utilities
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation


## License and Ownership
This code is the proprietary property of Alicia Bassiere. Redistribution, modification, or use without explicit permission is prohibited.

