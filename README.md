
# Lennard-Jones 2D Simulation

This Python script simulates a 2D system of particles interacting via the Lennard-Jones potential, using both NVE (microcanonical) and NVT (canonical with Berendsen thermostat) ensembles. The simulation tracks energy, temperature, momentum, and radial distribution function (RDF), with visualizations of particle dynamics and statistical properties.

## Overview

The `LennardJones2D` class implements:
- **Particle Dynamics**: Particles move according to the Lennard-Jones potential, with periodic boundary conditions and minimum image convention.
- **Ensembles**:
  - **NVE**: Constant number of particles, volume, and energy.
  - **NVT**: Constant number of particles, volume, and temperature (using Berendsen thermostat).
- **Tracking**: Kinetic energy, potential energy, total energy, temperature, momentum, and RDF.
- **Visualization**: Plots of energy evolution, temperature, momentum, RDF, and particle distribution.

The script runs multiple cases to explore the effects of particle number (`N`), time step (`dt`), and temperature (`T`).

## Requirements

- Python 3.x
- Libraries:
  - `numpy` (for numerical operations)
  - `matplotlib` (for plotting)
  - `collections` (for `defaultdict`)
  - `time` (for timing)

Install the required libraries using pip:
```bash
pip install numpy matplotlib
```

## Usage

1. Save the script as `lennard_jones_2d.py`.
2. Adjust global parameters (e.g., `L`, `rc`, `dt`, `num_steps_nve`, `num_steps_nvt`) or class initialization parameters as needed.
3. Run the script:
   ```bash
   python lennard_jones_2d.py
   ```
4. The script will execute multiple simulation cases and display plots for each case.

## Parameters

- `L = 30.0`: Size of the simulation box (length of square domain).
- `rc = 2.5`: Cutoff radius for Lennard-Jones interactions.
- `N`: Number of particles (e.g., 100, 400, 625, 900).
- `dt`: Time step (e.g., 0.01, 0.005, 0.02).
- `m = 1.0`: Particle mass (default).

### Simulation Parameters
- `initial_T`: Initial temperature (e.g., 0.5 for NVE, 0.1 or 1.0 for NVT).
- `ensemble`: 'NVE' or 'NVT'.
- `target_T`: Target temperature for NVT (required for NVT ensemble).
- `tau_dt_ratio = 0.0025`: Ratio of thermostat relaxation time to time step for NVT.

## Output

- **Plots**: Generated for each simulation case, including:
  - Energy evolution (kinetic, potential, total).
  - Temperature evolution.
  - Total momentum magnitude (log scale).
  - Radial distribution function (RDF) for NVT cases.
  - Final particle distribution.
- **Console Output**: Progress updates every 1000 steps with energy, temperature, and momentum values, plus total simulation time.

## Simulation Cases

### Part a) NVE Simulations


### Part b) NVT Simulations with Berendsen Thermostat
