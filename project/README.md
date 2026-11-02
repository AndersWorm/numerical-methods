# Numerical simulation of a transmon qubit

This project simulates a superconducting transmon qubit numerically, from building the Hamiltonian to calibrating a gate and extracting the qubit's coherence times.

## Overview of the parts
- **Part 1 – Hamiltonian:** The transmon Hamiltonian is set up in the phase basis and diagonalised to find the lowest three energy levels, the transition frequency and the anharmonicity.
- **Part 2 – Hamiltonian:** The transmon Hamiltonian is set up in the charge (Cooper-pair number) basis and diagonalised to find the lowest three energy levels, the transition frequency and the anharmonicity. A time-dependent drive term is added.
- **Part 3 – Open-system dynamics:** The Lindblad master equation is solved with my own RK4 integrator, including relaxation ($T_1$) and pure dephasing ($T_2$) on a three-level system.
- **Part 4 – Pulse optimisation:** A Gaussian π-pulse is optimised with a Powell search over the drive amplitude and drive frequency, taking leakage to the third level into account.
- **Part 5 – Characterisation:** Using simulated "measurements", the Rabi frequency, $T_1$ and $T_2$ are extracted by fitting, and compared with the values put into the simulation.

## A note on the scope

I am aware that the project grew larger and more complex than the assignment required. Several parts – for example the three-level treatment, the two-dimensional pulse optimisation and the extraction of $T_2$ from the decay of the Rabi oscillations – go beyond what was asked for.

This was not an attempt to inflate the project. I did it out of my own interest in quantum physics and quantum computing, and because I wanted to understand how a real qubit is calibrated and characterised. I hope the extra material is read in that light, and that the core requirements of the assignment are still clear in the code and the results.
