# portHamiltonian_FEM
This repository contains the FEniCS implementation of the paper "Port-Hamiltonian Symplectic Coupling of the Wave Equation
with a Lumped Parameter Model" - by Finbar Argus, Chris Bradley and Peter Hunter.

For replication of the simulations done in the above paper use the branch stableForPaper.

The different cases in the paper code can be chosen and run with main_wave.py. main_eigen.py runs the eigenvalue analysis and plotting_wave_2D.py runs
the plot creation for all of the results.

There is currently also work being done on implementing control through the coupled wave-electromechinical model. This is being implemented in
main_control.py and also in wave_2D.py. 