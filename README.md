# HOD
Generate mock galaxy catalogues from dark matter (DM) halo data for a given Halo Occupation Distribution (HOD) parameterization. 
Constructs a large number of catalogues from each DM simulation, and compute the Two-point correlation function (TPCF) of galaxies. 
The resulting data is used to construct a neural network emulator to predict the TPCF given a set of input cosmological parameters.

Also generate mock signal and compute the covariance matrix of mock signals to perform cosmological parameter inference.
 
### DarkQuest

Main addition to the original package is: 
 - The file `cosmological_parameters.dat`, containing the cosmological parameters for all Abacus simulations  
 - The function `from_custom` in `cosmology.py`, used to load the cosmology with the Abacus parameters. 

### HaloModel

The main scripts for generating halocats.
See separate README of this subdirectory. 