# Generating galaxy catalogues from Abacus data

Galaxy catalogues are created with the script `make_HOD.py`. Starting with Abacus simulation data, and provided that the prerequisites are in place, galaxy catalogues are generated via the following pipeline:

0. `make_fiducial_xi_sample.py`: Performs steps 1.-3. in the list below for the fiducial $\Lambda\mathrm{CDM}$ cosmology and the central values of the HOD parameters, to be used as a fiducial data set during emulation. 
1. `generate_HOD_parameters.py`: Creates 700 HOD parameter sets for each simulation. If `fix_ng=True`, the parameter $M_\mathrm{min}$ (named log10Mmin) is adjusted to match a desired expected galaxy number density, with the function `estimate_log10Mmin_from_gal_num_density` from the script `numdens_HOD.py`. It outputs `csv` files containing all HOD parameters for each simulation.
2. `make_HOD.py`: Populates halos with galaxies according to the HOD parameters. For each simulation, each of the 700 HOD parameter sets yield one galaxy catalogue. The catalogues store various quantities, such as position and velocities of the galaxies. 
3. `compute_tpcf.py`: Using galaxy catalogues as input, it computes the two-point correlation function (2PCF) for each catalogue in each simulation. All correlation functions are computed with respect to a common binning defined by the user.
4. `make_tpcf_emulation_data_files.py`: Prepare data for emulation. 
a) `make_TPCF_HDF_files_arrays_at_fixed_r`: Stores all TPCFs data contained in each node for every cosmology in a single hdf5 file, storing all cosmological and HOD parameters as well. Allows for simple extraction of desired emulation data, e.g. bin range, cosmological feature parameters etc. Uses the same bin values for ALL, allowing for e.g. ratios to be emulated.  
b) `xi_over_xi_fiducial_hdf5`: Using the main hdf5 file, creates hdf5 files for train/test/val with data of xi/xi_fiducial. Here, one specifies the bin range to consider, as well as the HOD and cosmological parameters to be used as emulator features. The output files are used to create csv files, as described in 4c), and to extract data to be used to evaluate the emulator after training.   
c) `xi_over_xi_fiducial_hdf5_to_csv`: From the hdf5 files created in 4b), create csv files used as the actual input for the emulator. 

## 1. Prerequisites
Before catalogues can be created, some preparation of the data and prerequisites are required. 
#### 1.1 Storing Arrays of halo positions, velocities and mass
To reduce computational cost, we only consider halos of mass $M > 10^{12}\,\mathrm{M_\odot}/h$, as smaller halos have a low probability of containing galaxies at late redshift. We therefore want to store arrays containing the positions, velocities and masses of all massive halos in each simulation. These halos are the ones that we'll populate with galaxies later on. 

**Important:** These arrays are also used to constrain the galaxy number density, so omitting low mass halos affects the resulting HOD parameters.   

To create and store the arrays, run the script `generate_pos_vel_mass.py`, which outputs three arrays, `L1_*.npy` in `/.../d13/.../AbacusSummit/emulation_files/AbacusSummit_base_c***_ph***/pos_vel_mass_arrays/`.
The resulting arrays are then loaded, and used as input to create the halo catalogues, which in turn are populated with galaxies. 

#### 1.2 Creating .dat file with cosmological parameters 
To make halo catalogues, one needs to define the cosmology. The cosmology is set defined in the `dq` class, by reading parameter values from a `.dat` file, which is required to make halo catalogues. 

To create the `.dat` files, run the script `make_cosmo_params_dat_files.py`. Some parameters are retrieved from the header file of the Abacus data, while the remaining parameters are read from the table `AbacusSummit/cosmologies.csv`, which is downloaded from the AbacusSummit website. The resulting output for a given version is stored in `.../emulation_files/AbacusSummit_base_c***_ph***/cosmological_parameters.dat`, which is where `dq` reads it.


## 2. Generate HOD parameter 
Describe in detail what `generate_HOD_parameters.py` does

## 3. Make HOD catalogues 
Describe in detail what `make_HOD.py` does

## 4. Compute TPCF
Describe in detail what `compute_tpcf.py` does

## 5. Make emulation data files 
Describe in detail what `make_tpcf_emulation_data_files.py` does

## 6. Other scripts 
Describe other scripts in thse directory not directly connected to the main pipeline:
 - `change_r_bins_in_TPCF_data.py`
 - `make_fiducial_xi_sample.py`
 - `/parameter_samples_plot/`
 

