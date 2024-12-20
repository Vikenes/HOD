# Generating galaxy catalogues from Abacus data
Below is a short summary of what needs to be done in order to repeat inference analysis. Descriptions of which scripts that are not needed to rerun is included. 

See `old_README.md` for a detailed description of the workings of various scripts (may be outdated), focusing on the dq package mainly. 

### Scripts that do NOT need to be run 
Most scripts in this directory (not referring to scripts in subdirectories) does not need to be run. These either work, or their outputs were stored on d13. These files/directories listed below have the common root `/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/Abacus_XXX/`
 - `compute_tpcf.py`: Computes xi(r,C,G) for all halocats. The output is stored in: 
     - `~/TPCF_data/TPCF_X.hdf5` 
 - `compute_wp.py`: Computes w_p(r_z,C,G) for all halocats. The output is stored in: 
     - `~/wp_data/wp_X.hdf5` 
 - `generate_HOD_parameters_varying_ng.py`: For each abacus simulations a csv file with 500,100,100 HOD params train,test,val is created. The csv files is stored in:
     - `~/HOD_parameters/HOD_parameters_X.csv`   
 - `ng_fixed_generate_HOD_parameters.py`: Same as above, but with the *old method*, were gal. num. dens. was held fixed for each generated sample. **Obsolete**. 
 - `numdens_HOD.py`: Contains functions to adjust HOD parameter (Mmin) to obtain the desired n_g value. Used by both scripts above. 
 - `generate_pos_vel_mass_arrays.py`: Store pos, vel and mass for halos in all abacus simulations. Used to construct halocats. Output stored in:
     - `~/pos_vel_mass_arrays/L1_X.npy`
 - `make_HOD.py`: Creates the halocats for all HOD params of every sim. Output is stored in:
     - `~/HOD_catalogues/halocat_X.hdf5`


### Scripts to run 
The script `make_tpcf_emulation_data_files.py` is essentially ready to go. The output from these were stored in d5, and are hence deleted. Once this has been run, emulation can begin. 

The script does the following (see script doc. for detailed expl.):
 - Creates various hdf5 files "TPCF_{flag}.hdf5" with TPCF(r,C,G). These are used to obtain parameter priors (see parameter_samples/ paragraph).
 - Create csv files from the hdf5 files, which are the input data used during emulation.


## Subdirectories

### fiducial/
Contains data with fiducial C+G values. **Needed to run inference**.

The scripts are:
 - `store_fiducial_HOD_parameters.py`: Generates the csv files containing fid. HOD param. values. Used to make fiducial halocat. Output found in:
     - `/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/fiducial_data/HOD_parameters_fiducial.csv`
     - `/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/Abacus_XXX/HOD_parameters/HOD_parameters_fiducial.csv` (Not sure when these are used, or why)
 - `make_fiducial_HOD.py`: Creates a halocat for the fid. params. Output stored in: `/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/fiducial_data/halocat_fiducial.hdf5`  
 - `compute_fiducial_wp_and_xi.py`: Compute xi and wp from fiducial values. Although the fiducial halocat appears to exist in d13 (I can't check), it appears to load the cat. from d5. Should be easy to fix/rerun the above script. The output was stored in d5. **Output needed for inference**.



### parameter_samples/
This subdirectory contains script for various parameter investigations. Some scripts only concern plotting and studying the effects of single parameter variations etc. 
However, I suspect that some scripts are needed to generate tables/files that are used in inference analyzis later on. I will check this later. 

The scripts in the subdirectory are the following:
    - `central_occupation_distr_for_logMmin_13.py`: Plots N_c(M) (Fig. A.2 in thesis) for different values of sigma_logM and M_min, to confirm that a cutoff mass of M=12 solar masses was a reasonable choice. 
    - `plot_cosmo_params.py`: Make corner plot of cosmological parameters used in the emulator.
    - `param_table_tex.py`: Creates latex tables for HOD+cosmo prior ranges. 
    - `HOD_parameters.py`: Creates csv files *HOD_params_{flag}.csv* for train, test, val, containing all HOD params in each dataset, with each HOD param as key. Also creates histogram (Fig. A.1 in thesis) of number of samples with different M_min values. 
    - `HOD_and_cosmo_prior_ranges.py`: Script for retrieveing HOD prior ranges, and storing/retrieving cosmo prior ranges. **Important**: Requires the hdf5 files from `../make_tpcf_emulation_data_files.py` to work. 
    - `make_param_prioirs_yaml_file.py`: makes yaml file for HOD and cosmo priors. **used in MCMC**. Imports two functions from `HOD_and_cosmo_prior_ranges.py`.
     
