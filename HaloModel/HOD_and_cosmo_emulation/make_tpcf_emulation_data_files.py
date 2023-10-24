import numpy as np 
from pathlib import Path
import pandas as pd
import h5py
import time 

################# 
##### TBD: ######
#################

# Save one TPCF hdf5 file for xi only, and one for xi/xi_fiducial?
# Do all manipulations of the TPCF's in the end when generating csv files? 


"""
Use HOD galaxy catalogues to store TPCF data.
Make hdf5 and csv files with TPCF data for each node in each data set, with columns:
    - sigma_logM
    - alpha
    - kappa
    - log10M1
    - log10Mmin
    - r
    - xi
"""
D13_BASE_PATH       = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit")
D13_EMULATION_PATH  = Path(D13_BASE_PATH / "emulation_files")
D13_OUTPATH         = Path(D13_EMULATION_PATH / "TPCF_emulation")



DATASET_NAMES = ["train", "test", "val"]
COSMOLOGY_PARAM_KEYS = ["wb", "wc", "Ol", "lnAs", "ns", "w", "Om", "h", "N_eff"]

# Make list of all simulations containing emulation files 
get_path         = lambda version, phase: Path(D13_EMULATION_PATH / f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}")
Phase_paths      = [get_path(0, ph) for ph in range(25) if get_path(0, ph).is_dir()]
SIMULATION_PATHS = Phase_paths + [get_path(v, 0) for v in range(1,182) if get_path(v, 0).is_dir()]



def make_TPCF_HDF_files_arrays_at_sliced_r(
    version: int = 0,
    phase: int = 0,
    r_low=0.6, 
    r_high=100,
    log10=True, 
    key="N_eff",
    ):
    ##### NOT READY YET #####
    ##### Must decide which cosmologies to use for train/test/val #####
    ##### and which cosmological parameters to use for each cosmology #####


    exit()



    """
    Create hdf5 files with TPCF data for each node in each data set.
    containing the HOD parameters for each node.
    The HDF5 data is then used to create csv files, which is the data actually used for emulation.

    r_low and r_high are the lower and upper limits of the r interval used for the TPCF data.
    TPCF data outside this interval is noisy, and therefore left out.
    """

    # Check if simulation path exists
    SIMULATION_PATH     = Path(D13_EMULATION_PATH / f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}")
    if not SIMULATION_PATH.exists():
        print(f"Error! Simulation path {SIMULATION_PATH} does not exist. Aborting...")
        exit()

    # Path of HOD parameters for simulation 
    HOD_PARAMETERS_PATH = Path(SIMULATION_PATH / "HOD_parameters")
    cosmo_dict          = pd.read_csv(Path(SIMULATION_PATH / "cosmological_parameters.dat"), 
                                      sep=" "
                                      ).iloc[0].to_dict()

    outfname = f"TPCF_{key}.hdf5"
    if log10:
        outfname = f"log_{outfname}"
    
    outfile = Path(D13_OUTPATH / outfname)
    fff     = h5py.File(outfile, "w")
    for key, value in cosmo_dict.items():
        fff.attrs[key] = value




    # print(cosmo_dict["N_eff"])
    # return cosmo_dict[key]
    exit()

    # # Loop over data sets
    # for flag in dataset_names:
        
    #     # Read HOD parameters
    #     parameters_file = Path(f"HOD_parameters_{flag}_ng_fixed.csv")
    #     parameters_df   = pd.read_csv(HOD_PARAMETERS_PATH / parameters_file)

    #     # Make list of npy files containing TPCF data
    #     # Numeric iteration ensures correct correspondance between TPCF and HOD parameters
    #     N_files     = len(parameters_df)
    #     NPY_FILES   = [Path(f"{TPCF_DATAPATH}/TPCF_{flag}_node{i}_ng_fixed.npy") for i in range(N_files)]

    #     # Read r values from first file
    #     # Make mask, omitting the noisy data outside r_low and r_high
    #     r_common            = np.load(NPY_FILES[0])[0]
    #     r_mask_low_limit    = r_common > r_low
    #     r_mask_high_limit   = r_common < r_high
    #     r_mask              = r_mask_low_limit * r_mask_high_limit

    #     exit()
        
    #     # Create hdf5 file
    #     outfname = f"TPCF_{flag}.hdf5"
    #     if log10:
    #         outfname = f"log_{outfname}"
    #     outfile = f"{D13_OUTPATH}/{outfname}"

        
    #     print(f"Creating file {outfile}")
    #     file_tpcf_h5py = h5py.File(outfile, 'w')
    #     # Loop over nodes
    #     for node_idx, npy_file in enumerate(NPY_FILES):
    #         # Ensure correct correspondance between TPCF and HOD parameters
    #         npy_node_number = int(npy_file.stem.split("_")[2][4:])
    #         if node_idx != npy_node_number:
    #             print('Error. Node index does not match node number in file name.')
    #             exit()

    #         # Load TPCF data
    #         r, xi = np.load(npy_file)[:, r_mask]

    #         # Create data group
    #         node_group = file_tpcf_h5py.create_group(f'node{node_idx}')

    #         # Store HOD parameters in hdf5 file
    #         node_group.attrs['sigma_logM']  = parameters_df['sigma_logM'].iloc[node_idx]
    #         node_group.attrs['alpha']       = parameters_df['alpha'].iloc[node_idx]
    #         node_group.attrs['kappa']       = parameters_df['kappa'].iloc[node_idx]
    #         node_group.attrs['log10M1']     = parameters_df['log10M1'].iloc[node_idx]
    #         node_group.attrs['log10Mmin']   = parameters_df['log10Mmin'].iloc[node_idx]            

    #         # Store TPCF data in hdf5 file
    #         if log10:
    #             r_data_name     = 'log10r'
    #             xi_data_name    = 'log10xi'
    #             r_data          = np.log10(r)
    #             xi_data         = np.log10(xi)
    #         else:
    #             r_data_name     = 'r'
    #             xi_data_name    = 'xi'
    #             r_data          = r
    #             xi_data         = xi

    #         node_group.create_dataset(r_data_name, data=r_data)
    #         node_group.create_dataset(xi_data_name, data=xi_data)

    #     file_tpcf_h5py.close()


def make_TPCF_ratio_HDF_files_arrays_at_sliced_r(
    r_low:      float = 0.6, 
    r_high:     float = 100.0,
    ng_fixed:   bool = True,
    log_r:      bool = True,
    ):

    
    if log_r:
        TPCF_OUTPATH = Path(D13_EMULATION_PATH / "TPCF_emulation/log_r_xi_over_xi_fiducial")
    else:
        TPCF_OUTPATH = Path(D13_EMULATION_PATH / "TPCF_emulation/xi_over_xi_fiducial")

    TPCF_OUTPATH.mkdir(parents=True, exist_ok=True)


    """
    Create hdf5 files with TPCF data for each node in each data set.
    containing the HOD parameters for each node.
    The HDF5 data is then used to create csv files, which is the data actually used for emulation.

    r_low and r_high are the lower and upper limits of the r interval used for the TPCF data.
    TPCF data outside this interval is noisy, and therefore left out.
    """
    
    TPCF_fiducial_fname = "TPCF_fiducial"
    
    if ng_fixed:
        ng_suffix = "_ng_fixed"
    else:
        ng_suffix = ""
    
    # Load fiducial TPCF data
    TPCF_fiducial_fname += f"{ng_suffix}.hdf5"
    TPCF_fiducial = h5py.File(SIMULATION_PATHS[0] / f"TPCF_data/{TPCF_fiducial_fname}", "r")
    r_fiducial    = TPCF_fiducial["node0"]["r"][:]

    # Make mask, omitting the noisy data outside r_low and r_high
    r_mask_low_limit    = r_fiducial > r_low
    r_mask_high_limit   = r_fiducial < r_high
    r_mask              = r_mask_low_limit * r_mask_high_limit
    xi_fiducial         = TPCF_fiducial["node0"]["xi"][:][r_mask]


    for flag in DATASET_NAMES:
        filename_suffix = f"{flag}{ng_suffix}"
        
        # Create outfile 
        outfname = f"TPCF_{filename_suffix}.hdf5"
        outfile  = Path(TPCF_OUTPATH / outfname)
        
        print(f"Storing TPCF's for {outfile.parent.name}/{outfile.name}")
        fff     = h5py.File(outfile, "w")

        t0 = time.time()
        for SIMULATION_PATH in SIMULATION_PATHS:

            fff_cosmo  = fff.create_group(SIMULATION_PATH.name)
            cosmo_dict = pd.read_csv(Path(SIMULATION_PATH / "cosmological_parameters.dat"), 
                                     sep=" "
                                     ).iloc[0].to_dict()

            for key in COSMOLOGY_PARAM_KEYS: 
                # Store cosmological parameters in hdf5 file
                fff_cosmo.attrs[key] = cosmo_dict[key]

            # Path of HOD parameters for simulation 
            HOD_PARAMETERS_PATH = Path(SIMULATION_PATH / "HOD_parameters")
            hod_params_fname    = Path(f"HOD_parameters_{filename_suffix}.csv")
            node_params_df      = pd.read_csv(HOD_PARAMETERS_PATH / hod_params_fname)
            
            TPCF_data = h5py.File(SIMULATION_PATH / f"TPCF_data/TPCF_{filename_suffix}.hdf5", "r")
            for node_idx in range(len(node_params_df)):
                node_group = fff_cosmo.create_group(f"node{node_idx}")

                # Store HOD parameters in hdf5 file
                node_group.attrs['sigma_logM']  = node_params_df['sigma_logM'].iloc[node_idx]
                node_group.attrs['alpha']       = node_params_df['alpha'].iloc[node_idx]
                node_group.attrs['kappa']       = node_params_df['kappa'].iloc[node_idx]
                node_group.attrs['log10M1']     = node_params_df['log10M1'].iloc[node_idx]
                node_group.attrs['log10Mmin']   = node_params_df['log10Mmin'].iloc[node_idx]

                r  = TPCF_data[f"node{node_idx}"]["r"][:][r_mask]
                xi = TPCF_data[f"node{node_idx}"]["xi"][:][r_mask]

                xi_over_xi_fiducial = xi / xi_fiducial
                if log_r:
                    r_name = "log10r"
                    r_data = np.log10(r)
                else:
                    r_name = "r"
                    r_data = r

                node_group.create_dataset(r_name, data=r_data)
                node_group.create_dataset("xi_over_xi_fiducial", data=xi_over_xi_fiducial)

        
        fff.close()
        print("========================================================")
        print(f"Done with {outfile.parent.name}/{outfile.name}. Took {time.time() - t0:.2f} s")
        print("========================================================")
        print()

                   
# make_TPCF_ratio_HDF_files_arrays_at_sliced_r(log_r=True)
# make_TPCF_ratio_HDF_files_arrays_at_sliced_r(log_r=False)


def hdf5_to_csv(
        ng_fixed: bool = True,
        xi_ratio: bool = True,
        log_xi: bool = False,
        log_r: bool = False
        ):
    """
    Make csv files from hdf5 files.
    This is the data actually used for emulation. 
    """
    if xi_ratio:
        datafolder = "xi_over_xi_fiducial"
        if log_r:
            datafolder = "log_r_" + datafolder
    else:
        datafolder = "xi"
        if log_xi:
            datafolder = "log_" + datafolder
        if log_r:
            datafolder = "log_r_" + datafolder
        
    TPCF_HDF5_PATH = Path(D13_EMULATION_PATH / f"TPCF_emulation/{datafolder}")

    if ng_fixed:
        ng_suffix = "_ng_fixed"
    else:
        ng_suffix = ""
    

    for flag in DATASET_NAMES:
        ### Create csv file from hdf5 file 

        print(f"Making csv files in {datafolder} for {flag}...")
        t0 = time.time()
        filename_suffix     = f"{flag}{ng_suffix}"
        TPCF_hdf5_filename  = Path(f"TPCF_{filename_suffix}.hdf5")
    
        fff_TPCF = h5py.File(TPCF_HDF5_PATH / TPCF_hdf5_filename, 'r')

        _out_list = []
        for SIMULATION_PATH in SIMULATION_PATHS:
            fff_TPCF_cosmo = fff_TPCF[SIMULATION_PATH.name]
            cosmo_params_dict = {key: fff_TPCF_cosmo.attrs[key] for key in fff_TPCF_cosmo.attrs.keys()}

            for node_idx in range(len(fff_TPCF_cosmo)):
                fff_TPCF_cosmo_HOD = fff_TPCF_cosmo[f"node{node_idx}"]

                r_key   = "r" if not log_r else "log10r"
                xi_key  = "xi_over_xi_fiducial" if xi_ratio else "xi"
                r_data  = fff_TPCF_cosmo_HOD[r_key][...]
                xi_data = fff_TPCF_cosmo_HOD[xi_key][...]


                HOD_params_dict = {HOD_param: fff_TPCF_cosmo_HOD.attrs[HOD_param] for HOD_param in fff_TPCF_cosmo_HOD.attrs.keys()}
                tot_params_dict = cosmo_params_dict | HOD_params_dict
                df = pd.DataFrame({
                    **tot_params_dict,
                    r_key  : r_data,
                    xi_key : xi_data,
                })
                _out_list.append(df)
            
            # print(f"{SIMULATION_PATH.name} Finished, took {time.time() - t0:.2f} s")

            
        df_all = pd.concat(_out_list)
        df_all.to_csv(
            f'{TPCF_HDF5_PATH}/TPCF_{filename_suffix}.csv',
            index=False,
        )
        fff_TPCF.close()
        print(f"Done. Took {time.time() - t0:.2f} s")
        print()

# slice_TPCF_arrays_at_r_interval()
# make_TPCF_HDF_files_arrays_at_sliced_r()
# make_TPCF_ratio_HDF_files_arrays_at_sliced_r()
hdf5_to_csv()
