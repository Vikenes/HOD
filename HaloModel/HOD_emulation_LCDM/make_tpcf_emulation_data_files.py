import numpy as np 
from pathlib import Path
import pandas as pd
import h5py

### TBD:
# Take log of xi? 
# Check if both hdf5 AND csv files are needed.
# Check if hdf5 files are needed at all.


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

DATA_PATH           = Path("/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation/TPCF_emulation")
TPCF_DATAPATH       = Path(DATA_PATH / "corrfunc_arrays")
HOD_PARAMETERS_PATH = Path(DATA_PATH / "HOD_parameters")
EMULATION_DATA_PATH = Path(DATA_PATH / "emulation_files")

dataset_names = ["train", "test", "val"]




def make_TPCF_HDF_files_arrays_at_sliced_r(r_low=0.6, r_high=60, 
                                           log10=True, overwrite=False):
    """
    Create hdf5 files with TPCF data for each node in each data set.
    containing the HOD parameters for each node.
    The HDF5 data is then used to create csv files, which is the data actually used for emulation.

    r_low and r_high are the lower and upper limits of the r interval used for the TPCF data.
    TPCF data outside this interval is noisy, and therefore left out.
    """

    # Loop over data sets
    for flag in dataset_names:
        # Create hdf5 file
        outfile = f"{EMULATION_DATA_PATH}/TPCF_{flag}.hdf5"
        if Path(outfile).exists() and not overwrite:
            print(f"File {outfile} already exists. Skipping.")
            # continue
        file_tpcf_h5py = h5py.File(outfile, 'w')

        # Read HOD parameters
        parameters_file = Path(f"HOD_parameters_{flag}_ng_fixed.csv")
        parameters_df   = pd.read_csv(HOD_PARAMETERS_PATH / parameters_file)

        # Make list of npy files containing TPCF data
        # Numeric iteration ensures correct correspondance between TPCF and HOD parameters
        N_files     = len(parameters_df)
        NPY_FILES   = [Path(f"{TPCF_DATAPATH}/TPCF_{flag}_node{i}_ng_fixed.npy") for i in range(N_files)]

        # Read r values from first file
        # Make mask, omitting the noisy data outside r_low and r_high
        r_common            = np.load(NPY_FILES[0])[0]
        r_mask_low_limit    = r_common > r_low
        r_mask_high_limit   = r_common < r_high
        r_mask              = r_mask_low_limit * r_mask_high_limit

        # Loop over nodes
        for node_idx, npy_file in enumerate(NPY_FILES):
            # Ensure correct correspondance between TPCF and HOD parameters
            npy_node_number = int(npy_file.stem.split("_")[2][4:])
            if node_idx != npy_node_number:
                print('Error. Node index does not match node number in file name.')
                exit()

            # Load TPCF data
            r, xi = np.load(npy_file)[:, r_mask]

            # Create data group
            node_group = file_tpcf_h5py.create_group(f'node{node_idx}')

            # Store HOD parameters in hdf5 file
            node_group.attrs['sigma_logM']  = parameters_df['sigma_logM'].iloc[node_idx]
            node_group.attrs['alpha']       = parameters_df['alpha'].iloc[node_idx]
            node_group.attrs['kappa']       = parameters_df['kappa'].iloc[node_idx]
            node_group.attrs['log10M1']     = parameters_df['log10M1'].iloc[node_idx]
            node_group.attrs['log10Mmin']   = parameters_df['log10Mmin'].iloc[node_idx]            

            # Store TPCF data in hdf5 file
            if log10:
                r_data_name     = 'log10r'
                xi_data_name    = 'log10xi'
                r_data          = np.log10(r)
                xi_data         = np.log10(xi)
            else:
                r_data_name     = 'r'
                xi_data_name    = 'xi'
                r_data          = r
                xi_data         = xi

            node_group.create_dataset(r_data_name, data=r_data)
            node_group.create_dataset(xi_data_name, data=xi_data)

        file_tpcf_h5py.close()


def hdf5_to_csv(log10=True):
    """
    Make csv files from hdf5 files.
    This is the data actually used for emulation. 
    """
    for flag in dataset_names:
    ### Create csv file from hdf5 file 
    
        file_tpcf_h5py = h5py.File(f'{EMULATION_DATA_PATH}/TPCF_{flag}.hdf5', 'r')

        _out_list = []
        for key in file_tpcf_h5py.keys():
            if log10:
                r_data_name     = 'log10r'
                xi_data_name    = 'log10xi'
                r_data          = np.log10(file_tpcf_h5py[key]['r'][...])
                xi_data         = np.log10(file_tpcf_h5py[key]['xi'][...])
            else:
                r_data_name     = 'r'
                xi_data_name    = 'xi'
                r_data          = file_tpcf_h5py[key]['r'][...]
                xi_data         = file_tpcf_h5py[key]['xi'][...]
            df = pd.DataFrame({
                'sigma_logM' : file_tpcf_h5py[key].attrs['sigma_logM'],
                'alpha'      : file_tpcf_h5py[key].attrs['alpha'],
                'kappa'      : file_tpcf_h5py[key].attrs['kappa'],
                'log10M1'    : file_tpcf_h5py[key].attrs['log10M1'],
                'log10Mmin'  : file_tpcf_h5py[key].attrs['log10Mmin'],
                r_data_name  : r_data,
                xi_data_name : xi_data,
            })
            _out_list.append(df)
        
        df_all = pd.concat(_out_list)
        df_all.to_csv(
            f'{EMULATION_DATA_PATH}/TPCF_{flag}.csv',
            index=False,
        )
        
        file_tpcf_h5py.close()

# slice_TPCF_arrays_at_r_interval()
# make_TPCF_HDF_files_arrays_at_sliced_r()
hdf5_to_csv(log10=True)
