import numpy as np 
from pathlib import Path
import pandas as pd
import h5py

"""
Make csv files with TPCF data for each node in each data set, with columns:
    - sigma_logM
    - alpha
    - kappa
    - log10M1
    - log10Mmin
    - r
    - xi

"""

LCDM_PATH = Path("/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation")
TPCF_DATAPATH = Path(LCDM_PATH / "emulation_data")
PARAMS_INPATH = Path(LCDM_PATH / "HOD_data")

dataset_names = ["train", "test", "val"]




def make_TPCF_HDF_files_arrays_at_sliced_r(r_low=0.6, r_high=60):
    for flag in dataset_names:
        file_tpcf_h5py = h5py.File(f'{TPCF_DATAPATH}/TPCF_{flag}.hdf5', 'w')

        parameters = Path(f"HOD_parameters_ng_fixed_{flag}.csv")
        parameters_df = pd.read_csv(PARAMS_INPATH / parameters)
        N_files = len(parameters_df)

        NPY_FILES = [Path(f"{TPCF_DATAPATH}/TPCF_{flag}_node{i}_115bins_ng_fixed.npy") for i in range(N_files)]

        r_common = np.load(NPY_FILES[0])[0]
        r_mask_low_limit  = r_common > r_low
        r_mask_high_limit = r_common < r_high
        r_mask = r_mask_low_limit * r_mask_high_limit


        for node_idx, npy_file in enumerate(NPY_FILES):
            npy_node_number = int(npy_file.stem.split("_")[2][4:])
            if node_idx != npy_node_number:
                print('Error. Node index does not match node number in file name.')
                exit()

            r, xi = np.load(npy_file)[:, r_mask]

            node_group = file_tpcf_h5py.create_group(f'node{node_idx}')

            node_group.attrs['sigma_logM']  = parameters_df['sigma_logM'].iloc[node_idx]
            node_group.attrs['alpha']       = parameters_df['alpha'].iloc[node_idx]
            node_group.attrs['kappa']       = parameters_df['kappa'].iloc[node_idx]
            node_group.attrs['log10M1']     = parameters_df['log10M1'].iloc[node_idx]
            node_group.attrs['log10Mmin']   = parameters_df['log10Mmin'].iloc[node_idx]            
            node_group.create_dataset('r', data=r)
            node_group.create_dataset('xi', data=xi)

        file_tpcf_h5py.close()


def hdf5_to_csv():
    for flag in dataset_names:
    ### Create csv file from hdf5 file 
    
        file_tpcf_h5py = h5py.File(f'{TPCF_DATAPATH}/TPCF_{flag}.hdf5', 'r')

        _out_list = []
        for key in file_tpcf_h5py.keys():

            df = pd.DataFrame({
                'sigma_logM': file_tpcf_h5py[key].attrs['sigma_logM'],
                'alpha'     : file_tpcf_h5py[key].attrs['alpha'],
                'kappa'     : file_tpcf_h5py[key].attrs['kappa'],
                'log10M1'   : file_tpcf_h5py[key].attrs['log10M1'],
                'log10Mmin' : file_tpcf_h5py[key].attrs['log10Mmin'],
                'r'         : file_tpcf_h5py[key]['r'][...],
                'xi'        : file_tpcf_h5py[key]['xi'][...],
            })
            _out_list.append(df)
            
        df_all = pd.concat(_out_list)
        df_all.to_csv(
            f'{TPCF_DATAPATH}/TPCF_{flag}.csv',
            index=False,
        )
        
        file_tpcf_h5py.close()

# slice_TPCF_arrays_at_r_interval()
# make_TPCF_HDF_files_arrays_at_sliced_r()
# hdf5_to_csv()
