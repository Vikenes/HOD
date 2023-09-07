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
TPCF_DATAPATH = Path(LCDM_PATH / "data_measurements")
TPCF_OUTPATH = Path(LCDM_PATH / "emulation_data")
PARAMS_INPATH = Path(LCDM_PATH / "HOD_data")

dataset_names = ["train", "test", "val"]


def slice_TPCF_arrays_at_r_interval(r_low=0.6, r_high=60):
    for flag in dataset_names:
        NPY_FILES_TRAIN = TPCF_DATAPATH.glob(f"TPCF_{flag}_node*115bins*.npy")

        r_common = np.load(f"{TPCF_DATAPATH}/TPCF_{flag}_node0_115bins_ng_fixed.npy")[0]
        r_mask_low_limit  = r_common > r_low
        r_mask_high_limit = r_common < r_high
        r_mask = r_mask_low_limit * r_mask_high_limit

        parameters = Path(f"HOD_parameters_ng_fixed_{flag}.csv")
        parameters_df = pd.read_csv(PARAMS_INPATH / parameters)
        _out_list = []

        for i, npy_file in enumerate(NPY_FILES_TRAIN):
            r, xi = np.load(npy_file)[:, r_mask]
            _out_list.append(pd.DataFrame({
                'sigma_logM': parameters_df['sigma_logM'].iloc[i],
                'alpha': parameters_df['alpha'].iloc[i],
                'kappa': parameters_df['kappa'].iloc[i],
                'log10M1': parameters_df['log10M1'].iloc[i],
                'log10Mmin': parameters_df['log10Mmin'].iloc[i],
                'r': r,
                'xi': xi,
            }))

        out_df = pd.concat(_out_list)
        out_df.to_csv(TPCF_OUTPATH / f"TPCF_{flag}.csv", index=False)


def hdf5_to_csv():
    for flag in dataset_names:
    ### Create csv file from hdf5 file 
    
        file_tpcf_h5py = h5py.File(f'{TPCF_OUTPATH}/TPCF_{flag}.hdf5', 'r')

        _out_list = []
        for key in file_tpcf_h5py.keys():
            _out_list.append(pd.DataFrame({
                'sigma_logM': file_tpcf_h5py[key].attrs['sigma_logM'],
                'alpha'     : file_tpcf_h5py[key].attrs['alpha'],
                'kappa'     : file_tpcf_h5py[key].attrs['kappa'],
                'log10M1'   : file_tpcf_h5py[key].attrs['log10M1'],
                'log10Mmin' : file_tpcf_h5py[key].attrs['log10Mmin'],
                'r'         : file_tpcf_h5py[key]['r'][...],
                'xi'        : file_tpcf_h5py[key]['xi'][...],
            }))
            
        df_all = pd.concat(_out_list)
        df_all.to_csv(
            f'{TPCF_OUTPATH}/_TPCF_{flag}.csv',
            index=False,
        )
        
        file_tpcf_h5py.close()


slice_TPCF_arrays_at_r_interval()