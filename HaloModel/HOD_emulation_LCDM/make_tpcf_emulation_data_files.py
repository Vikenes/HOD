import numpy as np 
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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
TPCF_INPATH = Path(LCDM_PATH / "data_measurements")
TPCF_OUTPATH = Path(LCDM_PATH / "emulation_data")
PARAMS_INPATH = Path(LCDM_PATH / "HOD_data")

data_set_names = ["train", "test", "val"]

def slice_TPCF_arrays_at_r_interval(r_low=0.6, r_high=60):
    for flag in data_set_names:
        NPY_FILES_TRAIN = TPCF_INPATH.glob(f"TPCF_{flag}_node*115bins*.npy")

        r_common = np.load(f"{TPCF_INPATH}/TPCF_{flag}_node0_115bins_ng_fixed.npy")[0]
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


slice_TPCF_arrays_at_r_interval()