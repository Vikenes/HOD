import numpy as np
from pathlib import Path
import h5py 

DATAPATH    = Path(f"/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data/MGGLAM")
wp_file     = h5py.File(Path(DATAPATH / "wp_from_sz_fiducial_ng_fixed.hdf5"), "r")

rperp_mean  = wp_file["rp_mean"][:]
N_rperp     = rperp_mean.shape[0]
N_sims      = int(100)

wp_array    = np.zeros([N_rperp, N_sims])

for i in range(N_sims):
    key             = f"box{i+1}"
    wp_group        = wp_file[key]
    wp_array[:, i]  = wp_group["w_p"][:]

wp_file.close()

cov_data         = np.cov(wp_array)
corr_data        = np.corrcoef(wp_array)

cov_file    = Path(f'{DATAPATH}/cov_wp_fiducial')
corr_file   = Path(f'{DATAPATH}/corrcoef_wp_fiducial')

if not cov_file.exists():    
    print(f"Saving {cov_file}")
    np.save(
        cov_file,
        cov_data,
    )
else:
    print(f"{cov_file} already exists.")

if not corr_file.exists():
    print(f"Saving {corr_file}")
    np.save(
        corr_file,
        corr_data,
    )
else:
    print(f"{corr_file} already exists.")