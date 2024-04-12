import numpy as np
from pathlib import Path
import h5py 

DATAPATH    = Path(f"/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data")
wp_file     = h5py.File(Path(DATAPATH / "wp_from_sz_small_ng_fixed.hdf5"), "r")

rperp_mean  = wp_file["rp_mean"][:]
N_rperp     = rperp_mean.shape[0]
phase_keys  = [key for key in wp_file.keys() if key.startswith("Abacus")]
N_sims      = len(phase_keys)

wp_array    = np.zeros([N_rperp, N_sims])

for i, key in enumerate(phase_keys):
    wp_group        = wp_file[key]
    wp_array[:, i]  = wp_group["w_p"][:]

wp_file.close()

cov_data         = np.cov(wp_array)
corr_data        = np.corrcoef(wp_array)

cov_file    = Path(f'{DATAPATH}/cov_wp_small')
corr_file   = Path(f'{DATAPATH}/corrcoef_wp_small')

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