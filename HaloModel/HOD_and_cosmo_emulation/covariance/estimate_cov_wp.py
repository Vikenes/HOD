import numpy as np
from pathlib import Path
import h5py 
import matplotlib

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
from matplotlib import gridspec


DATAPATH    = Path(f"/mn/stornext/d5/data/vetleav/HOD_AbacusData/covariance_data_fiducial")
wp_file     = h5py.File(Path(DATAPATH / "wp_from_sz_fiducial_ng_fixed.hdf5"), "r")

rperp_mean  = wp_file["rp_mean"][:]
N_rperp     = rperp_mean.shape[0]
N_sims      = int(25)

wp_array    = np.zeros([N_rperp, N_sims])

for i in range(N_sims):
    key             = f"AbacusSummit_base_c000_ph{i:03d}"
    wp_group        = wp_file[key]
    wp_array[:, i]  = wp_group["w_p"][:]

wp_file.close()

cov         = np.cov(wp_array)
corr        = np.corrcoef(wp_array)

cov_file    = Path(f'{DATAPATH}/cov_wp_fiducial')
corr_file   = Path(f'{DATAPATH}/corrcoef_wp_fiducial')


np.save(
    f'{DATAPATH}/cov_wp_fiducial',
    cov,
)

np.save(
    f'{DATAPATH}/corrcoef_wp_fiducial',
    corr,
)
