import numpy as np 
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import gridspec

DATAPATH     = Path(f"/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data")

cov     = np.load(f'{DATAPATH}/cov_wp_fiducial.npy')
corr    = np.load(f'{DATAPATH}/corrcoef_wp_fiducial.npy')

# cov = np.where(cov < 0, 0, cov)
cov_inv = np.linalg.inv(cov)
cov_inv_pos = np.where(cov_inv < 0, np.nan, cov_inv)
cov_inv_neg = np.where(cov_inv > 0, np.nan, cov_inv)

print(f"{np.nanmin(cov_inv_pos)=:6.2e} | {np.nanmax(cov_inv_pos)=:6.2e}")
print(f"{np.nanmin(cov_inv_neg)=:6.2e} | {np.nanmax(cov_inv_neg)=:6.2e}")

exit()
fig = plt.figure(figsize=(12, 5))
gs  = gridspec.GridSpec(1, 2,)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_title("Covariance matrix")
ax1.set_title("Inverse covariance matrix")
im0 = ax0.imshow(cov, origin="lower", cmap='bwr')
im1 = ax1.imshow(cov_inv, origin="lower", cmap='bwr')
# im0 = ax0.imshow(cov_inv, origin="lower", cmap='bwr', vmin=-1e17, vmax=-1e14)
# im1 = ax1.imshow(cov_inv, origin="lower", cmap='bwr', vmin=1e14, vmax=1e17)

fig.colorbar(im0, fraction=0.046, pad=0.04)
fig.colorbar(im1, fraction=0.046, pad=0.04) #, ticks=np.linspace(-1, 1, 5))

# plt.savefig(
#     "/uio/hume/student-u74/vetleav/Documents/thesis/ParameterInference/figures/likelihood_tests/cov_and_cov_inverse.png",
#     bbox_inches="tight",
#     dpi=200,
#     pad_inches=0.05,
# )