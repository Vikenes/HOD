import numpy as np 
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import gridspec

DATAPATH     = Path(f"/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data")

cov     = np.load(f'{DATAPATH}/cov_wp_fiducial.npy')
corr    = np.load(f'{DATAPATH}/corrcoef_wp_fiducial.npy')


fig = plt.figure(figsize=(12, 5))
gs  = gridspec.GridSpec(1, 2,)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_title("unscaled colorbar")
ax1.set_title("symmetric colorbar")


im0 = ax0.imshow(corr, cmap='bwr')
im1 = ax1.imshow(corr, cmap='bwr', vmin=-1, vmax=1)
fig.colorbar(im0, fraction=0.046, pad=0.04)
fig.colorbar(im1, fraction=0.046, pad=0.04, ticks=np.linspace(-1, 1, 5))

plt.savefig(
    "corrcoef_wp.pdf",
    bbox_inches="tight",
    pad_inches=0.05,
)