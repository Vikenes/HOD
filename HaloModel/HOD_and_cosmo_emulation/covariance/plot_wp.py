import numpy as np
import sys
import h5py 
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
# matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{physics}'
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)

fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(1, 1,)
ax0 = plt.subplot(gs[0])
ax0.set_xscale("log")

D13_BASE_PATH       = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"
D13_EMULATION_PATH  = Path(f"{D13_BASE_PATH}/emulation_files/fiducial_data")

OUTPATH  = Path(D13_EMULATION_PATH / "wp_data")
outfile  = Path(OUTPATH / "wp_from_sz_fiducial_ng_fixed.hdf5")

wp_file = h5py.File(outfile, "r")


for key in wp_file.keys():
    if not key.startswith("Abacus"):
        continue 
    wp_group = wp_file[key]
    # Load galaxy positions for node from halo catalogue

    r_perp = wp_group["r_perp"][:]
    w_p    = wp_group["w_p"][:]

    ax0.plot(
    r_perp,
    r_perp * w_p,
    lw=0.2,
    c="gray",
    alpha=0.3,
    # label=r'from $\xi^R_{gg}(r)$'
)
    

wp_mean = wp_file["wp_mean"][:]
wp_stddev = wp_file["wp_stddev"][:]
rp_mean = wp_file["rp_mean"][:]
wp_file.close()

ax0.errorbar(
    rp_mean,
    rp_mean * wp_mean,
    yerr=rp_mean * wp_stddev,
    lw=0,
    elinewidth=0.8,
    marker='o',
    markersize=2,
    c="red",
    label='Mean',
)


ax0.legend()
ax0.set_xlabel(r'$r_{\perp} / (h^{-1}\mathrm{Mpc})$')
ax0.set_ylabel(r'$r_{\perp} w_{p}(r_{\perp})$')
plt.savefig(
    "projected.pdf",
    bbox_inches="tight",
    pad_inches=0.05,
)

