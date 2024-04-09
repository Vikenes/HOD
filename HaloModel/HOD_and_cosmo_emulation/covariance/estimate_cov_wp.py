import numpy as np
from pathlib import Path
import h5py 
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
# matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{physics}'
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)


DATAPATH    = Path(f"/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data")

def compute_cov(filename = "wp_from_sz_fiducial_ng_fixed.hdf5"):
    wp_file     = h5py.File(Path(DATAPATH / filename), "r")
    rperp_mean  = wp_file["rp_mean"][:]
    N_rperp     = rperp_mean.shape[0]
    N_sims      = int(25)

    wp_array    = np.zeros([N_rperp, N_sims])

    for i in range(N_sims):
        key             = f"AbacusSummit_base_c000_ph{i:03d}"
        wp_group        = wp_file[key]
        wp_array[:, i]  = wp_group["w_p"][:]

    wp_file.close()
    cov     = np.cov(wp_array)
    corr    = np.corrcoef(wp_array)

    return cov, corr 

def plot_wp_cov(filename = "wp_from_sz_fiducial_ng_fixed.hdf5", show = True):
    wp_file     = h5py.File(Path(DATAPATH / filename), "r")
    wp_mean     = wp_file["wp_mean"][:]
    rperp_mean  = wp_file["rp_mean"][:]
    N_rperp     = rperp_mean.shape[0]
    N_sims      = int(25)
    fig, ax    = plt.subplots(figsize=(8, 6))

    for i in range(N_sims):
        key         = f"AbacusSummit_base_c000_ph{i:03d}"
        wp_group    = wp_file[key]
        wp          = wp_group["w_p"][:]
        r_perp      = wp_group["r_perp"][:]
        ax.plot(r_perp, wp, alpha=0.7, lw=0.7)

    wp_file.close()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r_{\perp}$ [Mpc/h]")
    ax.set_ylabel(r"$r_{\perp} w_p$ [Mpc/h]")
    ax.set_title("wp from HOD catalogues")
    plt.tight_layout()
    plt.show()


plot_wp_cov()

def save_cov_coeff(cov, corr, filename):
    np.save(
        f'{DATAPATH}/cov_{filename}',
        cov,
    )

    np.save(
        f'{DATAPATH}/corrcoef_{filename}',
        corr,
    )

cov_sz, corr_sz = compute_cov("wp_from_sz_fiducial_ng_fixed.hdf5")
cov_rz, corr_rz = compute_cov("wp_from_z_fiducial_ng_fixed.hdf5")
