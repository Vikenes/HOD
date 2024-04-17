import numpy as np
import sys
import h5py 
from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simps

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

fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2,)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_xscale("log")
ax1.set_xscale("log")

OUTPATH   = Path("/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data/MGGLAM")
wp_fname  = Path(OUTPATH / "wp_from_sz_fiducial_ng_fixed.hdf5")
xi_fname  = Path(OUTPATH / "tpcf_r_fiducial_ng_fixed.hdf5")

wp_file = h5py.File(wp_fname, "r")
xi_file = h5py.File(xi_fname, "r")

wp_from_xiR_lst = []

for key in wp_file.keys():
    if not key.startswith("box"):
        continue 
    wp_group = wp_file[key]
    # Load galaxy positions for node from halo catalogue

    r_perp = wp_group["r_perp"][:]
    w_p    = wp_group["w_p"][:]

    ax0.plot(
        r_perp,
        r_perp * w_p,
        "x-",
        lw=0.2,
        ms=1,
        c="blue",
        alpha=0.5,
    )

    xi_group    = xi_file[key]

    r_bincentre = xi_group["r"][:]
    xiR         = xi_group["xi"][:]

    xiR_func = ius(
        r_bincentre,
        xiR,
    )
    pi_upper_lim    = np.sqrt(np.max(r_bincentre.reshape(-1,1)**2 - r_perp.reshape(1,-1)**2))
    pi_max          = np.min([pi_upper_lim, 100.0]) #- 10
    rpara_integral = np.linspace(0, pi_max, int(1000))
    rpp = rpara_integral.reshape(1, -1)
    wp_fromxiR = 2.0 * simps(
        xiR_func(
            np.sqrt(r_perp.reshape(-1, 1)**2 + rpara_integral.reshape(1, -1)**2)
        ),
        rpara_integral,
        axis=-1,
    )

    wp_from_xiR_lst.append(wp_fromxiR)

    ax1.plot(
        r_perp,
        r_perp * wp_fromxiR,
        "x-",
        lw=0.2,
        ms=1,
        c="blue",
        alpha=0.5,
    )


ax0.plot(
    [],
    "x-",
    lw=0.2,
    ms=1.5,
    c="blue",
    alpha=0.5,
    label=r'$w_p$'
)
    

ax1.plot(
    [],
    "x-",
    lw=0.2,
    ms=1.5,
    c="blue",
    alpha=0.5,
    label=r'$w_p(\xi)$'
)

    
wp_mean = wp_file["wp_mean"][:]
wp_stddev = wp_file["wp_stddev"][:]
rp_mean = wp_file["rp_mean"][:]
wp_file.close()
xi_file.close()

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


wp_fromxiR_all = np.array(wp_from_xiR_lst)
wp_fromxiR_mean = np.mean(wp_fromxiR_all, axis=0)
wp_fromxiR_stddev = np.std(wp_fromxiR_all, axis=0)



ax1.errorbar(
    rp_mean,
    rp_mean * wp_fromxiR_mean,
    yerr=rp_mean * wp_fromxiR_stddev,
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
ax0.set_title(r'$w_p$ from $r_{\perp}$ and $s_z$')
ax1.legend()
ax1.set_xlabel(r'$r_{\perp} / (h^{-1}\mathrm{Mpc})$')
ax1.set_ylabel(r'$r_{\perp} w_{p}(r_{\perp})$')
ax1.set_title(r'$w_p$ from $\xi(r)$')


# plt.show()
figname = Path("figures/wp_plot_individual_and_mean.png")
if figname.exists():
    plt.show()
else:
    plt.savefig(
        figname,
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=200,
    )



"""
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
    ax.set_ylabel(r"$w_p$ [Mpc/h]")
    ax.set_title(r"$w_p$ from 25 HOD fiducial catalogues")
    plt.tight_layout()
    fig.savefig(f"figures/wp_samples.png",
                bbox_inches='tight',
                dpi=200)
"""
