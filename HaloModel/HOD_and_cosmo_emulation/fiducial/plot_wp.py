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



FIDUCIAL_DATA_PATH  = Path("/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data")
wp_fname            = Path(FIDUCIAL_DATA_PATH / "wp_from_sz_fiducial.hdf5")
xi_fname            = Path(FIDUCIAL_DATA_PATH / "tpcf_r_fiducial.hdf5")

def get_wp_from_xiR(r_perp, xiR, r):
    xiR_func = ius(
        r,
        xiR,
        )
    pi_upper_lim    = np.sqrt(np.max(r.reshape(-1,1)**2 - r_perp.reshape(1,-1)**2))
    pi_max          = np.min([pi_upper_lim, 105.0]) #- 10
    rpara_integral  = np.linspace(0, pi_max, int(1000))
    wp_fromxiR = 2.0 * simps(
        xiR_func(
            np.sqrt(r_perp.reshape(-1, 1)**2 + rpara_integral.reshape(1, -1)**2)
        ),
        rpara_integral,
        axis=-1,
    )
    return wp_fromxiR

def plot_wp_from_sz_and_wp_from_xiR():
        
    fig = plt.figure(figsize=(10, 5))
    gs  = gridspec.GridSpec(1, 2,)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    wp_file = h5py.File(wp_fname, "r")
    xi_file = h5py.File(xi_fname, "r")

    r_perp_mean = wp_file["rp_mean"][:]
    N_samples = len([key for key in wp_file.keys() if key.startswith("Abacus")])
    wp_from_xiR_array = np.zeros((N_samples, len(r_perp_mean)))

    for ii, key in enumerate(wp_file.keys()):
        if not key.startswith("Abacus"):
            continue 

        wp_group    = wp_file[key]
        xi_group    = xi_file[key]

        r_perp      = wp_group["r_perp"][:]
        w_p         = wp_group["w_p"][:]

        r_bincentre = xi_group["r"][:]
        xiR         = xi_group["xi"][:]
        wp_fromxiR  = get_wp_from_xiR(r_perp_mean, xiR, r_bincentre)

        wp_from_xiR_array[ii] = wp_fromxiR

        # Plot wp data and wp from xiR 
        ax0.plot(r_perp,r_perp * w_p,"x-",lw=0.1,ms=0.7,c="blue",alpha=0.5,)
        ax1.plot(r_perp,r_perp * wp_fromxiR,"x-",lw=0.1,ms=0.7,c="blue",alpha=0.5,)

    # Make label for wp data and wp from xiR
    ax0.plot([],"x-",lw=0.2,ms=1.5,c="blue",alpha=0.5,label=r'data')
    ax1.plot([],"x-",lw=0.2,ms=1.5,c="blue",alpha=0.5,label=r'data')

    # Get mean and std of both wp's 
    wp_mean     = wp_file["wp_mean"][:]
    wp_stddev   = wp_file["wp_stddev"][:]
    wp_from_xiR_mean     = np.mean(wp_from_xiR_array, axis=0)
    wp_from_xiR_stddev   = np.std(wp_from_xiR_array, axis=0)
    wp_file.close()
    xi_file.close()
    
    # Plot mean of wp with errorbars from data and from xiR 
    ax0.errorbar(r_perp_mean,r_perp_mean * wp_mean,yerr=r_perp_mean * wp_stddev,lw=0,elinewidth=0.8,marker='o',markersize=2,c="red",label=r'Mean',)
    ax1.errorbar(r_perp_mean,r_perp_mean * wp_from_xiR_mean,yerr=r_perp_mean * wp_from_xiR_stddev,lw=0,elinewidth=0.8,marker='o',markersize=2,c="red",label=r'Mean',)

    ax0.set_xscale("log")
    ax1.set_xscale("log")

    ax0.legend()
    ax0.set_xlabel(r'$r_{\perp} / (h^{-1}\mathrm{Mpc})$')
    ax0.set_ylabel(r'$r_{\perp} w_{p}(r_{\perp})$')
    ax0.set_title(r'$w_p$ from $r_{\perp}$ and $s_z$')
    ax1.legend()
    ax1.set_xlabel(r'$r_{\perp} / (h^{-1}\mathrm{Mpc})$')
    ax1.set_ylabel(r'$r_{\perp} w_{p}(r_{\perp})$')
    ax1.set_title(r'$w_p$ from $\xi(r)$')

    outfig = Path("./wp_plot_individual_and_mean.png")
    if outfig.exists():
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(
            outfig,
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=200,
        )

def plot_wp_from_sz(savefig=False):
        
    fig = plt.figure(figsize=(8, 6))
    gs  = gridspec.GridSpec(1, 1,)
    ax0 = plt.subplot(gs[0])

    wp_file = h5py.File(wp_fname, "r")

    r_perp_mean = wp_file["rp_mean"][:]
    lw_ = 0.2
    ms_ = 1
    alpha_ = 0.5


    for key in wp_file.keys():
        if not key.startswith("Abacus"):
            continue 

        wp_group    = wp_file[key]
        r_perp      = wp_group["r_perp"][:]
        w_p         = wp_group["w_p"][:]

        # Plot wp data and wp from xiR 
        ax0.plot(
            r_perp,
            r_perp * w_p,
            "x-",
            lw=lw_,
            ms=ms_,
            c="blue",
            alpha=alpha_,
            )

    # Make label for wp data and wp from xiR
    ax0.plot(
        [],
        "x-",
        lw=lw_*2,
        ms=ms_,
        c="blue",
        alpha=alpha_*2,
        label=r'Individual'
        )

    # Get mean and std of both wp's 
    wp_mean     = wp_file["wp_mean"][:]
    wp_stddev   = wp_file["wp_stddev"][:]
    wp_file.close()
    
    # Plot mean of wp with errorbars from data and from xiR 
    ax0.errorbar(
        r_perp_mean,
        r_perp_mean * wp_mean,
        yerr=r_perp_mean * wp_stddev,
        lw=0,
        elinewidth=1.0,
        marker='o',
        markersize=3,
        # barsabove=True,
        capsize=2,
        c="red",
        label=r'Mean',
        )

    ax0.set_xscale("log")

    ax0.legend()
    ax0.set_xlabel(r'$r_{\perp} \quad [h^{-1}\mathrm{Mpc}]$')
    ax0.set_ylabel(r'$r_{\perp} w_{p}(r_{\perp}) \quad [h^{-2}\mathrm{Mpc}^2]$')
    # ax0.set_title(r'$w_p$ from $r_{\perp}$ and $s_z$')

    figname_stem = "wp_data_vector"
    outfig_png = Path(f"thesis_figures/{figname_stem}.png")
    outfig_pdf = Path(f"thesis_figures/{figname_stem}.pdf")

    if not savefig:
        plt.tight_layout()
        plt.show()
    else:
        print(f"Storing figure {outfig_png}")
        plt.savefig(
            outfig_png,
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=200,
        )
        print(f"Storing figure {outfig_pdf}")
        plt.savefig(
            outfig_pdf,
            bbox_inches="tight",
            pad_inches=0.05,
        )


# plot_wp_from_sz(savefig=True)