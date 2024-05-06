import numpy as np 
import matplotlib.pyplot as plt 
from scipy.special import erf 
from pathlib import Path
from HOD_and_cosmo_prior_ranges import get_fiducial_HOD_params, get_HOD_params_prior_range

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
params = {'xtick.top': True, 
          'ytick.right': True, 
          'xtick.direction': 'in', 
          'ytick.direction': 'in',
          }
plt.rcParams.update(params)


def N_c(logM, alpha, kappa, sigma_logM, logMmin, logM1):
    erf_term = erf((logM - logMmin)/sigma_logM)
    return 0.5*(1.0 + erf_term)

def lambda_sat(logM, alpha, kappa, sigma_logM, logMmin, logM1):
    frac = (10**logM - kappa * 10**logMmin) / 10**logM1
    mask = frac > 0.0
    power = np.zeros_like(frac)
    power[mask] = frac[mask] ** alpha
    return power 
   
def N_s(logM, alpha, kappa, sigma_logM, logMmin, logM1):
    nc = N_c(logM, alpha, kappa, sigma_logM, logMmin, logM1)    
    return nc * lambda_sat(logM, alpha, kappa, sigma_logM, logMmin, logM1) 

def N_g(logM, alpha, kappa, sigma_logM, logMmin, logM1):
    nc = N_c(logM, alpha, kappa, sigma_logM, logMmin, logM1)    
    ns = N_s(logM, alpha, kappa, sigma_logM, logMmin, logM1)
    return nc + ns

N = 100000
logM  = np.linspace(11.5, 15, N)

colors = ['r', 'g', 'b', 'k', 'm', 'c', 'gray', 'orange', 'purple', 'brown', 'pink']
HOD_priors = get_HOD_params_prior_range()
HOD_fiducial = get_fiducial_HOD_params()

kappa       = HOD_fiducial["kappa"]
alpha       = HOD_fiducial["alpha"]
log10M1     = HOD_fiducial["log10M1"]
sigma_logM_lst = HOD_priors["sigma_logM"] + [0.99] 

log10_Mmin_array = [13.0, 14.0]
colors      = ["blue", "red", "green"]
linestyles  = ["solid", "solid"]

def plot_varying_log10_Mmin_Nc():
    fig, ax = plt.subplots(figsize=(10, 8))
    for log10_Mmin in log10_Mmin_array:
        for jj_sigma, sigma_logM in enumerate(sigma_logM_lst):
            nc = N_c(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
            ax.plot(
                logM, 
                nc, 
                alpha=0.7,
                color=colors[jj_sigma], 
                )
    label_str = ["$\mathrm{(Min.\: prior)}$", "$\mathrm{(Max.\: prior)}$", ""]
    for ii in range(3):
        ax.plot([], [],  color=colors[ii],    label=fr"$\sigma_{{\log{{M}}}}={sigma_logM_lst[ii]:.3f}$ {label_str[ii]}")


    ax.text(11.7, 0.5, fr"$\log (M_\mathrm{{min}} / (\mathrm{{M}}_\odot\, h^{{-1}}))={log10_Mmin_array[0]:.0f}$", fontsize=16)
    ax.text(13.9, 0.35, fr"$\log (M_\mathrm{{min}} / (\mathrm{{M}}_\odot\, h^{{-1}}))={log10_Mmin_array[1]:.0f}$", fontsize=16)


    ax.set_xlabel(r"$\log(M / (\mathrm{M}_\odot\, h^{-1})) $")
    ax.set_ylabel(r"$\langle N_c \rangle$")
    ax.legend(loc='upper left')

    if not SAVE:
        plt.show()
        return 
    
    figname_stem = "Nc_log10Mmin_sigma_logM"
    figname_png = Path(f"plots/thesis_figures_HOD/{figname_stem}.png")
    figname_pdf = Path(f"plots/thesis_figures_HOD/{figname_stem}.pdf")
    print(f"Storing figure {figname_png}")
    fig.savefig(
        figname_png, 
        bbox_inches="tight", 
        dpi=200
        )
    print(f"Storing figure {figname_pdf}")
    fig.savefig(
        figname_pdf, 
        bbox_inches="tight"
        )

global SAVE
# SAVE = True
SAVE = False



plot_varying_log10_Mmin_Nc()
