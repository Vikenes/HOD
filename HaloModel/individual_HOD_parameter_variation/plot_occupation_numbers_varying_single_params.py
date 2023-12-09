import numpy as np 
import matplotlib.pyplot as plt 
import os 
from scipy.special import erf 
from pathlib import Path
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 14})
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

N = 100000
logM  = np.linspace(11, 15, N)

colors = ['r', 'g', 'b', 'k', 'm', 'c', 'gray', 'orange', 'purple', 'brown', 'pink']

kappa       = 0.51 
alpha       = 0.9168 
sigma_logM  = 0.6915 
# log10_Mmin  = 13.62 # Fiducial value
log10_Mmin  = 13.5228 # Gives ng=2.174e-4 as used for emulation
log10M1     = 14.42

XLABEL = r"$\log(M / (\mathrm{M}_\odot\, h^{-1})) $"

def plot_varying_sigma_logM():
    fig, ax = plt.subplots(figsize=(10, 8))
    sigma_logM_array = np.linspace(0.1, 1, 10)
    for i, sigma_logM in enumerate(sigma_logM_array):
        nc = N_c(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax.plot(logM, nc,       color=colors[i], label=rf"$\sigma_{{\log M}}={sigma_logM:.1f}$")
    fig.suptitle(r"Varying $\sigma_{\log M}$")
    # ax.set_xlabel(r"$\log M$")
    ax.set_xlabel(XLABEL)

    ax.set_ylabel(r"$\langle N_c \rangle$")
    ax.legend(loc="best")
    return fig 


def plot_varying_log10_Mmin():
    fig, axes = plt.subplots(1,2, figsize=(17, 8))
    ax0 = axes[0]
    ax1 = axes[1]
    log10_Mmin_array = np.linspace(12, 14, 11)
    # ax0.plot([],[], label=r"$N_c$", color='k')
    # ax1.plot([],[], '--', label=r"$N_s$", color='k')
    for i, log10_Mmin in enumerate(log10_Mmin_array):
        nc = N_c(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ns = N_s(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax0.plot(logM, nc, color=colors[i], label=rf"$\log_{{10}} M_\mathrm{{min}}={log10_Mmin:.1f}$")
        ax1.plot(logM, ns, color=colors[i], label=rf"$\log_{{10}} M_\mathrm{{min}}={log10_Mmin:.1f}$")
        # ls = lambda_sat(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        # ax1.plot(logM, ls, '--', color=colors[i], label=rf"$\log_{{10}} M_\mathrm{{min}}={log10_Mmin:.1f}$")


    fig.suptitle(r"Varying $\log_{10} M_\mathrm{min}$")
    ax0.set_xlabel(XLABEL)
    ax1.set_xlabel(XLABEL)
    ax1.set_yscale('log')
    ax0.set_ylabel(r"$\langle N_c \rangle$")
    ax1.set_ylabel(r"$\langle N_s \rangle$")
    ax1.set_ylim(1e-4, 7)
    ax0.set_title(r"$\langle N_c \rangle$")
    ax1.set_title(r"$\langle N_s \rangle$")
    ax10 = ax0.get_xlim()[0]
    ax0.set_xlim(ax10, 15.5)
    ax0.legend(loc='lower right', fontsize=12)
    ax1.legend(loc='upper left')
    return fig 

def plot_varying_log10M1():
    fig, ax1 = plt.subplots(figsize=(10, 8))
    log10M1_array = np.linspace(12, 16, 11)
    for i, log10M1 in enumerate(log10M1_array):
        ns = N_s(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax1.plot(logM, ns, color=colors[i], label=rf"$\log_{{10}} M_1={log10M1:.1f}$")
    fig.suptitle(r"Varying $\log_{10} M_1$")
    ax1.set_xlabel(XLABEL)
    ax1.set_ylabel(r"$\langle N_s \rangle$")
    ax1.set_yscale('log')
    ax1.set_ylim(0.5e-4, 1e3)
    ax1.set_xlim(12.5,15.1)
    ax1.legend(loc='upper left')
    return fig 

def plot_varying_kappa():
    fig, ax1 = plt.subplots(figsize=(10, 8))
    kappa_array = np.linspace(0.1, 1.1, 11)
    for i, kappa in enumerate(kappa_array):
        ns = N_s(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax1.plot(logM, ns, color=colors[i], label=rf"$\kappa={kappa:.1f}$")
    fig.suptitle(r"Varying $\kappa$")
    ax1.set_xlabel(XLABEL)
    ax1.set_ylabel(r"$\langle N_s \rangle$")
    ax1.set_yscale('log')
    ax1.set_ylim(1e-5, 8)
    ax1_lims = ax1.get_xlim()
    ax1.set_xlim(12,ax1_lims[1])
    ax1.legend(loc='lower right')
    return fig 

def plot_varying_alpha():
    fig, ax1 = plt.subplots(figsize=(10, 8))
    alpha_array = np.linspace(0.1, 1.1, 11)
    for i, alpha in enumerate(alpha_array):
        ns = N_s(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax1.plot(logM, ns, color=colors[i], label=rf"$\alpha={alpha:.1f}$")
    fig.suptitle(r"Varying $\alpha$")
    ax1.set_xlabel(XLABEL)
    ax1.set_ylabel(r"$\langle N_s \rangle$")
    ax1.set_yscale('log')
    ax1.set_ylim(0.8e-3, 6)
    ax1.set_xlim(13,15.1)
    ax1.legend(loc='lower right')
    return fig 
SAVE = False 

def plot_sigma_logM(save=SAVE):
    sigma_logM = plot_varying_sigma_logM()
    if save:
        title = Path(f"./plots/occ_varying_sigma_logM.png")
        print(f"Saving to {title}")
        sigma_logM.savefig(title, dpi=200)
    else:
        plt.show()
    sigma_logM.clf()
    
def plot_logMmin(save=SAVE):
    logMmin = plot_varying_log10_Mmin()
    if save:
        title = Path(f"./plots/occ_varying_logMmin.png")
        print(f"Saving to {title}")
        logMmin.savefig(title, dpi=200)
    else:
        plt.show()
    logMmin.clf()

def plot_logM1(save=SAVE):
    logM1 = plot_varying_log10M1()
    if save:
        title = Path(f"./plots/occ_varying_logM1.png")
        print(f"Saving to {title}")
        logM1.savefig(title, dpi=200)
    else:
        plt.show()
    logM1.clf()

def plot_kappa(save=SAVE):
    kappa = plot_varying_kappa()
    if save:
        title = Path(f"./plots/occ_varying_kappa.png")
        print(f"Saving to {title}")
        kappa.savefig(title, dpi=200)
    else:
        plt.show()
    kappa.clf()

def plot_alpha(save=SAVE):
    alpha = plot_varying_alpha()
    if save:
        title = Path(f"./plots/occ_varying_alpha.png")
        print(f"Saving to {title}")
        alpha.savefig(title, dpi=200)
    else:
        plt.show()
    alpha.clf()



def plotall(save=False):
    plot_sigma_logM(save=save)
    plot_logMmin(save=save)
    plot_logM1(save=save)
    plot_kappa(save=save)
    plot_alpha(save=save)
plotall(save=True)
# plotall(save=False)
