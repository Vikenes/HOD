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


SAVE = False 



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
        if i == 0 or i == len(sigma_logM_array)-1:
            ls = "solid"
            alpha_ = 1.0 
        else:
            ls = "dashed"
            alpha_ = 0.7 
        if i == len(sigma_logM_array)-1:
            color_ = "blue"
        else:
            color_ = colors[i]
        nc = N_c(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax.plot(logM, nc, ls=ls, alpha=alpha_, color=color_, label=rf"$\sigma_{{\log M}}={sigma_logM:.1f}$")
    
    ax.set_title(r"Varying $\sigma_{\log M}$")
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(r"$\langle N_c \rangle$")
    ax.legend(loc="best")
    return fig 


def plot_varying_log10_Mmin_Nc():
    fig, ax = plt.subplots(figsize=(10, 8))
    log10_Mmin_array = np.linspace(12, 14, 11)
    for i, log10_Mmin in enumerate(log10_Mmin_array):
        if i == 0 or i == len(log10_Mmin_array)-1:
            ls = "solid"
            alpha_ = 1.0 
        else:
            ls = "dashed"
            alpha_ = 0.7 
        if i == len(log10_Mmin_array)-1:
            color_ = "blue"
        else:
            color_ = colors[i]
        nc = N_c(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax.plot(logM, nc, ls=ls, alpha=alpha_, color=color_, label=rf"$\log_{{10}} M_\mathrm{{min}}={log10_Mmin:.1f}$")

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(r"$\langle N_c \rangle$")
    # ax.set_title(r"$\langle N_c \rangle$")
    ax.set_title(r"Varying $M_\mathrm{min}$")
    ax.legend(loc='lower right', fontsize=12)
    return fig 

def plot_varying_log10_Mmin_Ns():
    fig, ax = plt.subplots(figsize=(10, 8))
    log10_Mmin_array = np.linspace(12., 14, 11)

    for i, log10_Mmin in enumerate(log10_Mmin_array):
        if i == 0 or i == len(log10_Mmin_array)-1:
            ls = "solid"
            alpha_ = 1.0 
        else:
            ls = "dashed"
            alpha_ = 0.7 
        if i == len(log10_Mmin_array)-1:
            color_ = "blue"
        else:
            color_ = colors[i]

        ns = N_s(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax.plot(logM, ns, ls=ls, alpha=alpha_, color=color_, label=rf"$\log_{{10}} M_\mathrm{{min}}={log10_Mmin:.1f}$")


    ax.set_xlabel(XLABEL)
    ax.set_yscale('log')
    ax.set_ylabel(r"$\langle N_s \rangle$")
    ax.set_ylim(1e-4, 7)
    ax.set_title(r"Varying $M_\mathrm{min}$")
    ax.legend(loc='upper left')
    return fig 

def plot_varying_log10M1():
    fig, ax1 = plt.subplots(figsize=(10, 8))
    log10M1_array = np.linspace(12, 16, 11)
    for i, log10M1 in enumerate(log10M1_array):
        if i == 0 or i == len(log10M1_array)-1:
            ls = "solid"
            alpha_ = 1.0
        else:
            ls = "dashed"
            alpha_ = 0.7
        if i == len(log10M1_array)-1:
            color_ = "blue"
        else:
            color_ = colors[i]

        ns = N_s(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax1.plot(logM, ns, ls=ls, alpha=alpha_, color=color_, label=rf"$\log_{{10}} M_1={log10M1:.1f}$")

    ax1.set_title(r"Varying $ M_1$")
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
        if i == 0 or i == len(kappa_array)-1:
            ls = "solid"
            alpha_ = 1.0
        else:
            ls = "dashed"
            alpha_ = 0.7
        if i == len(kappa_array)-1:
            color_ = "blue"
        else:
            color_ = colors[i]

        ns = N_s(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax1.plot(logM, ns, ls=ls, alpha=alpha_, color=color_, label=rf"$\kappa={kappa:.1f}$")
    ax1.set_title(r"Varying $\kappa$")
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
        if i == 0 or i == len(alpha_array)-1:
            ls = "solid"
            alpha_ = 1.0
        else:
            ls = "dashed"
            alpha_ = 0.7
        if i == len(alpha_array)-1:
            color_ = "blue"
        else:
            color_ = colors[i]

        ns = N_s(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax1.plot(logM, ns, ls=ls, alpha=alpha_, color=color_, label=rf"$\alpha={alpha:.1f}$")
    ax1.set_title(r"Varying $\alpha$")
    ax1.set_xlabel(XLABEL)
    ax1.set_ylabel(r"$\langle N_s \rangle$")
    ax1.set_yscale('log')
    ax1.set_ylim(0.8e-3, 6)
    ax1.set_xlim(13,15.1)
    ax1.legend(loc='lower right')
    return fig 

def plot_sigma_logM(save=SAVE):
    sigma_logM = plot_varying_sigma_logM()
    if save:
        title = Path(f"./plots/occ_varying_sigma_logM.png")
        print(f"Saving to {title}")
        sigma_logM.tight_layout()
        sigma_logM.savefig(title, dpi=200)
        sigma_logM.clf()
    else:
        plt.show()
        plt.close()

    
def plot_logMmin_Nc(save=SAVE):
    logMmin = plot_varying_log10_Mmin_Nc()
    if save:
        title = Path(f"./plots/occ_varying_logMmin_Nc.png")
        print(f"Saving to {title}")
        logMmin.savefig(title, dpi=200)
        logMmin.clf()
    else:
        plt.show()
        plt.close()

def plot_logMmin_Ns(save=SAVE):
    logMmin = plot_varying_log10_Mmin_Ns()
    if save:
        title = Path(f"./plots/occ_varying_logMmin_Ns.png")
        print(f"Saving to {title}")
        logMmin.savefig(title, dpi=200)
        logMmin.clf()
    else:
        plt.show()
        plt.close()


def plot_logM1(save=SAVE):
    logM1 = plot_varying_log10M1()
    if save:
        title = Path(f"./plots/occ_varying_logM1.png")
        print(f"Saving to {title}")
        logM1.savefig(title, dpi=200)
        logM1.clf()
    else:
        plt.show()
        plt.close()


def plot_kappa(save=SAVE):
    kappa = plot_varying_kappa()
    if save:
        title = Path(f"./plots/occ_varying_kappa.png")
        print(f"Saving to {title}")
        kappa.savefig(title, dpi=200)
        kappa.clf()
    else:
        plt.show()
        plt.close()


def plot_alpha(save=SAVE):
    alpha = plot_varying_alpha()
    if save:
        title = Path(f"./plots/occ_varying_alpha.png")
        print(f"Saving to {title}")
        alpha.savefig(title, dpi=200)
        alpha.clf()
    else:
        plt.show()
        plt.close()




def plotall(save=False):
    plot_sigma_logM(save=save)
    plot_logMmin_Nc(save=save)
    plot_logMmin_Ns(save=save)
    plot_logM1(save=save)
    plot_kappa(save=save)
    plot_alpha(save=save)
plotall(save=True)
# plotall(save=False)
