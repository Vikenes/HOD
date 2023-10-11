import numpy as np 
import matplotlib.pyplot as plt 
import os 
from scipy.special import erf 


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
logM  = np.linspace(12, 15, N)

colors = ['r', 'g', 'b', 'k', 'm', 'c', 'gray', 'orange', 'purple', 'brown', 'pink']

kappa       = 0.51 
alpha       = 0.917 
sigma_logM  = 0.692 
log10_Mmin  = 13.5
log10M1     = 14.42

def plot_varying_sigma_logM():
    fig, ax = plt.subplots(figsize=(10, 8))
    sigma_logM_array = np.linspace(0.1, 1, 10)
    ax.plot([],[], label=r"$N_c$", color='k')
    # ax.plot([],[], '--', label=r"$N_s$", color='k')
    for i, sigma_logM in enumerate(sigma_logM_array):
        nc = N_c(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax.plot(logM, nc,       color=colors[i], label=rf"$\sigma_{{\log M}}={sigma_logM:.1f}$")
        # ax.plot(logM, ns, '--', color=colors[i])
    fig.suptitle(r"Varying $\sigma_{\log M}$")
    ax.set_xlabel(r"$\log M$")
    ax.set_ylabel(r"$\langle N_c \rangle$")
    fig.legend(loc='upper left')
    plt.show()


def plot_varying_log10_Mmin():
    fig, axes = plt.subplots(1,2, figsize=(16, 8))
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
    ax0.set_xlabel(r"$\log M$")
    ax1.set_xlabel(r"$\log M$")
    ax1.set_yscale('log')
    ax0.set_ylabel(r"$\langle N_c \rangle$")
    ax1.set_ylabel(r"$\langle N_s \rangle$")
    ax1.set_ylim(1e-4, 7)
    ax0.set_title(r"$\langle N_c \rangle$")
    ax1.set_title(r"$\langle N_s \rangle$")

    ax0.legend(loc='lower right')
    ax1.legend(loc='lower right')
    plt.show()

def plot_varying_log10M1():
    fig, ax1 = plt.subplots(figsize=(10, 8))
    log10M1_array = np.linspace(12, 16, 11)
    for i, log10M1 in enumerate(log10M1_array):
        ns = N_s(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax1.plot(logM, ns, color=colors[i], label=rf"$\log_{{10}} M_1={log10M1:.1f}$")
    fig.suptitle(r"Varying $\log_{10} M_1$")
    ax1.set_xlabel(r"$\log M$")
    ax1.set_ylabel(r"$\langle N_s \rangle$")
    ax1.set_yscale('log')
    ax1.set_ylim(0.5e-4, 1e3)
    ax1.set_xlim(13,15.1)
    ax1.legend(loc='upper left')
    plt.show()

def plot_varying_kappa():
    fig, ax1 = plt.subplots(figsize=(10, 8))
    kappa_array = np.linspace(0.1, 1.1, 11)
    for i, kappa in enumerate(kappa_array):
        ns = N_s(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax1.plot(logM, ns, color=colors[i], label=rf"$\kappa={kappa:.1f}$")
    fig.suptitle(r"Varying $\kappa$")
    ax1.set_xlabel(r"$\log M$")
    ax1.set_ylabel(r"$\langle N_s \rangle$")
    ax1.set_yscale('log')
    ax1.set_ylim(1e-5, 8)
    ax1.legend(loc='upper left')
    plt.show()

def plot_varying_alpha():
    fig, ax1 = plt.subplots(figsize=(10, 8))
    alpha_array = np.linspace(0.1, 1.1, 11)
    for i, alpha in enumerate(alpha_array):
        ns = N_s(logM, alpha, kappa, sigma_logM, log10_Mmin, log10M1)
        ax1.plot(logM, ns, color=colors[i], label=rf"$\alpha={alpha:.1f}$")
    fig.suptitle(r"Varying $\alpha$")
    ax1.set_xlabel(r"$\log M$")
    ax1.set_ylabel(r"$\langle N_s \rangle$")
    ax1.set_yscale('log')
    ax1.set_ylim(0.8e-3, 6)
    ax1.set_xlim(13,15.1)
    ax1.legend(loc='lower right')
    plt.show()

# plot_varying_sigma_logM()
# plot_varying_log10_Mmin()
# plot_varying_log10M1()
# plot_varying_kappa()
plot_varying_alpha()
exit()
# nc = N_c(logM, alpha, kappa, sigma_logM, logMmin, 14.42)
# plt.plot(logM, nc, label=r'$N_c$')

# sigma_logM = np.linspace(0.06, 1, 10) 
# for i in sigma_logM:
#     ns = N_s(logM, alpha, kappa, i, logMmin, logM1)
#     plt.plot(logM, ns, label=r'$\sigma_{\log M} = %.2f$' % i)

# logM1 = np.linspace(13.0, 15.5, 10)
# for i in logM1:
#     ns = N_s(logM, alpha, kappa, sigma_logM, logMmin, i)
    # plt.plot(logM, ns, label=r'$\log M_1 = %.2f$' % i)

# kappa = np.linspace(0.1, 2, 10)
# for i in kappa:
#     ns = N_s(logM, alpha, i, sigma_logM, logMmin, logM1)
#     plt.plot(logM, ns, label=r'$\kappa = %.2f$' % i)


alpha = np.linspace(0.1, 1.5, 10)
for i in alpha:
    ns = N_s(logM, i, kappa, sigma_logM, logMmin, logM1)
    plt.plot(logM, ns, label=r'$\alpha = %.2f$' % i)


plt.vlines(np.log10(kappa * 10**logMmin), 1e-3, 1e3, linestyle='--', color='k')

# plt.xlim(np.log10(kappa * 10**logMmin) - 0.1, logM[-1])

plt.xlabel(r'$\log M$')
plt.yscale('log')
plt.legend()
plt.show()