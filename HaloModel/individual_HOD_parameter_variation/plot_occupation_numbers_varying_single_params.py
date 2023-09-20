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
    # return N_c(logM, alpha, kappa, sigma_logM, logMmin, logM1) * 
    return lambda_sat(logM, alpha, kappa, sigma_logM, logMmin, logM1)

N = 100000
logM  = np.linspace(12, 15, N)
logM1 = np.linspace(13.0, 15.5, 10)


kappa = 0.51 
alpha = 0.917 
sigma_logM = 0.692 
logMmin = 13.5
logM1 = 14.42

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