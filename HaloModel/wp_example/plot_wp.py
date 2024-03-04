import numpy as np
import sys
from Corrfunc.theory.wp import wp
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{physics}'
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)

box_size = 2000.0
pimax = 120.0





fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(1, 1,)
ax0 = plt.subplot(gs[0])
ax0.set_xscale("log")

r_perp_real, wp_real, wp_real_stddev = np.loadtxt(
    f'./data/wp_real.dat',
    unpack=True,
)
ax0.errorbar(
    r_perp_real,
    r_perp_real * wp_real,
    yerr=r_perp_real * wp_real_stddev,
    lw=0,
    elinewidth=0.8,
    marker='o',
    markersize=2,
    label='sim, real',
)
r_perp_redshift, wp_redshift, wp_redshift_stddev = np.loadtxt(
    f'./data/wp_redshift.dat',
    unpack=True,
)
ax0.errorbar(
    r_perp_redshift,
    r_perp_redshift * wp_redshift,
    yerr=r_perp_redshift * wp_redshift_stddev,
    lw=0,
    elinewidth=0.8,
    marker='x',
    markersize=2,
    label='sim, redshift',
)


r_bincentre, xiR, xiR_stddev = np.loadtxt(
    f'/mn/stornext/d8/data/chengzor/void_abacussummit/data/xi-R-gg_LOWZ_cos0_z0.25.dat',
    unpack=True,
)
xiR_func = ius(
    r_bincentre,
    xiR,
)
rpara_integral = np.linspace(0, pimax, int(1000))
dr = rpara_integral[1] - rpara_integral[1]
wp_fromxiR = 2.0 * simps(
    xiR_func(
        np.sqrt(r_perp_redshift.reshape(-1, 1)**2 + rpara_integral.reshape(1, -1)**2)
    ),
    rpara_integral,
    axis=-1,
)
ax0.plot(
    r_perp_redshift,
    r_perp_redshift * wp_fromxiR,
    lw=1.0,
    label=r'from $\xi^R_{gg}(r)$'
)
ax0.legend()
ax0.set_xlabel(r'$r_{\perp} / (h^{-1}\mathrm{Mpc})$')
ax0.set_ylabel(r'$r_{\perp} w_{p}(r_{\perp})$')
plt.savefig(
    "projected.pdf",
    bbox_inches="tight",
    pad_inches=0.05,
)

