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
r_binedge = np.geomspace(0.5, 60, 30)


for dataflag in ['real', 'redshift']:
    _lst = []
    for phidx in range(0, 24+1):
        if (dataflag == 'real'):
            pos = np.load(f'/mn/stornext/d8/data/chengzor/void_abacussummit/data/multi/HOD_LOWZ_gal_pos_cos0_ph{phidx}_z0.25.npy')
        if (dataflag == 'redshift'):
            pos = np.load(f'/mn/stornext/d8/data/chengzor/void_abacussummit/data/multi/HOD_LOWZ_gal_pos_zspace_cos0_ph{phidx}_z0.25.npy')

        edges = (r_binedge,)
        results_wp = wp(
            boxsize=box_size, 
            pimax=200.0, 
            nthreads=64, 
            binfile=r_binedge, 
            X=pos[:, 0],
            Y=pos[:, 1],
            Z=pos[:, 2],
            output_rpavg=True,
        )
        # r_perp = results_wp['rmin']
        r_perp = results_wp['rpavg']
        w_p = results_wp['wp']
        _lst.append(w_p)
    wp_all = np.array(_lst)
    wp_mean = np.mean(
        wp_all,
        axis=0,
    )
    wp_stddev = np.std(
        wp_all,
        axis=0,
    )
    np.savetxt(
        f'./data/wp_{dataflag}.dat',
        np.array([r_perp, wp_mean, wp_stddev]).T,
    )
