
from dq.read_data import read_halo_data
import numpy as np
import pandas as pd
from dq import snapshot_to_redshift
from halotools.mock_observables import tpcf, mean_radial_velocity_vs_r, npairs_3d

snapshot = 18
print('z = ', snapshot_to_redshift(snapshot))
run = 101
boxsize=2000.
pos, vel, mass = read_halo_data(run=run, snapshot=snapshot)

rbins_small = np.logspace(-1,0.99,40)
rbins_large = np.arange(10.,71.0,1)
rbins = np.concatenate((rbins_small, rbins_large))

r_c = 0.5 * (rbins[1:] + rbins[:-1])
log_m_min = 12.5
log_m_max = 13.5
pos = pos[(mass>= 10**log_m_min) & (mass<= 10**log_m_max)]
xi = tpcf(pos, rbins=rbins, period=boxsize, num_threads=4)
df = pd.DataFrame({"r_c": r_c, "xi": xi})
df.to_csv('xi_mass_dq.csv', index=False)

