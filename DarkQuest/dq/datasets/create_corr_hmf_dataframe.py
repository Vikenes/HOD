from pathlib import Path
import numpy as np
import pandas as pd
from dq import snapshot_to_redshift,convert_run_to_cosmo_number,Cosmology
from convert_to_multioutput import pivot_k
from scipy.signal import savgol_filter

#TODO: store mass corrected
test_indices =  list(range(81,91)) + [0]
val_indices = list(range(91,101))


DATA_DIR = Path(f'/cosma7/data/dp004/dc-cues1/DarkQuest/halo_data')
smooth = True

M_bin = np.logspace(12,16,81)
count_varied = np.load(DATA_DIR / "halocount_varied_wcorr.npy")
count_fid, _ = np.load(DATA_DIR / "halocount_fiducial_wcorr.npy")
count_hr = np.zeros((count_varied.shape[0]+1, count_varied.shape[1], count_varied.shape[2],))
count_hr[0,...] = count_fid
count_hr[1:,...] = count_varied

hr_volume = 1000.**3
M_width = M_bin[1:]-M_bin[:-1]
M_c = 0.5*(M_bin[1:] + M_bin[:-1])
dndM_hr = count_hr/hr_volume/M_width

dndM = dndM_hr

#if smooth:
#    dndM = savgol_filter(dndM, 9, 3,axis=- 1)
n_runs, n_snapshots, n_mass = dndM.shape
cosmo_number = [convert_run_to_cosmo_number(run) for run in range(1,n_runs+1)]
cosmo_number2run = dict(zip(cosmo_number, range(1,n_runs)))
cosmo_number2run[0] = 0 # Planck
redshifts = [snapshot_to_redshift(s) for s in range(n_snapshots)]

cosmo_params = pd.read_csv('/cosma6/data/dp004/dc-cues1/emulator/cosmological_parameters.dat', sep= ' ')
cosmo_numbers = [cosmo_number2run[c] for c in range(len(count_varied))]
cosmo_params = cosmo_params.loc[cosmo_numbers].reset_index(drop=True).reset_index()

cosmo_params = cosmo_params.loc[cosmo_params.index.repeat(n_snapshots*n_mass)]
cosmo_params = cosmo_params.reset_index(drop=True)
flat_df = pd.DataFrame(dndM.reshape(-1),
        columns=[f'dndM'],
)
df = cosmo_params.join(flat_df)
mass = np.array(list(np.log10(M_c)) * (n_runs-1) * n_snapshots)
df['mass'] = mass
redshifts = list(np.repeat(redshifts,n_mass))*(n_runs-1)
df['redshift'] = np.array(redshifts, dtype=np.float16)
min_value = df[df['dndM'] != 0.]['dndM'].min()
#df = df[df['mass'] < log_m_max]
df.replace(0,min_value,inplace=True)
df['dndM'] = np.log10(df['dndM'])

df['index'] = df['index'].astype(int)

train_df = df[~df['index'].isin(test_indices + val_indices)]
train_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_hmf_corr.csv',index=False,)
test_df = df[df['index'].isin(test_indices)]
test_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_hmf_corr.csv',index=False,)
val_df = df[df['index'].isin(val_indices)]
val_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_hmf_corr.csv',index=False,)
'''
train_multi_df = pivot_k(train_df, column='dndM', column_to_pivot='mass')
train_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_hmf_fit_multi_stitch.csv',index=False,)
test_multi_df = pivot_k(test_df, column='dndM', column_to_pivot='mass')
test_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_hmf_fit_multi_stitch.csv',index=False,)
val_multi_df = pivot_k(val_df, column='dndM', column_to_pivot='mass')
val_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_hmf_fit_multi_stitch.csv',index=False,)
'''
