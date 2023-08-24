from pathlib import Path
import numpy as np
import pandas as pd
from dq import snapshot_to_redshift,convert_run_to_cosmo_number,Cosmology
from convert_to_multioutput import pivot_k

#TODO: store mass corrected
test_indices =  list(range(81,91)) + [0]
val_indices = list(range(91,101))


DATA_DIR = Path(f'/cosma7/data/dp004/dc-cues1/DarkQuest/halo_data')
correction = False

hmf_varied = np.load(DATA_DIR / "dndM_fit_varied_HR_wcorr_signal.npy")
hmf_fid_fit = np.load(DATA_DIR / "dndM_fit_fiducial_HR_wcorr_signal.npy")
hmf_fit = np.zeros((hmf_varied.shape[0]+1, hmf_varied.shape[1], hmf_varied.shape[2],))
hmf_fit[0,...] = hmf_fid_fit
hmf_fit[1:,...] = hmf_varied


dndM= hmf_fit

M_c = np.logspace(12,16,401)
skip_factor = 4
#M_c = 0.5*(M_bin[1:] + M_bin[:-1])

dndM = dndM[...,::skip_factor]
M_c = M_c[...,::skip_factor]
print(dndM.shape)
print(M_c.shape)
n_runs, n_snapshots, n_mass = dndM.shape
cosmo_number = [convert_run_to_cosmo_number(run) for run in range(1,n_runs+1)]
cosmo_number2run = dict(zip(cosmo_number, range(1,n_runs)))
cosmo_number2run[0] = 0 # Planck
redshifts = [snapshot_to_redshift(s) for s in range(n_snapshots)]

cosmo_params = pd.read_csv('/cosma6/data/dp004/dc-cues1/emulator/cosmological_parameters.dat', sep= ' ')
cosmo_numbers = [cosmo_number2run[c] for c in range(len(dndM))]
cosmo_params = cosmo_params.loc[cosmo_numbers].reset_index(drop=True).reset_index()

cosmo_params = cosmo_params.loc[cosmo_params.index.repeat(n_snapshots*n_mass)]
cosmo_params = cosmo_params.reset_index(drop=True)
flat_df = pd.DataFrame(dndM.reshape(-1),
        columns=[f'dndM'],
)
df = cosmo_params.join(flat_df)
mass = np.array(list(np.log10(M_c)) * (n_runs) * n_snapshots)
df['mass'] = mass
redshifts = list(np.repeat(redshifts,n_mass))*(n_runs)
df['redshift'] = np.array(redshifts, dtype=np.float16)
min_value = df[df['dndM'] != 0.]['dndM'].min()
#df = df[df['mass'] < log_m_max]
df.replace(0,min_value,inplace=True)
df['dndM'] = np.log10(df['dndM'])

df['index'] = df['index'].astype(int)

train_df = df[~df['index'].isin(test_indices + val_indices)]
train_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_hmf_fit_skip.csv',index=False,)
test_df = df[df['index'].isin(test_indices)]
test_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_hmf_fit_skip.csv',index=False,)
val_df = df[df['index'].isin(val_indices)]
val_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_hmf_fit_skip.csv',index=False,)
train_multi_df = pivot_k(train_df, column='dndM', column_to_pivot='mass')
train_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_hmf_fit_multi_skip.csv',index=False,)
test_multi_df = pivot_k(test_df, column='dndM', column_to_pivot='mass')
test_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_hmf_fit_multi_skip.csv',index=False,)
val_multi_df = pivot_k(val_df, column='dndM', column_to_pivot='mass')
val_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_hmf_fit_multi_skip.csv',index=False,)

