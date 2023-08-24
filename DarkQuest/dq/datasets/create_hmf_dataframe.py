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

ncount_varied_not_fid = np.load(DATA_DIR / "halocount_varied_wcorr.npy")[...,1:]
ncount_fiducial, ncount_fid_err = np.load(DATA_DIR / "halocount_fiducial_wcorr.npy")
particle_masses = np.load(DATA_DIR / "mass_list_LR.npy")


ncount_varied = np.zeros((ncount_varied_not_fid.shape[0]+1, ncount_varied_not_fid.shape[1], ncount_varied_not_fid.shape[2],))
ncount_varied[0,...] = ncount_fiducial[...,1:]
ncount_varied[1:,...] = ncount_varied_not_fid

log_m_max = 16.
M_bin = np.logspace(12,16,81)[1:]
volume = 1000.**3 #2000.**3

M_width = M_bin[1:]-M_bin[:-1]
if correction:
    N_par = M_bin.reshape(1,-1)/particle_masses.reshape(-1,1)
    M_bin_corr = (1+N_par**(-0.55))*M_bin.reshape(1,-1)
    Mwidth_corr = M_bin_corr[:,1:]-M_bin_corr[:,:-1]
    dndM =  ncount_varied/volume/Mwidth_corr[:, np.newaxis,:]
    M_bin = M_bin_corr
    M_c = 0.5*(M_bin[:,1:] + M_bin[:,:-1])
else:
    dndM = ncount_varied/volume/M_width
    M_c = 0.5*(M_bin[1:] + M_bin[:-1])
n_runs, n_snapshots, n_mass = ncount_varied.shape
cosmo_number = [convert_run_to_cosmo_number(run) for run in range(1,n_runs+1)]
cosmo_number2run = dict(zip(cosmo_number, range(1,n_runs)))
cosmo_number2run[0] = 0 # Planck
redshifts = [snapshot_to_redshift(s) for s in range(n_snapshots)]

cosmo_params = pd.read_csv('/cosma6/data/dp004/dc-cues1/emulator/cosmological_parameters.dat', sep= ' ')
cosmo_numbers = [cosmo_number2run[c] for c in range(len(ncount_varied))]
cosmo_params = cosmo_params.loc[cosmo_numbers].reset_index(drop=True).reset_index()

cosmo_params = cosmo_params.loc[cosmo_params.index.repeat(n_snapshots*n_mass)]
cosmo_params = cosmo_params.reset_index(drop=True)
flat_df = pd.DataFrame(dndM.reshape(-1),
        columns=[f'dndM'],
)
df = cosmo_params.join(flat_df)
if correction:
    mass = np.array(list(np.log10(M_c.reshape(-1))) * n_snapshots)
else:
    mass = np.array(list(np.log10(M_c)) * n_runs * n_snapshots)
df['mass'] = mass
redshifts = list(np.repeat(redshifts,n_mass))*n_runs
df['redshift'] = np.array(redshifts, dtype=np.float16)
min_value = df[df['dndM'] != 0.]['dndM'].min()
df = df[df['mass'] < log_m_max]
df.replace(0,min_value,inplace=True)
df['dndM'] = np.log10(df['dndM'])

df['index'] = df['index'].astype(int)

train_df = df[~df['index'].isin(test_indices + val_indices)]
if correction:
    train_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_hmf_corr.csv',index=False,)
else:
    train_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_hmf.csv',index=False,)
test_df = df[df['index'].isin(test_indices)]
if correction:
    test_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_hmf_corr.csv',index=False,)
else:
    test_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_hmf.csv',index=False,)
val_df = df[df['index'].isin(val_indices)]
if correction:
    val_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_hmf_corr.csv',index=False,)
else:
    val_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_hmf.csv',index=False,)

train_multi_df = pivot_k(train_df, column='dndM', column_to_pivot='mass')
if correction:
    train_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_hmf_multi_corr.csv',index=False,)
else:
    train_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_hmf_multi.csv',index=False,)
test_multi_df = pivot_k(test_df, column='dndM', column_to_pivot='mass')
if correction:
    test_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_hmf_multi_corr.csv',index=False,)
else:
    test_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_hmf_multi.csv',index=False,)
val_multi_df = pivot_k(val_df, column='dndM', column_to_pivot='mass')
if correction:
    val_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_hmf_multi_corr.csv',index=False,)
else:
    val_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_hmf_multi.csv',index=False,)

