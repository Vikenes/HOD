from pathlib import Path
import numpy as np
import pandas as pd
from dq import snapshot_to_redshift,convert_run_to_cosmo_number,Cosmology
from convert_to_multioutput import pivot_k

#TODO: store mass corrected
test_indices =  list(range(81,91)) + [0]
val_indices = list(range(91,101))


DATA_DIR = Path(f'/cosma7/data/dp004/dc-cues1/DarkQuest/halo_data')

def combine(resolution='LR'):
    if resolution == 'LR':
        ncount_varied_not_fid = np.load(DATA_DIR / "halocount_varied_lr.npy")[...,1:]

        ncount_fiducial, ncount_fid_err = np.load(DATA_DIR / "halocount_fiducial_lr.npy")
    elif resolution == 'HR':
        ncount_varied_not_fid = np.load(DATA_DIR / "halocount_varied_wcorr.npy")[...,1:]
        ncount_fiducial, ncount_fid_err = np.load(DATA_DIR / "halocount_fiducial_wcorr.npy")

    ncount_varied = np.zeros((ncount_varied_not_fid.shape[0]+1, ncount_varied_not_fid.shape[1], ncount_varied_not_fid.shape[2],))
    ncount_varied[0,...] = ncount_fiducial[...,1:]
    ncount_varied[1:,...] = ncount_varied_not_fid
    return ncount_varied 

def compute_dndM(counts, boxsize):
    volume = boxsize**3
    return counts/volume/M_width

log_m_max = np.log10(3.e15)
M_bin = np.logspace(12,16,81)[1:]
M_width = M_bin[1:]-M_bin[:-1]
M_c = 0.5*(M_bin[1:] + M_bin[:1])

count_lr = combine(resolution='LR')
count_hr = combine(resolution='HR')

# Stitch for each cosmology and each redshift
dndM_hr = compute_dndM(count_hr, boxsize=1000.)
dndM_lr = compute_dndM(count_lr, boxsize = 2000.)
print(dndM_hr.shape)

#mask = (dndM_hr != 0.) & (dndM_lr != 0)
#mass_mask = M_c[mask]
diff = np.abs(dndM_hr - dndM_lr)
diff[diff == 0.] = 1.e20
idx_stitch = np.argmin(diff, axis=-1)
print(idx_stitch)
print(idx_stitch.shape)
mass_stitch = M_c[idx_stitch]

stitch_dndM = np.where(
    M_c.reshape(1,1,1,-1) > mass_stitch[...,np.newaxis],
    dndM_lr,
    dndM_hr
)[0]
print(stitch_dndM.shape)

# Store
n_runs, n_snapshots, n_mass = stitch_dndM.shape
cosmo_number = [convert_run_to_cosmo_number(run) for run in range(1,n_runs+1)]
cosmo_number2run = dict(zip(cosmo_number, range(1,n_runs)))
cosmo_number2run[0] = 0 # Planck
redshifts = [snapshot_to_redshift(s) for s in range(n_snapshots)]

cosmo_params = pd.read_csv('/cosma6/data/dp004/dc-cues1/emulator/cosmological_parameters.dat', sep= ' ')
cosmo_numbers = [cosmo_number2run[c] for c in range(len(stitch_dndM))]
cosmo_params = cosmo_params.loc[cosmo_numbers].reset_index(drop=True).reset_index()

cosmo_params = cosmo_params.loc[cosmo_params.index.repeat(n_snapshots*n_mass)]
cosmo_params = cosmo_params.reset_index(drop=True)
flat_df = pd.DataFrame(stitch_dndM.reshape(-1),
        columns=[f'dndM'],
)
df = cosmo_params.join(flat_df)
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
train_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_shmf.csv',index=False,)
test_df = df[df['index'].isin(test_indices)]
test_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_shmf.csv',index=False,)
val_df = df[df['index'].isin(val_indices)]
val_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_shmf.csv',index=False,)

train_multi_df = pivot_k(train_df, column='dndM', column_to_pivot='mass')
train_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_shmf_multi.csv',index=False,)
test_multi_df = pivot_k(test_df, column='dndM', column_to_pivot='mass')
test_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_shmf_multi.csv',index=False,)
val_multi_df = pivot_k(val_df, column='dndM', column_to_pivot='mass')
val_multi_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_shmf_multi.csv',index=False,)

