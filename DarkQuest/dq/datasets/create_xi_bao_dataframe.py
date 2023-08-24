import numpy as np
from pathlib import Path
import pandas as pd
from dq import snapshot_to_redshift,convert_run_to_cosmo_number,Cosmology

HALO_DATA_DIR = Path("/cosma7/data/dp004/dc-cues1/DarkQuest/")
OLD_HALO_DATA_DIR = Path("/cosma6/data/dp004/dc-cues1/DarkQuest/")

test_indices =  list(range(81,91)) + [0]
val_indices = list(range(91,101))

def read_halo_data(r_max):
    r_xi =  np.loadtxt(OLD_HALO_DATA_DIR / "xi/separation.dat")
    xi_hh_data = np.load(HALO_DATA_DIR / "halo_data/xihh_model_v2.npy")
    xi_hh_fiducial = np.load(HALO_DATA_DIR / "halo_data/xihh_fiducial_model_v2.npy")
    xi_hh = np.zeros((xi_hh_data.shape[0]+1, xi_hh_data.shape[1], xi_hh_data.shape[2], xi_hh_data.shape[3]))
    xi_hh[0,...] = xi_hh_fiducial
    xi_hh[1:,...] = xi_hh_data
    lognh_table = np.loadtxt(OLD_HALO_DATA_DIR / "xi/log10density_table.dat")
    r_mask = (r_xi < r_max)
    xi_hh = xi_hh[...,r_mask]
    r_xi = r_xi[r_mask]
    return r_xi, xi_hh, lognh_table


def get_full_df(r_max=160., multi_output=False):
    r_xi, xi_hh_data, lognh_table = read_halo_data(r_max=r_max)
    n_runs, n_snapshots, n_logn, n_r = xi_hh_data.shape
    cosmo_number = [convert_run_to_cosmo_number(run) for run in range(1,len(xi_hh_data)+1)]
    cosmo_number2run = dict(zip(cosmo_number, range(1,len(xi_hh_data))))
    cosmo_number2run[0] = 0 # Planck
    flat_xi_hh_data = xi_hh_data.reshape((-1, len(r_xi)))
    cosmo_params = pd.read_csv('/cosma6/data/dp004/dc-cues1/emulator/cosmological_parameters.dat', sep= ' ')
    cosmo_numbers = [cosmo_number2run[c] for c in range(len(xi_hh_data))]
    cosmo_params = cosmo_params.loc[cosmo_numbers].reset_index(drop=True).reset_index()
    redshifts = [snapshot_to_redshift(s) for s in range(n_snapshots)]
    if multi_output:
        cosmo_params = cosmo_params.loc[cosmo_params.index.repeat(n_snapshots*n_logn)]
        cosmo_params = cosmo_params.reset_index(drop=True)
        flat_df = pd.DataFrame(flat_xi_hh_data,
                columns=[f'xi_{i}' for i in range(len(flat_xi_hh_data.T))])
        df = cosmo_params.join(flat_df)
        sampling = len(df) // len(lognh_table)
        logn_bin = list(range(len(lognh_table))) * sampling
        df['logn_bin'] = logn_bin
        df['logn1'] = lognh_table[df['logn_bin'].values][:,0]
        df['logn2'] = lognh_table[df['logn_bin'].values][:,1]
        redshifts = list(np.repeat(redshifts,len(lognh_table)))*n_runs
        df['redshift'] = redshifts
    else:
        cosmo_params = cosmo_params.loc[cosmo_params.index.repeat(n_snapshots*n_logn*n_r)]
        cosmo_params = cosmo_params.reset_index(drop=True)
        flat_df = pd.DataFrame(flat_xi_hh_data.reshape(-1),
                columns=[f'xi'],
        )
        df = cosmo_params.join(flat_df)
        sampling = len(df) // n_r
        r = list(r_xi) * sampling
        df['r'] = r
        logn_bin = list(np.repeat(list(range(len(lognh_table))), n_r))* n_runs * n_snapshots
        df['logn_bin'] = logn_bin
        df['logn1'] = lognh_table[df['logn_bin'].values][:,0]
        df['logn2'] = lognh_table[df['logn_bin'].values][:,1]
        redshifts = list(np.repeat(redshifts,len(lognh_table)*n_r))*n_runs
        df['redshift'] = redshifts

    #df = df.dropna()
    df = df.fillna(0.)
    return df


if __name__ == '__main__':
    df = get_full_df(multi_output=True)
    df['index'] = df['index'].astype(int)

    train_df = df[~df['index'].isin(test_indices + val_indices)]
    train_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_xi_bao_multioutput.csv',index=False,)
    test_df = df[df['index'].isin(test_indices)]
    test_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_xi_bao_multioutput.csv',index=False,)
    val_df = df[df['index'].isin(val_indices)]
    val_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_xi_bao_multioutput.csv',index=False,)

    df = get_full_df(multi_output=False)
    
    train_df = df[~df['index'].isin(test_indices + val_indices)]
    train_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_xi_bao.csv',index=False,)
    test_df = df[df['index'].isin(test_indices)]
    test_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_xi_bao.csv',index=False,)
    val_df = df[df['index'].isin(val_indices)]
    val_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_xi_bao.csv',index=False,)




