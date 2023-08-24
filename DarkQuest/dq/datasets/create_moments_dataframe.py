import numpy as np
from pathlib import Path
import pandas as pd
from dq import snapshot_to_redshift,convert_run_to_cosmo_number,Cosmology, Measured

HALO_DATA_DIR = Path("/cosma7/data/dp004/dc-cues1/DarkQuest/")

test_indices =  list(range(81,91)) + [0]
val_indices = list(range(91,101))

def get_full_df(r_max=70.):
    measured = Measured()
    runs = np.arange(1,102)
    snapshots = np.arange(1,21)
    logn = np.arange(-2.5,-6.5,-0.5)
    dfs = []
    for run in runs:
        for snapshot in snapshots:
            for logn1 in logn:
                for logn2 in logn:
                    if logn2 <= logn1:
                        redshift = snapshot_to_redshift(snapshot)
                        cosmo = Cosmology.from_run(run=run)
                        try:
                            df = measured.moments_hh(
                                    run=run,snapshot=snapshot,logn1=logn1, logn2=logn2
                            )
                            df = df[df['r'] <= r_max]
                            df['wb'] = cosmo.wb0
                            df['wc'] = cosmo.wc0
                            df['Ol'] = cosmo.Ode0
                            df['lnAs'] = cosmo.lnAs
                            df['ns'] = cosmo.n_s
                            df['w'] = cosmo.w0
                            df['run'] = run
                            df['logn1'] = logn1
                            df['logn2'] = logn2
                            df['redshift'] = redshift
                            dfs.append(df)
                        except:
                            print(f"Missing run = {run}, snapshot = {snapshot}, logn1 = {logn1}, logn2 = {logn2}")
    return pd.concat(dfs)

def pivot_r(df, column='v_r'):
    df = df.drop(columns=['skew_r','skew_rt', 'kurtosis_r', 'kurtosis_t', 'kurtosis_rt', 'sigma_r', 'sigma_t'])
    r = sorted(df["r"].unique())
    r_mapping = {r[i]: i for i in range(len(r))}
    df["r_value"] = df["r"].apply(lambda x: f"r_{r_mapping[x]}")
    df = df.fillna(0.)
    df = pd.pivot_table(
        df,
        values=column,
        index=[col for col in df.columns if col not in (column, "r", "r_value")],
        columns="r_value",
    ).reset_index(drop=False)
    columns_ordered = [col for col in df.columns if not col.startswith('r_')]
    for i in range(len(r)):
        columns_ordered += [f'r_{i}']
    return df[columns_ordered]


if __name__ == '__main__':
    df = get_full_df()
    df['index'] = df['index'].astype(int)
    df = df.replace(np.nan, 0).replace(np.inf,0)
    train_df = df[~df['run'].isin(test_indices + val_indices)]
    train_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_moments.csv',index=False,)
    test_df = df[df['run'].isin(test_indices)]
    test_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_moments.csv',index=False,)
    val_df = df[df['run'].isin(val_indices)]
    test_df.to_csv('/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_moments.csv',index=False,)

    train_df =  pd.read_csv(HALO_DATA_DIR / 'dataframes/train_moments.csv')
    multi_train_df = pivot_r(train_df)
    multi_train_df.to_csv(HALO_DATA_DIR / 'dataframes/train_moments_multioutput.csv', index=False,)

    test_df =  pd.read_csv(HALO_DATA_DIR / 'dataframes/test_moments.csv')
    multi_test_df = pivot_r(test_df)
    multi_test_df.to_csv(HALO_DATA_DIR / 'dataframes/test_moments_multioutput.csv', index=False,)

    val_df =  pd.read_csv(HALO_DATA_DIR / 'dataframes/val_moments.csv')
    multi_val_df = pivot_r(val_df)
    multi_val_df.to_csv(HALO_DATA_DIR / 'dataframes/val_moments_multioutput.csv', index=False,)


